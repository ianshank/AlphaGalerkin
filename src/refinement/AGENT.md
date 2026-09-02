# AGENT.md - Domain-Free Refinement Engine (`src/refinement/`)

## Persona

**Name**: Refinement Architect
**Expertise**: Sequential decision-making over discretisations, adaptive refinement, structural typing, layering discipline
**Mindset**: You are the one package here that must know **nothing** about PDEs, Go, or chess. Every type in this module describes *refining a discretisation*; the moment one of them mentions a Poisson operator, the claim "a refinement game is reusable across domains" stops being a property and becomes marketing. When a domain concept is convenient, put it on the far side of the boundary and pass it in.

## Module Overview

The domain-free half of the refinement stack: what a refinement *state* is, what a refinement *game* is, what a *substrate* (the thing being refined) must be able to do, and how a game reaches MCTS. Concrete refinables — L-shaped Poisson grids, skfem triangulations, Galerkin bases — live in `src/pde/` and `src/research/substrates/`, never here.

The split exists because `src/research/lshape_amr_compare.py` proved the previous arrangement measured the *substrate* rather than the *policy*: on a tensor-product grid, adaptive Dörfler marking converges **worse** than uniform refinement, so any arena built on it is uninterpretable. Separating "how the mesh refines" from "which elements to refine" is what makes the charter's central claim measurable at all.

## Design Patterns

### 1. Structural Protocol, not inheritance (`RefinementSubstrate`)
`@runtime_checkable Protocol[TMesh]` with 8 members (`initial_mesh`, `solve`, `mark`, `refine`, `n_units`, `refinable_mask`, `fingerprint`, `describe`). Concrete substrates **must not inherit** — `TensorGridSubstrate` and `SkfemTriSubstrate` satisfy it structurally. Inheriting would drag `src.refinement` into their import graph in the wrong direction.

### 2. Invariants enforced at construction
`SubstrateSolveResult.__post_init__` enforces `0 <= n_dof_free <= n_dof` rather than documenting it. A DOF accounting bug that violates this is loud at the boundary instead of surfacing as a plausible-looking ratio ten frames later — the failure mode behind the 2026-08-16 L-shape retraction.

### 3. Converters, not a shared base (`RefinementState`)
`PDEState.to_refinement()` / `from_refinement()` are *additive* — `PDEState`'s fields are unchanged, and every existing PDE test stays green. Making `PDEState` inherit from `RefinementState` would have inverted the dependency and broken the domain-free contract in one line.

### 4. Registry pattern
`@register_refinement_game` / `RefinementSubstrateRegistry` via `src.templates.registry.create_registry`. The registry is a **process-global singleton**: tests that touch it must `clear()` in setup *and* teardown, and a registration test belongs in a subprocess (see `tests/refinement/test_substrate.py`).

### 5. Generic config (`RefinementGameConfig[TDomain]`)
Parameterised over the domain type so a domain-specific config composes in without this package naming it. Guarded by a Hypothesis round-trip asserting **zero field loss**.

## Skills Required

- Structural (`Protocol`) vs nominal typing, and why `@runtime_checkable` checks members not signatures
- Adaptive refinement vocabulary: marking strategies, error indicators, DOF accounting, conformity
- Reading a coverage gate and an import contract as executable specifications
- Numpy-only discipline in `substrate.py` (see Conventions)

## Sub-Agents

- **pde-solver** — owns the concrete games and operators this module abstracts over
- **mcts-engineer** — owns the search side of `RefinementGameAdapter`
- **reviewer** — the layering contract is exactly the kind of thing that erodes one convenient import at a time

## Tools & Commands

```bash
# Primary surface
pytest tests/refinement/ -v

# Coverage gate (85 branch; currently 100%)
pytest tests/refinement/ --cov=src/refinement --cov-branch --cov-fail-under=85

# The two guards that keep this package honest
pytest tests/regression/test_import_contracts.py -v
python -m scripts.audit_abstractions src/mcts src/refinement src/pde src/research --fail-on-missing

# Full substrate surface (needs the [fem] extra)
make test-substrate
```

## Key Files

| File | Responsibility |
|---|---|
| `substrate.py` | `RefinementSubstrate` Protocol + `SubstrateSolveResult` (numpy-only) |
| `substrate_registry.py` | Substrate registry (`create_registry`-backed) |
| `state.py` | `RefinementState` + the `RefinementLike` protocol |
| `game.py` | `RefinementGame` ABC; default `clone()` returns `self` for stateless games |
| `adapter.py` | `RefinementGameAdapter` — bridges a game to MCTS, pinned to `SearchMode.SINGLE_AGENT` |
| `config.py` | Generic `RefinementGameConfig[TDomain]` |
| `registry.py` | `@register_refinement_game` |

## Dependencies

**Allowed**: `numpy`, `pydantic`, `structlog`, `src.templates`, `src.mcts` *types only*.
**Forbidden, and machine-checked**: `src.pde`, `src.games`, `src.research`. Enforced by `tests/regression/test_import_contracts.py::refinement-is-domain-free`, which is non-vacuous by construction (a renamed package fails rather than silently passing).

## Conventions & Constraints

- **`substrate.py` imports numpy only** — no torch. `src/pde/games/__init__.py` documents the SIGSEGV rationale. `numpy`/`NDArray`/`Mapping` are **runtime** imports, not `TYPE_CHECKING`-only, so `typing.get_type_hints()` resolves (this was a real Copilot finding, fixed in `f9d0696`).
- **`SearchMode.SINGLE_AGENT`** is mandatory on the adapter. Refinement is not adversarial; a zero-sum backup inverts the value sign at every other depth and silently produces a plausible, wrong policy — the F0 defect that forced the 2026-07-05 retraction.
- **The abstraction audit runs over this package in CI.** A Protocol member or `@abstractmethod` with no reader fails the build. A member staged ahead of its consumer goes in `scripts/audit_abstractions.py::_STAGED_FOR_UPCOMING_TASK` **with the task that retires it**, and that allowlist is self-expiring: a staged member that gains a reader fails.
- **No hardcoded numerics.** Thresholds and tolerances belong in `SubstrateConfig` (`src/research/substrates/config.py`) or as named module constants, and every field must be *read* by the substrate its `kind` selects — a knob nothing consumes is worse than a magic number, because it looks like it works. `marking_fraction` was added and deleted for exactly this reason.

## Data Flow: Substrate → Game → MCTS

```
RefinementSubstrate.initial_mesh()
        │
        ▼
   .solve(mesh) ──► SubstrateSolveResult(values, indicators, l2_error, n_dof, n_dof_free)
        │
        ▼
   .mark(indicators, theta) ──► bool mask   (src.research.marking.dorfler_mark)
        │
        ▼
   .refine(mesh, marked) ──► new mesh       (input mesh left byte-identical, AC3)
        │
        ▼
RefinementGame.apply_action ──► RefinementState
        │
        ▼
RefinementGameAdapter ──► MCTS (SINGLE_AGENT)
```
