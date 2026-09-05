# Design: `refinement-game-registrant`

## Technical Approach

Ship the smallest pure `RefinementGame` that (a) sits on `RefinementSubstrate`, (b) registers
safely, (c) creates the first production registry lookup, and (d) gives `fingerprint` a real
reader via a solve cache. Do not invent arena statistics here.

## Architecture Decisions

### AD1 — Game lives outside `src/refinement/`

**Decision.** Place the concrete game under `src/pde/games/` (e.g. `substrate_refinement.py`) or
an equivalent domain package — **not** under `src/refinement/`.

**Rationale.** `src/refinement/` is domain-free (numpy Protocol + ABC + adapter). Import-contract
tests forbid research/PDE imports into that package. The substrate concretes already live under
`src/research/substrates/`.

**Consequence.** `src/refinement/` gains a *consumer*, not a new domain dependency.

### AD2 — Registration only via explicit module

**Decision.** Add e.g. `src/pde/register_refinement_games.py` (sibling of `register_games.py`).
Call sites import that module for side-effect registration (`# noqa: F401`). Never register from
`__init__.py`.

**Rationale.** `src/pde/games/__init__.py` documents the SIGSEGV / C-coverage-tracer failure mode
from eager game+torch/MCTS imports. Task 7.2/7.3 are load-bearing.

**Consequence.** Import-graph test in the same PR; subprocess registry clear in tests.

### AD3 — Purity is a contract, not a style preference

**Decision.** `apply_action(state, action)` must not mutate the game instance. Mesh identity is
carried by substrate immutability + `fingerprint`. Default `clone()→self` on the ABC is only safe
if state is immutable / copy-on-write.

**Rationale.** `LShapeAMRGame.apply_action` mutates `self._xs/_ys` and breaks MCTS
node-by-action-sequence identity. Prefix-keyed mesh cache depends on purity.

**Consequence.** Unit tests assert instance fields unchanged across `apply_action`; fingerprint
stable for identical meshes.

### AD4 — Action model is per-unit; classical bulk mark stays in the arena

**Decision.** Slice E actions select a single refinement unit (or a frozen top-1 policy for the
smoke path). Dörfler θ-bulk marking remains a *classical arena arm*, not this game's default
policy.

**Rationale.** Bundling budget-matched classical vs MCTS comparison into Slice E recreates the
"no experiment in this change" violation.

### AD5 — Fingerprint cache retires the staged audit exemption

**Decision.** Implement an LRU/dict keyed by `fingerprint`, bounded by
`SubstrateConfig.solve_cache_max_entries`. Remove `("RefinementSubstrate", "fingerprint")` from
`scripts/audit_abstractions.py::_STAGED_FOR_UPCOMING_TASK` in the same PR.

**Rationale.** Estimator dominates solve (~2.5×); MCTS without cache is infeasible. Staged
exemptions that gain a reader without allowlist removal fail
`test_every_staged_exemption_is_still_forward`.

### AD6 — First registry lookup is mandatory

**Decision.** Config-driven construction must call `RefinementSubstrateRegistry` (or equivalent
get-by-kind). Direct `SkfemTriSubstrate(...)` in production paths is not enough to retire the
"zero lookups" deviation.

## Interfaces to satisfy

1. `RefinementGame` ABC (`action_space`, `state`, `valid`/`apply`, terminal, reward, `to_tensor`).
2. `RefinementGameAdapter` → MCTS (`SearchMode.SINGLE_AGENT`) — smoke wiring only; no multi-seed
   campaign.
3. `RefinementSubstrate` + `SubstrateSolveResult` mapping into `RefinementState`.
4. Optional CPU path: `tensor_grid` for smoke without `[fem]`; skfem path `fem_required`.

## Test plan (Slice E)

- Unit: purity, fingerprint stability, cache hit/miss, registry get-by-kind.
- Import-graph: no registration from `__init__.py`.
- Guard: production registrant count ≥ 1 (or charter deviation row removed).
- Adequacy gate remains green; do not weaken thresholds.
- Focus gate: no substantive frozen-track churn.
