# Spec: Element-local refinement substrate

> **Status:** Draft
> **Owner:** pde-solver
> **Primary module(s):** `src/refinement/substrate.py`, `src/research/marking.py`
> **[CORRECTED during implementation — not `src/refinement/marking.py` as originally planned:
> `tests/regression/test_import_contracts.py`'s `reference-baselines-do-not-import-the-candidate`
> contract forbids `src/research/baselines.py`/`fem_baseline.py` from importing anything under
> `src.refinement`, and `dorfler_mark` is active marking behaviour those two files delegate to —
> not the kind of inert protocol/type import the contract's one exemption (`src/mcts/gumbel.py`)
> tolerates. Moved to `src.research`, alongside its only callers.]**,
> `src/research/substrates/{config,tensor_grid,skfem_tri}.py`
> **Config class:** `src.refinement.config.RefinementGameConfig` (engine knobs) +
> `src.research.substrates.SubstrateConfig` (this spec's data contract)
> **Tracking:** `openspec/changes/element-local-substrate/`, PR #134 (spike evidence)

## Context

AlphaGalerkin's central claim is that MCTS multi-step look-ahead beats classical adaptive
refinement. **That claim cannot currently be measured**, and the project charter says so:
`DorflerAMRSolver._dorfler_mark_2d` (`src/research/baselines.py:969`) projects element-wise
marks onto the x and y axes, and `_refine_grid` runs per axis — so marking one element
inserts full grid *lines* across the whole domain. The refinement budget is spent away from
the singularity.

The measured consequence, now a committed artifact
(`results/lshape_adaptive_vs_uniform.{csv,run.json}`): adaptive Dörfler marking is **worse
than plain uniform refinement** at matched DOF, from 1.5× at 56 DOF to 10.5× at 2847, with
convergence `N^-0.14` against uniform's `N^-0.63`. Comparing two *marking policies* on that
substrate measures the substrate, not the policies.

`specs/lshape_amr_compare.spec.md:150` already names the fix — "the blocking prerequisite for
the v2.1 element-local (skfem / quadtree) work, not an optional upgrade" — and
`src/research/fem_baseline.py` already contains a genuine element-local FEM solver that
nothing in the repository consumes. A task-zero spike
(`evidence/spikes/2026-08-23-skfem-substrate.md`) established that wiring it in **inverts the
result**: on an element-local substrate, adaptive beats uniform by 4–10× at matched DOF,
recovering the optimal P1 rate (`N^-1.26`) against uniform's singularity-limited `N^-0.71`.

This spec defines the substrate abstraction that lets both arms of any refinement comparison
provably share one discretisation.

## User Story

**As a** reviewer assessing whether MCTS look-ahead beats classical marking,
**I want** both arms driven through one stepwise substrate interface, with the *only*
difference being how they choose what to refine,
**so that** a difference in the result is attributable to the policy rather than to the
plumbing — and so that the substrate's own adequacy is a gate I can see, not an assumption.

## Data Contract

Every tunable is a typed Pydantic `Field`. No hardcoded values.

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `kind` | `Literal["tensor_grid","skfem_tri"]` | `"skfem_tri"` | — | Which substrate to build. `tensor_grid` reproduces today's behaviour and is the control. |
| `element_type` | `Literal["P1","P2","P3"]` | `"P1"` | — | Lagrange order (`skfem_tri` only). |
| `initial_refinements` | `int` | `2` | `ge=0, le=8` | Uniform refinements applied to the coarse L-shape before the sweep. |
| `initial_side` | `int` | `4` | `ge=2, le=64`, even | Elements per axis (`tensor_grid` only); even so the reentrant corner is a node. |
| `marking_variant` | `Literal["squared","linear"]` | `"squared"` | — | Dörfler bulk quantity. `squared` is the textbook form; `linear` reproduces `fem_baseline`'s existing behaviour. |
| `error_metric` | `Literal["quadrature","nodal_rms"]` | `"quadrature"` | — | Which L2 the substrate reports. See AC6 — `nodal_rms` exists only to reproduce legacy numbers. |
| `enforce_immutable_meshes` | `bool` | `True` | — | Clear numpy write flags on mesh arrays. See AC3. |
| `solve_cache_max_entries` | `int` | `4096` | `ge=1, le=1e6` | Fingerprint-keyed solve cache bound. |

Named module-level constants for numerical-stability literals (mirroring
`DEFAULT_TRANSFER_RATIO_FLOOR`, `EVAL_SEED_STRIDE`):
`RATIO_FLOOR = 1e-15`, `AREA_FLOOR = 1e-30`, `RATE_FIT_MIN_POINTS = 3`.

## Acceptance Criteria

### AC1: `TensorGridSubstrate` is a byte-for-byte no-op
- **Given** the committed demo configuration
- **When** the Dörfler arm is driven through `TensorGridSubstrate` instead of the legacy
  `run_dorfler_arm`
- **Then** the trajectory matches the legacy arm **bitwise** (same functions, same order of
  operations), and matches the `dorfler` rows of `results/lshape_mcts_vs_dorfler.csv` to float
  tolerance — the CSV is 9-significant-figure text, so asserting text-exactness against a
  serialised artifact would be brittle
- **And** if bitwise equality against the live legacy arm cannot hold, the deviation is
  **recorded explicitly** rather than the assertion being quietly loosened

### AC2: `SkfemTriSubstrate` refinement is element-local and conforming
- **Given** a mesh with N elements
- **When** a strict subset M ⊂ N is marked and refined
- **Then** element count grows by O(|M|), not O(N) — verified against a full uniform refine
  on the same mesh (spike: 32 → 49 local, versus 32 → 128 uniform)
- **And** the result is conforming: **zero** edges shared by more than two elements, after one
  local refinement and after four successive ones

### AC3: Meshes are immutable in practice, not merely by convention
- **Given** any substrate mesh
- **When** `refine()` is called
- **Then** the input mesh's coordinate and connectivity bytes are unchanged, and a new object
  is returned
- **And** with `enforce_immutable_meshes=True` the arrays are non-writeable, because the spike
  found `skfem` leaves `mesh.p.flags.writeable` **True** — immutability is a property of the
  refinement *API*, not of the array, and the cheap-`clone()` design depends on it

### AC4: One marking function, two frozen variants, byte-identical to both call sites
- **Given** the existing `DorflerAMRSolver._dorfler_mark`/`_dorfler_mark_2d` (squared bulk;
  marks one element when the total is zero) and `ScikitFEMPoissonSolver._dorfler_mark` (linear
  bulk; returns all-False when the total is zero)
- **When** both delegate to `refinement.marking.dorfler_mark`
- **Then** each produces **byte-identical** output to its previous implementation, over a
  Hypothesis sweep of indicator arrays including the all-zeros case

### AC5: The DOF convention is declared, never assumed
- **Given** two substrates counting different things (in-domain FDM nodes versus FEM basis DOFs)
- **When** a comparison is run
- **Then** `describe()["dof_convention"]` is recorded in the run manifest, all arms are asserted
  to share **one substrate instance**, and `n_dof_free <= n_dof` holds

### AC6: The reported L2 is mesh-independent, and differs from the nodal RMS on a graded mesh
- **Given** an adaptively refined mesh
- **When** both the quadrature L2 and the legacy nodal RMS are computed
- **Then** they **differ**, and the ratio is not constant across refinement
- **Rationale, measured rather than argued** (spike): `nodalRMS/quadratureL2` drifts 0.34→0.53
  for the uniform arm but 0.34→0.76 for the adaptive arm, because adaptive clusters nodes at
  the singularity. Reporting the nodal RMS would flatter whichever arm refines hardest — a
  *fabricated* result, not a measurement error. `ScikitFEMPoissonSolver.solve()`'s existing
  asserted outputs are unchanged; the quadrature L2 is additive.

### AC7: The substrate is adequate — the gate that makes any comparison meaningful
- **Given** the L-shaped Poisson benchmark and a fixed θ
- **When** adaptive Dörfler marking and uniform refinement are both run
- **Then** adaptive beats uniform at matched DOF, asserted as a **log-log rate separation over
  a specified DOF range** (not a single favourable point, since adaptive routinely loses at
  coarse DOF), with the range, θ, margin and tolerance all pinned in config
- **And** the same assertion **fails** on `TensorGridSubstrate` — a gate that passes on both
  substrates is not a gate
- **Scope note:** this reproduces a textbook AFEM result. It is an *implementation-correctness*
  gate, not a research finding, and is labelled as such.

### AC8: The reentrant corner is where the benchmark says it is
- **Given** the initial mesh
- **When** it is constructed
- **Then** the reentrant corner is asserted to be a mesh node at the origin
- **Rationale:** `skfem.MeshTri.init_lshaped()` already places it there, and translating or
  scaling the mesh moves the singularity outside the domain. The spike did exactly that and
  produced a confident, entirely wrong *"adaptive loses on skfem too"* result (both arms at
  `N^-1.05`), caught **only** because a uniform rate that good is impossible for a singular
  problem. The uniform-arm rate band is therefore also a tripwire: a substrate that converges
  too *well* is as diagnostic as one that diverges.

## Thresholds

This spec defines a substrate, not a PoC scenario, so it contributes **no**
`MetricThreshold` entries: there is no `get_default_thresholds()` to agree with, and inventing
a scenario wrapper purely to satisfy the template would add a registry entry with no consumer.
The gated experiment metrics live in the arena spec that consumes this substrate.

The substrate's own gates are test constants, surfaced as named module-level values so they are
tunable without editing assertions:

| Constant | Value | Meaning |
|---|---|---|
| `UNIFORM_RATE_BAND` | `(-0.85, -0.55)` | Uniform P1 L2 rate on the L-shape. Spike measured `-0.710`; theory gives `-2/3`. A rate *outside* this band on either side is a defect — too good means the singularity is not in the domain (AC8). |
| `ADAPTIVE_RATE_MIN` | `-1.10` | Adaptive must reach at least this. Spike measured `-1.256`. |
| `ADAPTIVE_VS_UNIFORM_MAX_RATIO` | `1.0` | Adaptive must beat uniform at matched DOF over `RATE_FIT_DOF_RANGE`. |
| `RATE_FIT_DOF_RANGE` | ~~`(200, 2600)`~~ → `(200, 4000)` | The asymptotic window the rate is fitted over. Below it, neither arm has separated. **CORRECTED during Slice D implementation** — see below. |

**Correction: `RATE_FIT_DOF_RANGE` `(200, 2600)` → `(200, 4000)`.** The originally pinned
window is *physically incapable* of holding `RATE_FIT_MIN_POINTS` (3) **uniform** points: a 2D
uniform arm quadruples DOF per level, so a 13× window spans at most two of them. This was not a
judgement call — `fit_log_log_rate` raised `InsufficientSweepPointsError` rather than fitting a
two-point slope, which is how the mismatch surfaced at all. Measured on **both** substrates:
`skfem_tri` uniform lands on `[225, 833, 3201]` and `tensor_grid` uniform on `[208, 800, 3136]`,
each giving n=2 inside `(200, 2600)` and n=3 inside `(200, 4000)`. Widened to the cheapest
window that admits three uniform points on both substrates (measured cost: +0.3 s adaptive,
+1.1 s uniform). The other three constants were **not** touched — all three hold at their
originally pinned values against the production primitives (measured `skfem_tri`: adaptive
`-1.2515`, uniform `-0.6710`; `tensor_grid`: adaptive `-0.2325`, uniform `-0.6489`), so this is a
window-width correction, not a threshold loosening.

## Regression Surface

```bash
# Substrate contract + marking parity + the adequacy gate (CPU)
COVERAGE_CORE=pytrace pytest tests/refinement/ tests/research/test_tensor_grid_substrate.py \
  tests/research/test_skfem_substrate.py tests/research/test_amr_arena_interpretability.py -v

# Back-compat: the legacy harness and the FEM solver must be untouched in behaviour
COVERAGE_CORE=pytrace pytest tests/research/test_lshape_amr_compare.py \
  tests/research/test_fem_baseline.py tests/research/test_baselines.py \
  tests/pde/test_lshape_amr_game.py -v

# Per-module coverage gates (branch) — the new code rides the existing gates
COVERAGE_CORE=pytrace pytest tests/refinement/ --cov=src/refinement --cov-branch --cov-fail-under=85
COVERAGE_CORE=pytrace pytest tests/research/ --cov=src/research --cov-fail-under=85

# skfem-dependent tests must be VISIBLY skipped on CPU CI, and hard-fail in test-extras
pytest tests/research/test_skfem_substrate.py -m fem_required -v
ALPHAGALERKIN_REQUIRE_EXTRAS=1 pytest tests/research/test_skfem_substrate.py -v  # errors if skfem absent
```

## Out of Scope

- **The three/four-arm experiment itself.** This spec delivers the substrate and its adequacy
  gate. Arms, budget matching, statistics and the pre-registration land in the arena spec.
- **A trained evaluator.** Deliberately excluded so the eventual comparison isolates planning
  depth; training a prior is its own workstream.
- **Replacing `LShapeAMRGame` / `lshape_amr_compare.py`.** They are **frozen as the
  back-compat golden reference** — not extended, not deleted — and marked superseded in
  `specs/lshape_amr_compare.spec.md`. The two-path period is recorded as a time-boxed charter
  deviation with its retirement condition stated (the golden test is the only remaining
  consumer), so it is disclosed rather than accidental.
- **Estimator vectorisation.** `_compute_zz_indicator` is a Python loop over elements with an
  inner loop over vertices, and the spike measured the estimator at ~2.5× the solve — it is the
  dominant cost inside MCTS, not mesh cloning. Vectorising perturbs floating-point results, so
  it lands **after** the goldens are captured, as a separate explicitly-tolerance'd change.
- **Quadtree / hanging-node AMR.** Conforming RGB refinement is what `skfem` provides and what
  this spec commits to. A non-conforming backend is a later decision.
