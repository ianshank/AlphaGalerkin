# Spec: Stochastic Galerkin Operator-Splitting Layer (NKE)

> **Status:** Implemented
> **Owner:** AlphaGalerkin core
> **Primary module(s):** `src/pde/stochastic/` (subpackage), `src/research/stochastic_galerkin_compare.py`, `src/poc/scenarios/stochastic_galerkin_compare.py`
> **Config class:** `src.poc.scenarios.stochastic_galerkin_compare_config.StochasticGalerkinCompareConfig` (scenario contract); `src.pde.stochastic.config.*` (layer contracts)
> **Tracking:** OpenSpec change `alphagalerkin-nke-integration` (external draft, 2026-07-22); branch `claude/nke-stochastic-galerkin-plan-s4odjz`

## Context

AlphaGalerkin's operator-learning stack is deterministic (Galerkin attention, FNet, PDE operators
with manufactured solutions). The external OpenSpec change `alphagalerkin-nke-integration` asks for
an **additive stochastic Galerkin layer** based on the NKE paper (arXiv:2607.19173, "Neural
Kolmogorov Equations: Parallelizable Learning of Stochastic Dynamics under General Noise"): a
Lagrangian Galerkin projection of a Kolmogorov-forward-equation generator `L = A + D + J`
(advection + diffusion + jump) onto a Gaussian-mixture basis, trained via Strang operator splitting
with a parallel-in-time (non-autoregressive) trapezoidal loss. NKE does *not* do MCTS or adaptive
basis/mesh selection, so it does not compete with AlphaGalerkin's novelty claim — it supplies a
citable stochastic-operator layer and an occasion to tighten the repo's novelty-boundary docs.

**Provenance caveat (binding):** the source paper was unreachable from the implementation
environment (arXiv rejects the datacenter IP; the paper postdates the model's training data). The
layer is implemented from the **standard, independently derivable formulation** (moment-matching
Galerkin projection of the Kolmogorov forward equation; symmetric Strang composition). The change
doc's own validation criteria (OU closed form < 1e-3; O(dt²) splitting; monotone loss ≤ 500 steps)
pin down correctness. A paper-exact cross-check of the projection ansatz (especially
mixture-weight dynamics and loss weighting) is an **open reviewer follow-up** — see Out of Scope.

### Declared deviations from the change doc

1. **Placement** (user-approved): `src/pde/stochastic/` + `tests/pde/stochastic/`, not the change
   doc's literal `src/operators/stochastic/` — no `src/operators/` package exists; repo conventions
   and CI gates put operator math in `src/pde/`, harnesses in `src/research/`, scenarios in
   `src/poc/scenarios/`.
2. **No `PDEType`/`PDE_TYPE_MAP` registration**: the generator acts on densities/moments, not
   pointwise fields — forcing the `PDEOperator` interface would create dead-abstraction stubs, and
   a `PDEType` entry would leak the operator into the MCTS basis-selection Literal enums,
   contradicting the change doc's "NO changes to the MCTS self-play components". A thin
   `FokkerPlanckOperator(PDEOperator)` adapter is deferred (Out of Scope).
3. **Monotone-loss reinterpretation** (spec change, not implementation detail): the change doc's
   "Strang-split trainer loss decreases monotonically on toy benchmark within 500 steps" is gated
   as (i) non-increasing `DEFAULT_MONOTONE_WINDOW`-step window means at a calibrated relative
   tolerance and (ii) `final_loss/initial_loss < DEFAULT_LOSS_RATIO_GATE`, in deterministic
   full-batch mode. Per-step monotonicity under stochastic optimization would be a dishonest gate.

### Change-doc transcription (source-of-truth for PR review)

The OpenSpec change doc is not otherwise committed to this repo. Its requirements and tasks,
verbatim:

**ADDED Requirement: Stochastic Galerkin Projection Layer** — "The system SHALL provide a
Lagrangian Galerkin projection of a linear operator `L` onto a finite Gaussian-mixture basis,
exposing projected mean/covariance dynamics for advection and diffusion terms and a jump semigroup
approximation." Scenarios: *(a)* "GIVEN a linear generator `L = A + D` with known drift `f` and
diffusion `g g^T` WHEN the system projects `L` onto a single Gaussian basis element THEN it SHALL
recover closed-form moment-matching ODEs for mean and covariance consistent with the reference
derivation in arXiv:2607.19173"; *(b)* "GIVEN a generator that includes jump terms without a
supplied MDN jump-semigroup model WHEN the system attempts projection THEN it SHALL raise a
configuration error rather than silently ignoring the jump component."

**ADDED Requirement: Operator-Splitting Trainer** — "The system SHALL provide a Strang-splitting
training procedure that computes parallel-in-time trapezoidal losses over precomputed particle
clusters, without requiring sequential SDE rollout." Scenario: "GIVEN a batch of particle
trajectories sampled at fixed time grid points WHEN the trainer computes the Strang-split loss
THEN all timestep losses SHALL be computable independently in a single forward pass (no
autoregressive dependency across timesteps)."

**ADDED Requirement: Novelty-Gap Documentation Guard** — "The project documentation SHALL
explicitly state the boundary between AlphaGalerkin's MCTS-guided basis/mesh selection claim and
any newly cited operator-learning paper's actual scope. … GIVEN a new paper is added to
`docs/related-work.md` WHEN the entry claims method overlap with AlphaGalerkin THEN the entry
SHALL include an explicit 'does NOT do' clause covering MCTS, adaptive mesh/basis selection, and
LBB stability unless the source paper demonstrably includes them."

**Tasks:** 1.1 read NKE method section and extract exact moment-projection formulas *(deviation:
paper unreachable — standard derivation used; extraction deferred to reviewer follow-up)*;
1.2 `GaussianMixtureBasis` with configurable `K`; 1.3 projected advection/diffusion ODE integrator
(moment matching); 1.4 MDN-based jump semigroup module (optional, gated behind config flag)
*(scope raised to full trained MDN in v1, user-approved)*; 1.5 Strang-splitting parallel-in-time
trainer with unit tests against a known analytic SDE (Ornstein-Uhlenbeck); 1.6 comparison harness:
deterministic Galerkin-attention path vs stochastic Galerkin-projection path on a shared toy
PDE/SDE benchmark; 1.7 update `docs/related-work.md` with NKE entry and explicit novelty-boundary
clause; 1.8 update SBIR narrative to reference the stochastic extension pathway as future work,
not current capability; 1.9 regression test asserting no MCTS/self-play code paths were modified
by this change *(interpreted as an AST import-isolation guard — see AC7)*.

## User Story

**As a** researcher extending AlphaGalerkin toward stochastic dynamics,
**I want** a Gaussian-mixture Galerkin projection of Kolmogorov-forward generators with a
parallel-in-time Strang-splitting trainer,
**so that** I can learn/propagate densities under diffusion and jump noise without touching the
deterministic MCTS/self-play core, and cite NKE with an honest novelty boundary.

## Data Contract

Layer configs (`src/pde/stochastic/config.py`, all `BaseModuleConfig` subclasses, `extra="forbid"`,
every tunable a typed bounded `Field`):

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `GaussianMixtureBasisConfig.dim` | `int` | — | `ge=1, le=8` | State dimension d |
| `GaussianMixtureBasisConfig.n_components` | `int` | `1` | `ge=1, le=32` | Mixture size K |
| `GaussianMixtureBasisConfig.dtype` | `Literal["float32","float64"]` | `"float64"` | — | Moment-math dtype |
| `GaussianMixtureBasisConfig.weight_dynamics` | `Literal["frozen"]` | `"frozen"` | — | v1 limitation surfaced as a forward-compatible knob |
| `JumpConfig.rate` | `float` | — | `ge=0` | Compound-Poisson intensity λ (0 ⇒ no jump term) |
| `JumpConfig.jump_mean` / `jump_cov` | `list` | — | validated shapes | ξ ~ N(μ_ξ, Σ_ξ) |
| `StochasticGeneratorConfig.drift_matrix` / `drift_bias` | `list \| None` | `None` | shape-validated | Linear drift f(x)=Ax+b (exact path) |
| `StochasticGeneratorConfig.diffusion` | `list[list[float]]` | — | shape-validated | g (d×m); Q = g gᵀ |
| `StochasticGeneratorConfig.jump` | `JumpConfig \| None` | `None` | — | Optional jump term |
| `MDNJumpConfig.n_components` | `int` | — | `== basis K` (validator) | MDN output mixture size |
| `MDNJumpConfig.hidden_dims` | `list[int]` | `[64, 64]` | `each ge=1` | MLP widths |
| `StrangSplittingConfig.dt` / `t_end` | `float` | — | `gt=0` | Coarse step / horizon |
| `StrangSplittingConfig.ad_integrator` | `Literal["exact_expm","rk4"]` | `"exact_expm"` | — | A+D flow method (rk4 reuses `src/pde/time_stepping.py`, fixed dt) |
| `StrangTrainerConfig.n_particles` | `int` | — | `ge=16, le=100000` | SDE particle count |
| `StrangTrainerConfig.n_time_slices` | `int` | — | `ge=3` | Coarse grid points M |
| `StrangTrainerConfig.max_steps` | `int` | `500` | `ge=1` | Optimizer step budget |
| `StrangTrainerConfig.full_batch` | `bool` | `True` | — | Deterministic AC mode (all slices every step) |

Named module-level constants (numerical-stability literals; no magic numbers):
`DEFAULT_COV_JITTER = 1e-8`, `DEFAULT_MDN_MIN_SCALE = 1e-4`, `DEFAULT_OU_MOMENT_TOL = 1e-3`,
`DEFAULT_STRANG_SLOPE_MIN = 1.7`, `DEFAULT_STRANG_SLOPE_MAX = 2.3`,
`DEFAULT_MONOTONE_WINDOW = 50`, `DEFAULT_MONOTONE_REL_TOL` (calibrated, see below),
`DEFAULT_LOSS_RATIO_GATE` (calibrated), `DEFAULT_TRAINED_MDN_MOMENT_TOL` (calibrated),
`DEFAULT_LOSS_GAP_CLOSURE` (calibrated), `DEFAULT_KMEANS_MAX_ITERS = 100`,
`DEFAULT_KMEANS_TOL = 1e-6`, `DEFAULT_CLUSTER_COV_FLOOR = 1e-6`,
`DEFAULT_STOCHASTIC_MSE_GATE` (calibrated, scenario gate).

### Pinned toy problems (gates are reproducible only against these)

- **OU (AC1)** — 1D: `A=[[-1.0]]`, `b=[0.0]`, `g=[[0.5]]`, `m0=[1.0]`, `P0=[[0.5]]`, `T=1.0`;
  2D: `A=[[-1.0,0.3],[0.0,-0.8]]`, `b=[0.1,-0.2]`, `g=diag(0.4,0.3)`, `m0=[1.0,-0.5]`,
  `P0=diag(0.3,0.2)`, `T=1.0`. RK4 path `dt=0.01`; Strang path `dt=0.05`.
- **Jump-diffusion OU (AC3/AC5)** — 1D: `A=[[-1.0]]`, `b=[0.0]`, `g=[[0.3]]`, `λ=2.0`,
  `μ_ξ=[0.5]`, `Σ_ξ=[[0.04]]`, `m0=[0.0]`, `P0=[[0.1]]`, `T=1.0`, coarse `dt=0.1` (11 slices),
  `sim_dt=0.005`, `n_particles=2000`, `seed=42`.
- **Strang order sweep (AC4)** — 1D OU: `A=[[-1.0]]`, `Q=[[0.5]]`, `P0=[[0.2]]`, `T=0.8`,
  `dt ∈ {0.2, 0.1, 0.05, 0.025}`; **covariance** error (the mean has zero splitting error for the
  no-jump linear case).

### Calibration procedure (repo precedent: `specs/transfer_baseline_compare.spec.md`)

`DEFAULT_TRAINED_MDN_MOMENT_TOL`, `DEFAULT_MONOTONE_REL_TOL`, `DEFAULT_LOSS_RATIO_GATE`, and
`DEFAULT_STOCHASTIC_MSE_GATE` ship as *placeholders* and are **calibrated from the first measured
run** on the pinned problems (commits 5, 6, 9): run the pinned configuration, record the observed
value, set the gate with headroom (~2× observed error / ~½ observed improvement), and record the
observed value in this section. Gates must never encode an unmeasured claim. Calibrated values:

| Constant | Placeholder | Observed (pinned run) | Shipped gate |
|---|---|---|---|
| `DEFAULT_TRAINED_MDN_MOMENT_TOL` | `5e-2` | direct jump training: mean err `6.7e-3`, cov err `2.7e-2` (300 steps, seeds 42/7, NLL 0.803→0.727); full Strang-trained trajectory at T=1: mean err `1.7e-2`, cov err `7.4e-3` | `5e-2` (≈1.8× headroom over the worst observed error) |
| `DEFAULT_MONOTONE_REL_TOL` | `1e-3` | max window-mean relative increase `2.8e-6` (500 steps, window 50, seeds 42) | `1e-3` (~350× headroom) |
| `DEFAULT_LOSS_RATIO_GATE` | `0.9` | `final/initial = 0.950` (initial 0.6916 → final 0.65714) — the 0.9 placeholder was **unreachable**: a dt-scaled residual MDN starts near identity, so the closable NLL gap is small | `0.98` (recalibrated; the gap-closure gate below is the sharper criterion) |
| `DEFAULT_LOSS_GAP_CLOSURE` *(added at calibration)* | — | oracle floor `0.65727`; trainer final `0.65714` → closure fraction `0.000` (floor reached) | `0.25` |
| `DEFAULT_STOCHASTIC_MSE_GATE` | `1e-5` | `2.3e-8` (grid 32, strang_dt 0.1, demo budget); CI micro-budget grid 16: `2.06e-8` | `1e-6` (~40× headroom; recorded in `config/baselines/stochastic_galerkin_ci.json`) |

## Acceptance Criteria

### AC1: OU moment recovery (change-doc scenario a)
- **Given** the pinned 1D/2D OU generators (`L = A + D`, linear drift, constant diffusion) and a
  K=1 Gaussian basis
- **When** moments are propagated to `T=1.0` via the unsplit RK4 path and the no-jump Strang path
- **Then** max abs error vs `ou_mean`/`ou_covariance` (van Loan closed forms) is `< 1e-3`
  (`DEFAULT_OU_MOMENT_TOL`), including under a Hypothesis sweep of stable `A` (bounded spectral
  radius so fixed-dt RK4 stays within tolerance).

### AC2: jump configured without a jump model → configuration error (change-doc scenario b)
- **Given** a `StochasticGeneratorConfig` with `jump.rate > 0` and no jump-semigroup model supplied
- **When** `KolmogorovGenerator` is constructed (and, defense-in-depth, when `StrangSplitStep` is
  composed)
- **Then** `JumpModelMissingError` is raised with a message naming the fix
  (`MDNJumpSemigroup` / `AnalyticCompoundPoissonMoments`); `rate == 0` or `jump=None` does not
  raise; no code path integrates A+D while silently dropping J.

### AC3: jump-diffusion OU (oracle exact; trained MDN calibrated)
- **Given** the pinned jump-OU problem
- **When** Strang-propagated with `AnalyticCompoundPoissonMoments` (exact first-two-moment oracle)
- **Then** mean/covariance error vs `jump_ou_mean`/`jump_ou_covariance` is `< 1e-3`; and
- **When** the MDN jump semigroup is trained on precomputed particle clusters (pinned seed)
- **Then** its NLL strictly decreases and the propagated trajectory error is
  `< DEFAULT_TRAINED_MDN_MOMENT_TOL` (calibrated); the float64↔float32 `advance()` boundary
  round-trips dtype and shape.

### AC4: Strang splitting is second-order
- **Given** the pinned AC4 sweep (oracle/no-jump substeps only — a trained MDN has an
  approximation floor that would falsify O(dt²))
- **When** covariance error at fixed `T` is measured across the dt-halving sweep
- **Then** the mean log2 error ratio (slope) lies in
  `[DEFAULT_STRANG_SLOPE_MIN, DEFAULT_STRANG_SLOPE_MAX] = [1.7, 2.3]`.

### AC5: trainer loss decreases within 500 steps (change-doc criterion, reinterpreted — deviation 3)
- **Given** the pinned jump-OU problem (the MDN is the only trainable component; a no-jump
  configuration raises `StochasticConfigurationError`), full-batch mode, fixed seed
- **When** `StrangParallelTrainer.train()` runs for ≤ 500 steps
- **Then** `DEFAULT_MONOTONE_WINDOW`-step window means are non-increasing within
  `DEFAULT_MONOTONE_REL_TOL` (calibrated), `final_loss/initial_loss < DEFAULT_LOSS_RATIO_GATE`
  (calibrated), and the fraction of the (initial − oracle-achievable) loss gap left open is
  `< DEFAULT_LOSS_GAP_CLOSURE` — the trainer computes the same batched loss with the exact
  compound-Poisson oracle substituted for the MDN, which is the honest achievable floor.

### AC6: parallel-in-time independence (change-doc trainer scenario)
- **Given** precomputed particle clusters on the pinned coarse grid
- **When** `compute_slice_losses()` evaluates all M−1 interval losses in one batched forward pass
- **Then** the result matches a per-slice loop within rtol (float32 GEMM batching is not bitwise),
  each `ℓ_i` is invariant under slice-order permutation, and the trapezoid-style weight vector is
  `dt·[½,1,…,1,½]` (an implementation choice, not asserted paper fidelity).

### AC7: MCTS/self-play untouched (change-doc task 1.9, interpreted)
- **Given** the new modules (`src/pde/stochastic/*`, harness, scenario, CLI)
- **When** their import statements are statically analyzed (AST; module-boundary prefix matching)
- **Then** none import `src.mcts`, `src.games`, `src.refinement`, `src.training.self_play`,
  `src.training.trainer`, `src.pde.mcts_adapter`, `src.pde.game`, `src.pde.games`,
  `src.pde.game_interface`, `src.pde.register_games`, or `src.pde.trainer`. *Scope: this proves
  static import direction only — `src/pde/__init__.py` side effects still execute at runtime.*
  Companion (run, not asserted here): the MCTS and F0/F1 regression surfaces stay green.

### AC8: comparison harness (change-doc task 1.6)
- **Given** the shared 2D Fokker-Planck/OU benchmark (analytic Gaussian ground truth)
- **When** the deterministic Galerkin-attention arm (trained micro-operator) and the stochastic
  moment-projection arm are evaluated on the identical held-out grid/targets
- **Then** both arms' density MSEs, their ratio, wall-clocks, and parameter counts are recorded;
  only `stochastic_density_mse < DEFAULT_STOCHASTIC_MSE_GATE` is gated (see Thresholds);
  `load_config_from_dict` returns `StochasticGalerkinCompareConfig`; CSV/PNG artifacts are
  emitted; the AQA test asserts this spec's Thresholds table equals `get_default_thresholds()`.

### AC9: novelty-gap documentation guard (change-doc requirement 3)
- **Given** `docs/related-work.md` with a delimited entries region
- **When** `tests/regression/test_related_work_guard.py` runs
- **Then** every entry in the region contains a case-insensitive "does not do" clause; the NKE
  entry's clause names MCTS, adaptive basis/mesh selection, and LBB; and `README.md` no longer
  contains the retracted blanket claim "no published papers combine MCTS with Galerkin".

## Thresholds

Scenario gate — must equal
`StochasticGalerkinCompareConfig.get_default_thresholds()` (AQA-asserted):

| Metric | Operator | Value | Meaning |
|---|---|---|---|
| `stochastic_density_mse` | `<` | `DEFAULT_STOCHASTIC_MSE_GATE = 1e-6` (calibrated; observed `2.3e-8`) | The stochastic Galerkin arm reproduces the analytic Fokker-Planck density on the shared grid; the floor is set by the Strang splitting error in the rendered density (rendering itself is exact evaluation) |

Recorded, **ungated** (honesty rule — "novelty ≠ superiority", `docs/proposals/PRIOR_ART_REVIEW.md`):
`deterministic_density_mse`, `stochastic_vs_deterministic_mse_ratio`, per-arm wall-clock and
parameter counts. On this benchmark the stochastic path is near-exact by construction; gating a
ratio would be a self-serving benchmark.

## Regression Surface

```bash
# Layer + harness + scenario + guards (CPU; coverage exactly as CI runs it — native runner form)
COVERAGE_CORE=pytrace python -m coverage run --branch \
  --include="*/src/pde/stochastic/*,*/src/research/stochastic_galerkin_compare.py,*/src/poc/scenarios/stochastic_galerkin_compare*.py" \
  -m pytest tests/pde/stochastic tests/research/test_stochastic_galerkin_compare.py \
    tests/poc/test_stochastic_galerkin_compare_config.py tests/poc/test_stochastic_galerkin_compare_scenario.py \
    tests/scripts/test_run_stochastic_galerkin_compare.py tests/regression/test_related_work_guard.py -q
python -m coverage report --fail-under=85

# Untouched-path safety (AC7 companion)
pytest tests/mcts tests/pde/test_reward_reachability.py tests/pde/test_clone_isolation.py tests/pde/test_mcts_adapter.py -q
```

## Out of Scope

- **Mixture-weight dynamics** (v1: `weight_dynamics="frozen"`; exact for linear drift, an
  approximation otherwise) — primary paper-fidelity follow-up.
- **Learnable drift/diffusion networks** (v1: drift/diffusion are known config inputs; only the
  MDN jump semigroup is trained) — the largest gap vs the paper's "learning of stochastic
  dynamics" framing.
- **EM-quality mixture fitting** (v1: seeded torch Lloyd's k-means + per-cluster empirical
  moments — deliberately crude).
- **Nonlinear-drift exactness** (sigma-point expectations are documented approximate).
- **`FokkerPlanckOperator(PDEOperator)` adapter** and any `PDEType`/`PDE_TYPE_MAP` registration
  (deviation 2).
- **Paper-exact formula cross-check** against arXiv:2607.19173 (unreachable at implementation
  time) — reviewer follow-up.
- **GPU paths** — the layer is CPU-first; no `gpu_required` surface is added.
