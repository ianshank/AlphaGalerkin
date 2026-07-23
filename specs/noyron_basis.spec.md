# Spec: noyron_basis — MCTS basis selection on the Leap 71 helical operator (v2.2)

> **Status:** Draft
> **Owner:** PDE Solver
> **Primary module(s):** `src/poc/scenarios/noyron_basis.py`, `src/poc/scenarios/noyron_basis_config.py`
> **Config class:** `src.poc.scenarios.noyron_basis_config.NoyronBasisConfig`
> **Tracking:** CLAUDE.md Next-Steps row **v2.2** ("Plug `BasisSelectionGame` into `HelicalHeatOperator`")

## Context

The Leap 71 / Noyron integration already ships the plumbing for MCTS-guided Galerkin basis
selection on SDF-defined helical geometries: `HelicalBasisSelectionInterface` (game key
`pde_basis_helical`, `src/pde/register_games.py:156`) wraps a `BasisSelectionGame` around the
`helical_heat` / `helical_stokes` / `helical_magnetostatics` operators. What is missing is the
**first documented MCTS-on-Noyron result** — a PoC scenario that drives basis selection on the
helical operator, aggregates per-seed error reduction, and gates on physics-meaningful thresholds.
This is the v2.2 roadmap item; it produces the headline datapoint that basis selection reduces the
Galerkin error on a real Leap 71 geometry.

## User Story

**As a** researcher validating AlphaGalerkin on Leap 71 additive-manufacturing geometries,
**I want** an MCTS basis-selection PoC scenario that runs on the helical operator,
**so that** I have a reproducible, threshold-gated measurement of error reduction on a Noyron part.

## Data Contract

Configured by `NoyronBasisConfig(BaseScenarioConfig)`. Every tunable is a typed Pydantic field
with bounds and a docstring — no hardcoded values. Numerical-stability literals are named module
constants (`_SEED_PRIME_STRIDE`, `_REDUCTION_FLOOR`).

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `name` | `str` | `"noyron_basis"` | locked | Scenario dispatch key |
| `operator_name` | `Literal["helical_heat","helical_stokes","helical_magnetostatics"]` | `"helical_heat"` | — | Which helical operator to select bases for |
| `arms` | `list[str]` | `["random"]` | non-empty, unique, ⊆ {random,trained,llm} | MCTS evaluator arms to compare |
| `n_seeds` | `int` | `3` | `1..64` | Independent per-seed repeats |
| `n_simulations` | `int` | `16` | `1..4096` | MCTS simulations per macro-step |
| `max_basis_functions` | `int` | `12` | `1..256` | Bases the game may add before terminating |
| `n_candidate_bases` | `int` | `24` | `1..1024` | Candidate library size (== action space) |
| `target_residual` | `float` | `1e-6` | `0<x<1` | Inner-loop stop tolerance |
| `rollout_headroom` | `int` | `2` | `1..16` | Multiplier for the per-cell rollout cap |
| `manufactured` | `bool` | `True` | — | Overlay a manufactured ∏sin target (raw helical ops are homogeneous) |
| `manufactured_wavenumber` | `int` | `1` | `1..8` | Wavenumber k of the manufactured target |
| `min_error_reduction_pct` | `float` | `0.0` | `0..100` | Threshold: monotone reduction (≥0); raise for a tuned run |
| `max_final_residual` | `float` | `1.0` | `0<x≤10` | Threshold: primary-arm median final residual stays bounded |
| `trained_checkpoint_path` | `str \| None` | `None` | — | Required only if `"trained"` in arms |
| `lm_studio` | `LMStudioConfig \| None` | `None` | — | Required only if `"llm"` in arms |

`primary_arm` = `arms[0]` (drives the headline thresholds). `max_rollouts_for_cell()` =
`n_simulations * max_basis_functions * rollout_headroom`.

## Acceptance Criteria

### AC1: Scenario builds a helical basis-selection game via the registered interface
- **Given** a `NoyronBasisConfig` with `operator_name="helical_heat"`
- **When** the scenario constructs its game
- **Then** it reuses `_create_helical_pde_config` + `BasisSelectionGame` (the same path as
  `pde_basis_helical`) so the SDF geometry is carried on `PDEConfig.geometry`, and the game's
  `action_space_size == n_candidate_bases`.

### AC2: MCTS basis selection reduces the Galerkin error (monotone)
- **Given** the `random` arm and a **manufactured** non-zero target (`manufactured=True`, the
  default — the raw helical operators are homogeneous so the un-augmented game is degenerate)
- **When** the scenario runs `run_basis_selection_cell` for each seed
- **Then** the median final residual is ≤ the initial error (least-squares basis addition is
  monotone non-increasing), so `error_reduction_pct = 100*(init-final)/init` is ≥ 0 and recorded
  per arm.

> **Known limitation (research follow-up).** The *magnitude* of the reduction on 3D SDF helix
> geometry is limited by the current candidate Galerkin basis library — empirically only ~2–4 %
> even when every candidate basis is fitted. The headline thresholds therefore assert the provable
> correctness property (`reduction ≥ 0`, bounded residual), and the achieved reduction is recorded
> as a metric. A large headline target (e.g. 20 %) requires a geometry-aware basis library and is
> tracked as a v2.x item; raise `min_error_reduction_pct` on such a run.

### AC3: Arm gating is graceful
- **Given** `arms=["random","trained","llm"]` but no checkpoint and a failing LLM preflight
- **When** the scenario sets up
- **Then** the `trained` and `llm` arms are disabled **and** their thresholds are removed from
  `self.config.thresholds`, so absent metrics do not auto-FAIL (mirrors `llm_prior_ablation`).

### AC4: Reuse boundary is correct
- **Given** the helical operators are **not** in `PDE_TYPE_MAP`
- **When** the scenario builds its operator/game
- **Then** it does **not** call `_centaur_common.build_pde_operator` (which is `PDE_TYPE_MAP`-keyed);
  it reuses only `build_arm_evaluator` and `run_basis_selection_cell` (both geometry-agnostic).

## Thresholds

Emitted by `NoyronBasisConfig.get_default_thresholds()` as `src.poc.config.MetricThreshold`s; the
AQA test asserts this method returns exactly these metrics/operators.

| Metric | Operator | Value | Meaning |
|---|---|---|---|
| `error_reduction_pct` | `>=` | `min_error_reduction_pct` (0.0) | Basis addition is monotone (does not increase error) |
| `final_residual` | `<=` | `max_final_residual` (1.0) | Primary-arm median final residual stays bounded (no divergence) |

The default values assert **provable correctness**, not an aspirational magnitude (see the AC2
known-limitation note). Both fields are configurable so a GPU / improved-basis-library run can
tighten them.

## Regression Surface

```bash
pytest tests/poc/test_noyron_basis_config.py tests/poc/test_noyron_basis_scenario.py -v -m "not gpu_required"
pytest tests/poc/test_noyron_basis_config.py tests/poc/test_noyron_basis_scenario.py \
  --cov=src/poc/scenarios/noyron_basis.py --cov=src/poc/scenarios/noyron_basis_config.py \
  --cov-branch --cov-fail-under=85
python -m src.poc.cli run --config config/scenarios/noyron_basis_cpu.yaml
```

## Out of Scope

- GPU/LLM headline run (captured in `headline_runs.spec.md`).
- Octree-on-SDF AMR (v2.1), Noyron RP/EA operators (v2.3/v3.1) — separate roadmap rows.
- Real PicoGK STL ingestion — the scenario uses the analytical helix surrogate.
