# Spec: L-shaped Poisson AMR — MCTS refinement vs Dörfler marking

> **Status:** Implemented
> **Owner:** pde-solver
> **Primary module(s):** `src/research/lshape_amr_compare.py`, `src/pde/games/lshape_amr.py`, `src/poc/scenarios/lshape_amr_compare.py`
> **Config class:** `src.poc.scenarios.lshape_amr_compare_config.LShapeAMRCompareConfig`
> **Tracking:** branch `claude/pde-solver-mcts-baseline-oymltc`

## Context

AlphaGalerkin's thesis is that an MCTS refinement policy beats classical adaptive methods. The
only prior MCTS-on-a-real-domain result (`noyron_basis`) runs on a proprietary-flavoured Leap 71
helical SDF with a 2–4% ceiling and gates on `error_reduction_pct >= 0` — a **non-test**, since
least-squares basis addition is monotone non-increasing, so that bound can never fail.

This feature builds the missing *competitive* baseline on the **standard L-shaped Poisson AMR
benchmark** (reentrant-corner singularity `u = r^(2/3) sin(2θ/3)`): MCTS refinement vs Dörfler
bulk marking on an identical masked solver, residual estimator, geometry mask and active-DOF
accounting. It replaces the monotone non-criterion with a **comparative, falsifiable** gate.

## User Story

**As a** reviewer assessing the AlphaGalerkin thesis,
**I want** an honest, reproducible head-to-head of the MCTS refinement policy against Dörfler
marking on a benchmark I recognise,
**so that** I can judge whether MCTS lookahead actually beats greedy marking — win or lose.

## Data Contract

Configured by `LShapeAMRCompareConfig` (`BaseScenarioConfig` subclass). Every tunable is a typed
Pydantic `Field` — no hardcoded values. Key fields:

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `scale` | `float` | `1.0` | `gt=0, le=100` | L-shape half-width `s` (domain `[-s,s]^2 \ (0,s]x[-s,0)`). |
| `initial_side` | `int` | `4` | `ge=2, le=64`, **even** | Elements per axis on the shared coarse grid (even so the reentrant corner at the origin is a grid node). |
| `n_seeds` | `int` | `5` | `ge=1, le=64` | Seeds swept; the gated ratio is the **median** across seeds (a single MCTS run is high-variance). |
| `max_dof` | `int` | `400` | `ge=10, le=1e6` | Active-DOF budget where both arms stop. |
| `max_steps` | `int` | `30` | `ge=1, le=1e4` | Max MCTS refinement steps. |
| `error_tolerance` | `float` | `1e-6` | `gt=0, lt=1` | Shared early-stop L2 tolerance. |
| `marking_fraction` | `float` | `0.5` | `gt=0, lt=1` | Dörfler bulk-marking θ (reused from `AMRConfig`). |
| `max_refinements` | `int` | `30` | `ge=1, le=1000` | Max Dörfler levels. |
| `n_candidate_elements` | `int` | `6` | `ge=1, le=256` | Top-ranked refinable elements MCTS chooses between (= action space). |
| `n_simulations` | `int` | `12` | `ge=1, le=4096` | MCTS simulations per accepted refinement. |
| `value_scale` | `float` | `4.0` | `gt=0, le=100` | tanh steepness for the encoded leaf value. |
| `c_puct` | `float` | `1.4` | `gt=0, le=10` | PUCT exploration constant. |
| `search_mode` | `str` | `single_agent` | `∈ {single_agent, zero_sum, legacy_adversarial}` | MCTS backup semantics. L-shape AMR is single-agent, so `single_agent` is correct. `legacy_adversarial` reproduces the pre-fix two-player backup (see AC3 note). |
| `max_l2_ratio_at_matched_dof` | `float` | `1.0` | `gt=0, le=10` | Primary gate value. |
| `output_dir` / `artifact_basename` | `str` | `results` / `lshape_mcts_vs_dorfler` | — | Committed artifact location. |

Named module-level constants (numerical-stability literals):
`src.pde.games.lshape_amr.DEFAULT_VALUE_SCALE`, `EPD_FLOOR`, `DEFAULT_MERGE_TOL`;
`src.research.lshape_amr_compare.RATIO_FLOOR`, `SEED_PRIME_STRIDE`.

## Acceptance Criteria

### AC1: Shared solver is not handicapped (backwards-compatible mask)
- **Given** `DorflerAMRSolver._solve_on_grid_2d` / `_compute_indicators_2d` with `inside=None`
- **When** solved on any rectangular grid
- **Then** the result is byte-for-byte identical to the pre-change behaviour (existing SBIR P40
  regression surface stays green); with an `inside` predicate, out-of-domain nodes are pinned to
  their Dirichlet value and out-of-domain elements are never marked.

### AC2: Both arms share solver / estimator / geometry / DOF accounting
- **Given** a fixed L-shape operator and seed
- **When** the Dörfler and MCTS arms run
- **Then** both call the same masked `make_solve_fn`, both count active (in-domain) nodes as DOF,
  and both measure the **area-weighted** (dual-cell) L2 over in-domain nodes — the *only*
  difference is the marking policy. (A plain node-wise RMS over-weights the densely-refined
  singular region and would bias the ratio; `_area_weighted_l2` recovers the mesh-independent norm.)

### AC3: MCTS refinement policy beats Dörfler at matched DOF (median over seeds)
- **Given** the CPU/demo config
- **When** `run_multiseed_comparison` completes
- **Then** the **median** `l2_error_ratio_at_matched_dof < 1.0` across `n_seeds` seeds (a single
  MCTS run is high-variance; the median is robust to an unlucky seed). Per-seed spread
  (`l2_ratio_seed_min/max/std`, `mcts_win_fraction`) is recorded for honesty.

> **Corrected-backup note (2026-07-10).** The originally committed headline (~11–14% win,
> median ratio ~0.89) was produced by MCTS running a **two-player adversarial backup on a
> single-agent game** — the F0 defect in `MCTSNode.backup`. Re-running the *same* canonical
> config under the corrected `search_mode="single_agent"` backup, over the same 5 seeds, gives:
>
> | Backup mode | Median L2 ratio @ matched DOF | Win fraction | Seed min / max |
> |---|---|---|---|
> | `legacy_adversarial` (pre-fix) | **0.8896** (≈11% win) | 0.80 | 0.7615 / 1.0299 |
> | `single_agent` (**corrected, committed**) | **0.9605** (≈4% win) | 0.80 | 0.8166 / 1.1157 |
>
> The corrected search is **still a win** at matched DOF (median < 1.0, primary gate passes) but a
> **smaller** one: ~4% rather than ~11%. The committed `results/lshape_mcts_vs_dorfler.{csv,png}`
> and the default config now use `single_agent`; `legacy_adversarial` remains selectable purely to
> reproduce the pre-fix number.

### AC4: Both comparisons are reported (honest dual metric)
- **Given** a completed run
- **When** metrics are recorded
- **Then** both `l2_error_ratio_at_matched_dof` (matched DOF, policy quality) and
  `error_per_dof_ratio_mcts_over_dorfler` (matched wall-clock, end-to-end) are present, plus
  per-arm convergence exponents and final DOF/L2 — no favourable-only headline.

### AC5: Reproducible artifacts
- **Given** a fixed seed
- **When** the scenario/CLI runs
- **Then** `results/lshape_mcts_vs_dorfler.{csv,png}` are written with the seed recorded in the CSV.

## Thresholds

Exactly what `LShapeAMRCompareConfig.get_default_thresholds()` returns (config is the single
source of truth; the AQA test asserts agreement):

| Metric | Operator | Value | Meaning |
|---|---|---|---|
| `l2_error_ratio_at_matched_dof` | `<` | `max_l2_ratio_at_matched_dof` (1.0) | MCTS refinement policy is at least as good as Dörfler at matched DOF. |

The matched-wall-clock ratio `error_per_dof_ratio_mcts_over_dorfler` is **recorded but not gated**:
each MCTS refinement costs `n_simulations` real solves, so it is expected `> 1` for an untrained
MCTS. Gating it would make the experiment a guaranteed failure that says nothing about policy
quality. Closing the wall-clock gap with a trained evaluator is Out of Scope (below).

## Regression Surface

```bash
# Scenario + config + harness + game + mask + CLI (CPU)
COVERAGE_CORE=pytrace pytest tests/research/test_lshape_notch_mask.py \
  tests/pde/test_lshape_amr_game.py tests/research/test_lshape_amr_compare.py \
  tests/poc/test_lshape_amr_compare_config.py tests/poc/test_lshape_amr_compare_scenario.py \
  tests/scripts/test_run_lshape_amr.py -v

# Shared-code regression: baselines.py is on the SBIR P40 surface — must stay green
pytest tests/research/test_baselines.py tests/research/test_baselines_2d.py \
  tests/research/test_pde_benchmarks.py tests/research/test_ns_baseline.py -v

# Per-module coverage gates (branch)
COVERAGE_CORE=pytrace pytest tests/research/test_lshape_amr_compare.py \
  --cov=src/research/lshape_amr_compare --cov-branch --cov-fail-under=85
COVERAGE_CORE=pytrace pytest tests/poc/test_lshape_amr_compare_config.py \
  tests/poc/test_lshape_amr_compare_scenario.py \
  --cov=src/poc/scenarios/lshape_amr_compare.py \
  --cov=src/poc/scenarios/lshape_amr_compare_config.py --cov-branch --cov-fail-under=85
```

## Out of Scope

- **Closing the matched-wall-clock gap.** An untrained MCTS trails Dörfler on end-to-end wall-clock
  (`error_per_dof_ratio_mcts_over_dorfler ≈ 15–55×`). A trained `FNetEvaluator` leaf value (removing
  the per-decision rollout cost) is the follow-up; lands with the trained-evaluator milestone.
- **True quadtree AMR.** The shared discretisation refines by tensor-product x/y edges (one
  element's refinement pulls a row+column), so this is a *controlled same-discretisation* MCTS-vs-
  Dörfler comparison, **not** the full local-quadtree L-shape benchmark (CLAUDE.md Next-Steps v2.1).
- **GPU run.** The comparison is CPU-only (scipy sparse); no GPU path.
