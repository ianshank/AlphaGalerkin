# Spec: Honest zero-shot transfer — AlphaGalerkin operator vs a retrained CNN

> **Status:** Implemented
> **Owner:** pde-solver
> **Primary module(s):** `src/research/transfer_baseline_compare.py`, `src/experiments/cnn_baseline.py`, `src/poc/scenarios/transfer_baseline_compare.py`
> **Config class:** `src.poc.scenarios.transfer_baseline_compare_config.TransferBaselineCompareConfig`
> **Tracking:** branch `claude/codebase-sprawl-benchmark-7kymbk`

## Context

AlphaGalerkin's headline result — *"Zero-Shot Transfer MSE 0.000209, 240× better than
threshold"* — was **fabricated**. It lived only as a hardcoded markdown cell in
`notebooks/AlphaGalerkin_Demo.ipynb` (no code ever computed it), was backed by **no
committed artifact**, and its three hardcoded copies disagreed with each other. "240×"
was merely `0.05 / 0.000209` — the achieved MSE measured against a fixed pass bar, *not*
against any competing method. A "240× below an arbitrary threshold" claim is not a result.

This feature replaces that self-comparison with a **committed, CI-gated, falsifiable**
benchmark against an honest baseline. The real claim is:

> The resolution-independent `PhysicsOperator`, trained ONLY at `train_resolution` (9×9)
> and applied zero-shot at `target_resolution` (19×19), matches or beats a discrete CNN
> **retrained at** 19×19.

The discrete CNN is the honest *specialist* foil: the standard discrete workflow retrains
it at each target resolution (the Poisson discretisation uses grid spacing `h = 1/(n+1)`, so
a fixed-pixel-radius stencil learned at 9×9 sits at a different physical scale at 19×19), and
that per-resolution retraining is exactly the workflow the operator claims to avoid. Whether
the retraining actually *buys accuracy* is **measured, not assumed** — the benchmark records
a retrained CNN AND a zero-shot CNN so the comparison is honest about whether the operator's
resolution-independence is even necessary.

**Measured operator zero-shot MSE** (spike, `scripts/demo_transfer.py`, d_model=128 /
n_layers=4 / 2000 samples / 50 epochs / seed 42): 9×9 = 2.5e-6, 13×13 = 2.0e-4,
**19×19 = 3.9e-4** — i.e. the operator *does* transfer (well under 0.05), but the honest
19×19 number is ~1.9× the fabricated 0.000209.

**Honest committed result (CI-tripwire run, 3 seeds, median seed).** On this in-distribution
Poisson task the operator does **not** win on accuracy: its zero-shot 19×19 MSE (~2.3e-3) is
~14× a retrained CNN's (~1.6e-4), and a *zero-shot* CNN (~7.7e-5) is more accurate still. The
finding is real and reported faithfully — **the operator's value is zero-retraining (one
model at any resolution), not peak accuracy.** The gated operator-vs-CNN ratio is therefore
committed and regression-gated as a *ceiling* (a broken operator arm would blow past it), not
as a `< 1` win claim.

## User Story

**As a** reviewer assessing the AlphaGalerkin zero-shot-transfer thesis,
**I want** an honest, reproducible head-to-head of the operator (zero-shot) against a CNN
retrained at the target resolution, with the result committed and CI-gated,
**so that** I can judge whether resolution-independence buys real accuracy — win or lose —
instead of trusting a number no code computed.

## Data Contract

Configured by `TransferBaselineCompareConfig` (`BaseScenarioConfig` subclass). Every
tunable is a typed Pydantic `Field` — no hardcoded values. Key fields:

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `train_resolution` | `int` | `9` | `ge=3, le=25` | Grid the operator (and CNN zero-shot arm) trains on. |
| `target_resolution` | `int` | `19` | `ge=5, le=51`, `> train` | Zero-shot / retrain target; the headline ratio is here. |
| `secondary_resolutions` | `list[int]` | `[9, 13]` | each `>= 3` | Extra resolutions for the operator's recorded zero-shot curve. |
| `n_train_samples` | `int` | `5000` | `ge=64` | Training samples per arm (matched data volume). |
| `n_eval_samples` | `int` | `500` | `ge=10` | Samples in the shared held-out eval set. |
| `n_seeds` | `int` | `5` | `ge=1, le=64` | Seeds swept; the headline (ratio + absolutes) is taken from the median-ranked seed (odd counts → the true median). |
| `n_epochs` | `int` | `100` | `ge=1` | Training epochs per arm (matched training budget). |
| `d_model` … `use_fnet` | — | 128/4/4/64/10.0/True | — | Operator architecture (mirrors `TransferScenarioConfig`). |
| `cnn_n_layers` | `int` | `6` | `ge=0, le=32` | CNN residual blocks. |
| `cnn_kernel_size` | `int` | `3` | `ge=1, le=7`, **odd** | CNN convolution kernel. |
| `cnn_channels` | `int \| None` | `None` | `ge=1` if set | CNN width; `None` auto-matches the operator's parameter count. |
| `cnn_param_match_tolerance` | `float` | `0.15` | `gt=0, le=1` | Relative band for the CNN param-count match (secondary sanity). |
| `matched_budget_mode` | `Literal["grad_steps","wall_clock"]` | `grad_steps` | — | Matched-compute CNN budget (grad_steps == primary arm; wall_clock retrains for the operator's seconds). |
| `transfer_ratio_pass_threshold` | `float` | `1.0` | `gt=0, le=100` | Primary gate ceiling (strictly `<`). Default 1.0 = strong win claim; calibrate to a regression ceiling above the measured median when the operator loses (CI config: 30.0). |
| `output_dir` / `artifact_basename` | `str` | `results` / `transfer_baseline_compare` | — | Committed artifact location. |

Named module-level constants (numerical-stability literals):
`src.research.transfer_baseline_compare.TRANSFER_RATIO_FLOOR`, `SEED_PRIME_STRIDE`;
`src.experiments.cnn_baseline.DEFAULT_CHANNEL_SEARCH_SPAN`.

## Acceptance Criteria

### AC1: Both CNN workflows are recorded so the retraining question is honest (mechanism)
- **Given** a CNN trained at `train_resolution`
- **When** it is evaluated at `target_resolution` *and* the same CNN is separately retrained at `target_resolution`
- **Then** both `mse_cnn_zeroshot` and `mse_cnn_retrained` are recorded, so whether the discrete baseline *needs* retraining is measured rather than assumed. **The direction is not asserted** — the earlier "the CNN cannot transfer / is materially worse zero-shot" claim was an over-statement: on the committed run the fully-convolutional CNN actually transfers *better* zero-shot than when retrained at this budget, and both beat the operator's zero-shot accuracy. The operator's advantage is architectural (zero-retraining), which AC4's dual metric records without a favourable-only headline.

### AC2: Both arms share the held-out eval set and ground truth (fairness)
- **Given** a fixed target resolution and seed
- **When** the operator and CNN arms are scored
- **Then** both evaluate on the identical `PoissonDataset` (eval seed = `eval_seed_base + resolution`, a function of resolution only) over the identical `PoissonSample.potential` targets, and both train on matched data volume and matched training budget — the only difference is the model and its training resolution.

### AC3: The gated ratio and the headline absolutes come from one real (median) seed
- **Given** `n_seeds` seeds derived via `resolved_seeds`
- **When** `run_multiseed_transfer_comparison` completes
- **Then** the gated `transfer_mse_ratio_<t>x<t>` **and** every headline absolute
  (`mse_alphagalerkin_zeroshot`, `mse_cnn_retrained`, `mse_cnn_zeroshot`) come from the same
  **representative (median-ranked) seed**, so dividing the published absolutes reproduces the
  published ratio exactly — for any `n_seeds` parity. With an odd `n_seeds` (the CI config uses
  3, the full config 5) the representative *is* the statistical median (`transfer_ratio_seed_median`).
  Per-seed spread (`transfer_ratio_seed_min/max/std/median`, `alphagalerkin_win_fraction`) is recorded for honesty.

### AC4: Both comparisons are reported (honest dual metric)
- **Given** a completed run
- **When** metrics are recorded
- **Then** both `transfer_mse_ratio_<t>x<t>` (matched training budget, gated) and
  `transfer_mse_ratio_<t>x<t>_matched_compute` (recorded, not gated) are present, plus
  the raw absolutes (`mse_alphagalerkin_zeroshot`, `mse_cnn_retrained`, `mse_cnn_zeroshot`),
  the operator zero-shot curve, and `param_count_ratio` — no favourable-only headline.

### AC5: Reproducible artifacts
- **Given** a fixed seed set
- **When** the scenario/CLI runs
- **Then** `results/transfer_baseline_compare.{csv,png}` are written with the seed in each CSV row.

### AC6: Spec ↔ config agreement (AQA)
- **Given** `TransferBaselineCompareConfig`
- **When** `get_default_thresholds()` is called
- **Then** it returns exactly one `MetricThreshold` named `transfer_mse_ratio_<target>x<target>`, operator `<`, value `transfer_ratio_pass_threshold` — matching the Thresholds table below.

## Thresholds

Exactly what `TransferBaselineCompareConfig.get_default_thresholds()` returns (config is the
single source of truth; the AQA test asserts agreement):

| Metric | Operator | Value | Meaning |
|---|---|---|---|
| `transfer_mse_ratio_<t>x<t>` | `<` | `transfer_ratio_pass_threshold` (default 1.0; **CI config 30.0**) | Operator-zero-shot / CNN-retrained MSE ratio at the target resolution must be strictly below this. The 1.0 default is the strong win claim; the CI config calibrates it to a **regression ceiling** (see honest-loss handling). |

The matched-compute ratio `transfer_mse_ratio_<t>x<t>_matched_compute` is **recorded but not
gated**: in `wall_clock` mode it depends on machine speed, so gating it would test training
throughput rather than architecture quality.

**Honest-loss handling (this benchmark loses on accuracy).** A measured CI run shows the
operator does **not** beat a retrained CNN — the median ratio is ~14 (per-seed 14.1–23.3). The
negative result is reported faithfully (mirroring `specs/lambda_scheduling.spec.md`) and
`transfer_ratio_pass_threshold` is set to a **regression ceiling above the measured median**
(30.0 in `transfer_baseline_compare_ci.yaml`) so the scenario gate flags a *broken operator
arm* rather than asserting a false `< 1` win. A `<` operator against the median itself would be
self-contradictory (`median < median` is false), which is why the ceiling carries drift/seed
headroom. Independently, the **CI regression gate** (`--baseline` diff in
`scripts/run_transfer_baseline_compare.py`) fails only on regression from the committed
`config/baselines/transfer_ci.json` — the honest loss is a green CI run, it just shows in the
recorded number.

## Regression Surface

```bash
COVERAGE_CORE=pytrace pytest tests/experiments/test_cnn_baseline.py \
  tests/research/test_transfer_baseline_compare.py \
  tests/poc/test_transfer_baseline_compare_config.py \
  tests/poc/test_transfer_baseline_compare_scenario.py \
  tests/scripts/test_run_transfer_baseline_compare.py -v

# per-module branch coverage gate (>= 85)
COVERAGE_CORE=pytrace pytest tests/research/test_transfer_baseline_compare.py \
  tests/poc/test_transfer_baseline_compare_config.py \
  tests/poc/test_transfer_baseline_compare_scenario.py \
  --cov=src/research/transfer_baseline_compare --cov=src/experiments/cnn_baseline \
  --cov=src/poc/scenarios/transfer_baseline_compare \
  --cov=src/poc/scenarios/transfer_baseline_compare_config --cov-branch --cov-fail-under=85
```

## Out of Scope

- **The 5-seed / full-sample headline number.** The committed `results/*.{csv,png}` and
  `config/baselines/transfer_ci.json` come from the fast `transfer_baseline_compare_ci.yaml`
  (reproducible in CI). The definitive higher-fidelity headline is the documented runbook
  `config/scenarios/transfer_baseline_compare_full.yaml` (~1 hour on CPU; record
  `config/baselines/transfer_headline.json` from it).
- **Stronger neural baselines.** A retrained U-Net or the existing FNO/DeepONet
  (`src/research/extra_solvers/neural_op.py`) are natural follow-on baselines; the CNN is the
  honest *discrete* foil the finding asked for. Deferred.
- **GPU run.** The comparison is CPU-only by default; a CUDA run is a device override, not a
  separate code path.
- **The Go board-size transfer arm.** The second thesis (win-rate/policy-agreement at 9/13/19
  vs a real Go opponent) is a separate follow-up PR — it needs an external Go engine.
