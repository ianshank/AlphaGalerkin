# Spec: λ-window scheduling ablation (`src/thermo/`)

**Status:** Implemented — **NEGATIVE result** (kill criterion triggered).
**Feature:** The first non-PDE `RefinementGame`. A falsification test for the generality of
`src/refinement/`: does single-agent MCTS planning beat greedy variance-weighted allocation for
free-energy (BAR/FEP) λ-window sample scheduling? **Not a chemistry product.**

## Problem

A free-energy calculation partitions `λ ∈ [0, 1]` into windows. Window *i* with `n_i` samples has
standard error `σ_i = c_i / sqrt(n_i)`; the ΔG standard error is `sqrt(Σ c_i² / n_i)`. Scheduling
chooses windows + sample counts to minimise that under a budget. A `VarianceSurrogate` supplies
`c(window)`.

## Kill criterion (binding, written before the result existed)

> If MCTS does not beat greedy at `surrogate_bias = 0.25` on the analytic surrogate, `src/thermo/`
> is a negative result: the thesis (MCTS lookahead helps λ-scheduling) is falsified for this
> configuration, the code is retained only as the falsification harness, and **no capability is
> claimed**.

## Result (measured)

`python -m scripts.run_lambda_scheduling` (5 seeds, matched full sample budget, tolerance at the
0.05 kcal/mol physical floor so the budget binds). Committed artifacts:
`results/lambda_scheduling.{png,csv}`.

| surrogate_bias | final ΔG-stderr ratio MCTS / greedy | verdict |
|---|---|---|
| 0.00 (perfect surrogate) | **2.05** | MCTS ~2× **worse** |
| 0.25 (the binding gate) | **2.05** | MCTS ~2× **worse** — kill criterion **not met** |

**Robustness to the reward-scale confound (checked, not assumed).** MCTS backs up
`R + γ^d·V(leaf)`; the per-edge shaped reward `R` is order `1e-3` while a terminal `V` from
`get_winner` is order `1`. To stop a non-converged terminal from swamping the shaped signal,
`get_winner` returns **0** (neutral) unless the schedule actually converged (then `+1`), and the
per-edge cost is keyed on the window-count delta (a split adds a window), not on a DOF side-effect.
Re-running with this neutralised terminal leaves the verdict **unchanged** (ratio 2.00 → 2.05), so
the negative result is driven by genuine over-splitting, not by a reward-scale artifact.

**MCTS loses even with a perfect surrogate (bias 0).** Root cause: with a uniform-prior
`RandomEvaluator` the search has no signal that splitting is usually harmful, so it splits to the
window cap (16 vs greedy's 6), fragmenting the same sample budget into thinner-sampled windows.
Greedy variance-weighted allocation converges to the near-optimal proportional allocation
`n_i ∝ c_i`; an uninformed lookahead cannot beat it and the split actions are a trap.

**Honest caveat on the bias sweep.** A purely *multiplicative* surrogate bias `(1 + b)` is
**scale-invariant** for allocation (it does not change `argmax_i c_i²/n_i`), so it does not
mis-inform greedy — the ratio is bias-invariant. Genuine mismatch requires *shape* distortion
(`MismatchedSurrogate(noise_amplitude=...)`). This does not change the verdict, because MCTS already
loses at zero mismatch.

**What would change the verdict** (out of scope, same conclusion as the L-shape wall-clock gap): a
*trained* evaluator that has learned splitting is usually bad. That is the `OperatorSurrogate` /
trained-net direction (P5), funded only if the mechanics warrant it — they do not, here.

## Data contract

- `HardnessProfileConfig` — analytic `baseline + peak_amplitude·gaussian(center, width)`.
- `SchedulingParams` (frozen) — game/harness knobs (windows, batch, budget, split credit, tol).
- `LambdaSchedulingConfig` (Pydantic) — every knob typed; `error_tolerance` validated to the
  **kcal/mol** range `[0.05, 1.0]` (rejects the inherited `1e-4`); `primary_bias ∈ surrogate_bias_sweep`.
- `RefinementState.values` packs `(K, 3)` `[lo, hi, n]` rows.

## Acceptance criteria (mechanics — these gate CI; the losing headline does not)

- **AC1 — deterministic `apply_action`.** Any legal action sequence replayed twice from
  `get_initial_state()` yields an identical state (Hypothesis). The invariant node identity depends on.
- **AC2 — monotone under allocate.** Allocation never increases the total standard error.
- **AC3 — non-monotone under split, reachable.** A split with `sample_split_credit = 0.5` conserves
  samples and *increases* total variance; the property test asserts this is **reachable**, so nobody
  "fixes" it by crediting children with the parent's `n` (which manufactures free variance MCTS
  would exploit).
- **AC4 — surrogate correctness.** Analytic `c > 0`; mismatched scales by `(1 + bias)` + deterministic
  noise (no RNG); recorded returns nearest-window `c`; `OperatorSurrogate()` raises without a fit.
- **AC5 — schedulers respect the budget** and greedy ≥ uniform at matched full budget.

## Thresholds

`LambdaSchedulingConfig.get_default_thresholds()` returns the falsifiable gates
(`dG_stderr_ratio_mcts_over_greedy_median < 1.0`, `..._at_bias_0p25_median < 1.0`,
`mcts_win_fraction ≥ 0.6`). **These are documented as FAILING** — they are the honest gates the
negative result did not clear. They are *not* wired into a CI-gated scenario (that would make CI red
for a documented negative result); CI gates the mechanics (AC1–AC5) only.

## Out of scope

- Real MD (OpenMM/GROMACS) — `RecordedSurrogate` replays a committed fixture.
- A trained evaluator — the only thing that would plausibly flip the verdict.
