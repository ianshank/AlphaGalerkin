# Spec: Look-ahead vs greedy — the Gate 1 go/no-go

> **Status:** Accepted — **pre-registered**. Every threshold, budget, arm and formula below was
> fixed and committed before any policy-comparison run on either testbed. The commit that added
> this file is the pre-registration (`git log --diff-filter=A -- specs/lookahead_vs_greedy.spec.md`).
> Never change a threshold after seeing a result; a reasoned objection goes in the run report and
> the change, if any, goes in a *new* pre-registration.
> **Owner:** pde-solver / research
> **Primary module(s):** `src/research/lookahead_vs_greedy.py` (arms), `src/research/lookahead_vs_greedy_metrics.py`
> (matched-DOF reading, break-even, verdict), `src/poc/scenarios/lookahead_vs_greedy.py` (scenario),
> `scripts/run_lookahead_vs_greedy.py` (CLI)
> **Config class:** `src.poc.scenarios.lookahead_vs_greedy_config.LookaheadVsGreedyConfig`
> **Tracking:** `openspec/changes/lookahead-vs-greedy/`; `docs/business/COMMERCIALIZATION_PEER_REVIEW.md`
> §1 and §6 (Gate 1)

## Context

The cycle thesis is that **MCTS multi-step look-ahead beats classical greedy marking** for
error-driven adaptive refinement. It has never been tested. On the committed arena
(`results/mcts_classical_amr_arena.csv`) the MCTS trajectory is bit-for-bit the single-element
greedy trajectory: the legal set is pre-ranked by residual indicator, the prior is a softmax over
the same indicators, and the search never departs from greedy at any budget tried. Its 0.9532 ratio
against Dörfler measures *marking granularity* (one element per step against a bulk fraction), not
search. Gate 0.1 added the control that separates the two — the single-element greedy arm
(`src/research/greedy_control.py`) and the `decisions_diverging_from_greedy` counter.

Gate 1 is the experiment the thesis needs: on problems where greedy has room to be myopic, run the
search with room to grow several plies, against the **best** classical marking policy rather than
one Dörfler θ, and decide GO or NO-GO by criteria written down first.

## Question

On problems where it has room to differ from greedy, does MCTS look-ahead produce a lower
quadrature-L2 error at matched DOF than the best classical marking policy?

## Testbeds

| Id | Substrate | Operator | Why greedy could be myopic | Status |
|---|---|---|---|---|
| **T1** | `skfem_tri`, P1 | `lshape_poisson` — one 270° corner, `u = r^(2/3) sin(2θ/3)` | Control: the committed arena problem. Greedy residual marking is expected to be near-optimal for one local elliptic singularity. | Built |
| **T2** | `skfem_tri`, P1 | `zshape_poisson` — Z-tetromino, two 270° corners of strength `c1 = 1.0` (at (0, 0)) and `c2 = 0.25` (at (1, 0)) | Two competing singularities under a hard step budget: budget allocation across corners is a place a one-step rule can misallocate. | Built |
| **T3** | time-dependent substrate | a moving front (advection, or Burgers before shock formation) | The current residual is blind to where error *will* appear — the regime of anticipatory refinement. **The decisive testbed.** | **Deferred, not built** |

T3 is deferred. Consequently:

- **A NO-GO on T1 or T2 does not close the thesis.** These testbeds are elliptic with a reliable
  estimator, where adaptive-FEM optimality theory already predicts that estimator-driven marking is
  near-optimal; a NO-GO there is the expected control result.
- **A GO on either needs replication** (on T3, or on an independent seed set) before any claim is
  written anywhere.

## Arms

All arms share one substrate configuration, one operator, the same initial mesh and the same
estimator. Greedy and both MCTS arms drive the *same* game (`build_arena_episode`), each through its
own `FingerprintSolveCache`; the classical arms drive `run_refinement_sweep` on fresh substrates.

| Arm | CSV `method` | Definition | Runs |
|---|---|---|---|
| greedy | `greedy` | Single-element maximum marking: refine the legal element with the largest residual indicator, ties to the lowest index (`greedy_control.greedy_action`). | 1 (deterministic, CSV seed −1) |
| Dörfler | `dorfler_theta0.1`, `dorfler_theta0.3`, `dorfler_theta0.5` | Bulk marking at θ ∈ {0.1, 0.3, 0.5} (`run_classical_arm(policy="adaptive")`). | 1 each (CSV seed −1) |
| uniform | `uniform` | Refine every refinable element (reference). | 1 (CSV seed −1) |
| **MCTS-primary** | `mcts_primary` | `MCTS.get_action` with `add_noise=False`, `temperature=0`, `top_k_actions=4`, `n_simulations=64`, `search_mode=SINGLE_AGENT`, `ResidualPriorErrorValueEvaluator` (prior temperature 1.0), `c_puct=1.4`, `value_scale=4.0`, `use_intermediate_rewards=False`, tree reuse via `advance`. | 1 (deterministic; CSV seed = the config seed, 42) |
| **MCTS-robustness** | `mcts_robust` | Identical to MCTS-primary except **Dirichlet root noise ON** (`add_noise=True`, engine defaults α = 0.03, ε = 0.25, not tuned). `numpy`'s global RNG is seeded per run. | 5 seeds: 42, 1051, 2060, 3069, 4078 (`seed + i·1009`) |

The noise belongs to the **decision rule**, not to the game builder. `MCTSClassicalAMRArenaConfig`
locks `add_noise=False`; the game is built from an arena config that satisfies the scored locks and
the robustness arm passes `add_noise=True` to `MCTS.get_action`.

## Budgets

| Budget | Value | Reason |
|---|---|---|
| greedy / MCTS `max_steps` | **30** single-element refinements | Pre-registered (`DEFAULT_LOOKAHEAD_MAX_STEPS`). |
| classical `max_dof` | **1000** | A single-element skfem RGB refinement adds a handful of nodes (3–5 measured on both coarse meshes), so 30 steps from the coarse meshes (225 nodes for T1, 297 for T2) reach roughly 400–500 DOF; 1000 is twice that. Every classical trajectory must therefore span the matched DOF. The harness also **checks** it: if any classical arm stops below the largest DOF a game-driven arm reached, the run fails as an error rather than silently shrinking the matched DOF. |
| classical `max_refinements_classical` | **1000** levels | Runaway guard only. Every level refines at least one element and so adds at least one node; 1000 levels from ≥ 225 nodes cannot stop an arm before 1000 DOF. |
| greedy / MCTS `max_dof` (episode stop) | the classical `max_dof` (1000) | So only `max_steps` ends a game-driven episode. |
| `max_action_space` | 16384 | Above any element count in the window (the arena's value). |
| `solve_cache_max_entries` | 8192 | Above the ≤ 30 · 64 unique meshes one MCTS run can solve. |

## Metrics

### Matched DOF and the primary metric

For one MCTS run `m`, let `A = {m, greedy, dorfler_theta0.1, dorfler_theta0.3, dorfler_theta0.5, uniform}`.

- **Matched DOF** `N* = min over a in A of max_k N_{a,k}` — the largest DOF every arm reached.
- `L_a(N*)` is arm `a`'s quadrature L2 at `N*`, read on its log-log-interpolated trajectory
  (`src.research.lshape_amr_compare._interp_log`, the reader `compare_trajectories` uses).
- **Best classical** `B(N*) = min over the five classical arms of L_a(N*)`.
- **Primary metric:** `ratio_m = L_m(N*) / B(N*)` (`< 1`: MCTS better than every classical policy).

### Break-even reuse count `K*` against greedy

The question a buyer asks: if MCTS reaches an accuracy with fewer DOF but takes longer to build the
mesh, after how many downstream solves on that mesh does it pay back?

1. **Equal accuracy.** `ε* = max(L_m(N*), L_greedy(N*))` — the worse of the two at matched DOF, so
   both arms provably reach it at or before `N*`.
2. **First passage.** For `a ∈ {m, greedy}`, take the first committed segment `(k−1 → k)` of `a`'s
   trajectory with `e_k ≤ ε*` (or point 0 if `e_0 ≤ ε*`). With
   `t = (ln ε* − ln e_{k−1}) / (ln e_k − ln e_{k−1})`:
   `N_a = exp(ln N_{k−1} + t·(ln N_k − ln N_{k−1}))` and `W_a = W_{k−1} + t·(W_k − W_{k−1})`, where
   `W` is the arm's measured cumulative wall-clock.
3. **DOF saving** `ΔN = N_greedy − N_m` (positive: MCTS reaches `ε*` with fewer DOF).
4. **Cost model**, calibrated from the measured per-solve time: a downstream solve on an `N`-DOF mesh
   costs `t_α(N) = κ_α · N^α`, with `κ_α = (W_{g,K} − W_{g,0}) / Σ_{k=1..K} N_{g,k}^α` — the greedy
   arm's measured wall-clock over its `K` committed steps (each one refine and one solve), divided by
   the modelled cost of those solves.
5. **Break-even**
   `K*_α = max(0, W_m − W_greedy) / (t_α(N_greedy) − t_α(N_m))` when `ΔN > 0`, and **`K*_α = ∞`**
   otherwise (no DOF saving at equal accuracy never pays back). Reported for **α ∈ {1, 1.5}**.

Because `t_α` is strictly increasing for α > 0 and `κ_α > 0`, `K*_α` is finite **iff** `ΔN > 0`,
for every α. Criterion 4 is therefore gated as `primary_dof_saving_vs_greedy > 0`, and `K*_α` is
recorded (only when finite — an infinite value is recorded as `primary_break_even_finite = 0`). A
greedy arm whose per-solve time cannot be calibrated (zero steps or zero elapsed time) makes the
run an error, not a pass.

### Realized tree depth

Before each committed MCTS action, after that step's search: `subtree_depth(mcts.root)` — the
largest number of edges from the root to a node that **at least one simulation has visited**.
Children that expansion created but no simulation reached hold only a prior (no state was evaluated
there) and do not count. The tree includes what `advance` reused from earlier steps: it is the tree
the decision was made on. Recorded per step (sidecar) with max / median / min (metrics). Depth is
recorded, not inferred from the `n_simulations / top_k_actions` ratio.

### Divergence

`decisions_diverging_from_greedy`: the number of committed MCTS actions that differ from
`greedy_action` on the same state (`GreedyDivergence`, the one definition). On a ranked legal set
(`top_k_actions ≥ 1`) a divergence is evidence the search overrode the indicator; at
`top_k_actions = 0` a single simulation diverges by index-order tie-break alone, which is why the
config refuses `top_k_actions = 0`.

## GO / NO-GO (per testbed)

**GO requires ALL of:**

| # | Criterion | Metric | Operator | Value |
|---|---|---|---|---|
| C1 | MCTS-primary beats the best classical arm at matched DOF by ≥ 2% | `primary_l2_ratio_vs_best_classical` | `<=` | 0.98 |
| C2 | MCTS-primary departs from greedy at least once, on a ranked legal set (`top_k_actions = 4`) | `primary_decisions_diverging_from_greedy` | `>=` | 1 |
| C3 | MCTS-robustness: median ratio below 1 … | `robust_median_l2_ratio_vs_best_classical` | `<` | 1.0 |
| C3 | … and ratio below 1 on at least 3 of 5 seeds | `robust_seeds_below_best_classical` | `>=` | 3 |
| C4 | A finite break-even reuse count `K*` against greedy, i.e. a positive DOF saving at equal accuracy (`K*` reported for α ∈ {1, 1.5}) | `primary_dof_saving_vs_greedy` | `>` | 0.0 |

**NO-GO** otherwise. A criterion whose measurement is non-finite does not pass (fail-closed).

### Abort and invalid runs (no verdict)

- **Abort (per testbed).** Before any arm runs: `measure_adequacy` at θ = 0.5 with
  `adequacy_gate_for_operator(operator_name)` (T1: rate window (200, 4000); T2: (200, 5000);
  thresholds shared). If `gate_violations` is non-empty that testbed aborts: no comparison, no
  artifact, no verdict, exit code 3. Thresholds are not retuned.
- **Invalid (error, exit code 1).** A classical arm that does not span the largest game-driven DOF;
  a greedy per-solve time that cannot be calibrated; a non-finite recorded metric.

### Recorded, ungated

Matched-solves and matched-wall-clock ratios (against greedy and against the best classical arm,
via `compare_trajectories`); divergence counts per run; `K*_α` and its inputs (`ε*`, `N_m`,
`N_greedy`, `W_m`, `W_greedy`, `κ_α`); realized tree depth per step; every classical arm's L2 at the
primary run's matched DOF and which arm was best; the adequacy rates.

## Interpretation

- **GO** on a testbed means: on that problem, at these budgets, an untrained search beat every
  classical marking policy tried, departed from greedy while doing it, held up under root noise, and
  paid back its extra construction cost after a stated number of downstream solves. It does not mean
  look-ahead wins in general, and it is not a headline until replicated.
- **NO-GO** on T1/T2 means the search did not beat the best classical policy on elliptic testbeds
  with a reliable estimator — the regime where theory predicts greedy is near-optimal. It does not
  close the thesis; T3 decides.
- Adequacy rates are a precondition, not a look-ahead result.
- These runs use an untrained evaluator; a trained one is out of scope.

## Data Contract

Configured by `LookaheadVsGreedyConfig`. Shipped YAMLs:
`config/scenarios/lookahead_vs_greedy_lshape.yaml` (T1),
`config/scenarios/lookahead_vs_greedy_zshape.yaml` (T2). The pre-registered values are the field
defaults; the YAMLs restate them and an AQA test pins both against this table.

| Field | Type | Default | Bounds / invariant | Meaning |
|---|---|---|---|---|
| `substrate` | `SubstrateConfig` | `skfem_tri`, P1, `initial_refinements=2`, quadrature L2, squared marking, immutable meshes, cache 8192 | — | Shared by every arm. |
| `operator_name` | `Literal` | `lshape_poisson` | `poisson` / `lshape_poisson` / `zshape_poisson`; Z requires `lshape_scale = 1` | Testbed operator (`poisson` is for mechanism smokes only). |
| `max_steps` | `int` | **30** | `ge=1` | Greedy / MCTS single-element refinements. |
| `classical_max_dof` | `int` | **1000** | `ge=10` | Classical DOF budget; also the game-driven episode stop. |
| `max_refinements_classical` | `int` | **1000** | `ge=1` | Classical runaway guard. |
| `dorfler_thetas` | `tuple[float, ...]` | **(0.1, 0.3, 0.5)** | each in (0, 1), unique, non-empty | Dörfler arms. |
| `n_simulations` | `int` | **64** | `> top_k_actions` | PUCT simulations per committed step. |
| `top_k_actions` | `int` | **4** | **`ge=1`** (a ranked legal set) | Legal-set cap by residual indicator. |
| `c_puct` | `float` | 1.4 | `gt=0` | PUCT constant (the arena's). |
| `value_scale` | `float` | 4.0 | `gt=0` | Leaf-value steepness (the arena's). |
| `prior_temperature` | `float` | 1.0 | `gt=0` | Evaluator softmax temperature (its default). |
| `dirichlet_alpha` / `dirichlet_epsilon` | `float` | 0.03 / 0.25 | `gt=0` / `(0, 1]` | Robustness-arm root noise (engine defaults). |
| `robust_n_seeds` | `int` | **5** | `ge=1` | Robustness seeds. |
| `seed` / `seed_stride` | `int` | 42 / 1009 | — | Robust seeds are `seed + i·seed_stride`. |
| `temperature` | `float` | 0.0 | locked 0 | Action selection (both MCTS arms). |
| `search_mode` | `Literal` | `single_agent` | locked | Engine default is `ZERO_SUM`; never omit. |
| `evaluator_name` | `Literal` | `ResidualPriorErrorValueEvaluator` | locked | Headline leaf evaluator. |
| `require_adequacy_precondition` | `bool` | True | requires `skfem_tri` + `lshape_poisson`/`zshape_poisson` | Abort gate. |
| `adequacy_theta` | `float` | 0.5 | `(0, 1)` | Dörfler θ of the adequacy measurement. |
| `max_primary_ratio` | `float` | **0.98** | `gt=0` | C1. |
| `min_primary_divergences` | `int` | **1** | `ge=1` | C2. |
| `max_robust_median_ratio` | `float` | **1.0** | `gt=0` | C3 (median). |
| `robust_win_ratio` | `float` | **1.0** | `gt=0` | A robustness seed "beats" the best classical arm iff its ratio `<` this. |
| `min_robust_wins` | `int` | **3** | `1 ≤ … ≤ robust_n_seeds` | C3 (count). |
| `min_dof_saving_vs_greedy` | `float` | **0.0** | `ge=0` | C4 (strictly greater). |
| `break_even_alphas` | `tuple[float, ...]` | **(1.0, 1.5)** | each `gt=0` | Downstream cost exponents for `K*`. |
| `output_dir` / `artifact_basename` | `str` | `results` / per testbed | no extension | Artifact location. |

Named constants: `DEFAULT_LOOKAHEAD_MAX_STEPS = 30`, `DEFAULT_CLASSICAL_MAX_DOF = 1000`,
`DEFAULT_DORFLER_THETAS = (0.1, 0.3, 0.5)`, `DEFAULT_LOOKAHEAD_N_SIMULATIONS = 64`,
`DEFAULT_LOOKAHEAD_TOP_K = 4`, `DEFAULT_ROBUST_N_SEEDS = 5`, `DEFAULT_BREAK_EVEN_ALPHAS = (1.0, 1.5)`,
plus the five threshold defaults.

## Acceptance Criteria

### AC1: Pre-registration precedes results
- **Given** this spec, committed before any comparison run
- **When** the shipped YAMLs and `LookaheadVsGreedyConfig` defaults are read
- **Then** every pre-registered value (budgets, arms, seeds, thresholds, α) equals this spec

### AC2: Adequacy abort
- **Given** `require_adequacy_precondition=True` and a substrate failing `gate_violations`
- **When** the harness starts
- **Then** it raises before any arm runs, writes no artifact, and the CLI exits 3

### AC3: Shared substrate, private caches, noise in the decision rule
- **Given** the arms
- **When** they run
- **Then** all share substrate config, operator, initial mesh and estimator; greedy and every MCTS
  run solve through distinct caches; the game is built from an arena config with `add_noise=False`
  and only the robustness rule passes `add_noise=True` to `get_action`

### AC4: Matched DOF, best classical, span check
- **Given** finished trajectories
- **When** a run is scored
- **Then** `N*` is the largest DOF every arm reached, `B(N*)` the minimum over the five classical
  arms, and a classical arm that does not reach the largest game-driven DOF is an error

### AC5: Depth and divergence per committed step
- **Given** an MCTS run
- **When** each action is committed
- **Then** its realized tree depth (visited nodes only) and greedy action are recorded, and the
  divergence count equals the number of steps whose action differed from greedy

### AC6: Break-even
- **Given** two trajectories
- **When** `K*_α` is computed
- **Then** it follows the formula above, is finite iff `ΔN > 0`, and is reported for α ∈ {1, 1.5}

### AC7: Verdict
- **Given** the measured criteria
- **When** the verdict is formed
- **Then** GO iff every threshold passes; each criterion failing alone yields NO-GO; the verdict is
  monotone (lowering a ratio, or raising a divergence count, win count or DOF saving, never turns
  GO into NO-GO)

### AC8: Artifacts and provenance
- **Given** a completed testbed
- **When** artifacts are written
- **Then** the CSV uses the arena column schema with the method labels above, a PNG is drawn, and
  the `.run.json` is written through `RunRecorder` (git snapshot before any write; under
  `--proposal-grade`, refuse to start on a dirty tree and re-verify the sidecar from disk) carrying
  every criterion value and the verdict

### AC9: Exit code is not the verdict
- **Given** a completed run
- **When** the CLI exits
- **Then** GO and NO-GO both exit 0; an adequacy abort exits 3; a non-proposal-grade run exits 2;
  any other error exits 1

### AC10: AQA
- **Given** `get_default_thresholds()`
- **When** compared to the Thresholds table
- **Then** they agree exactly (names, operators, values)

## Thresholds

These are exactly what `LookaheadVsGreedyConfig.get_default_thresholds()` returns; GO iff all pass.

| Metric | Operator | Value | Meaning |
|---|---|---|---|
| `primary_l2_ratio_vs_best_classical` | `<=` | `0.98` | C1 |
| `primary_decisions_diverging_from_greedy` | `>=` | `1` | C2 |
| `robust_median_l2_ratio_vs_best_classical` | `<` | `1.0` | C3, median over seeds |
| `robust_seeds_below_best_classical` | `>=` | `3` | C3, seeds with ratio `< 1.0` |
| `primary_dof_saving_vs_greedy` | `>` | `0.0` | C4, finite `K*` |

## Artifacts

`results/lookahead_vs_greedy_lshape.{csv,png,run.json}` (T1) and
`results/lookahead_vs_greedy_zshape.{csv,png,run.json}` (T2), produced with `--proposal-grade` from a
clean tree and frozen in `results/MANIFEST.sha256`. The verdict and every criterion value live in
the `.run.json` (`metrics` and `notes`), which the manifest hashes.

## Regression Surface

```bash
pytest tests/mcts/test_tree_depth.py tests/research/test_lookahead_vs_greedy_metrics.py \
  tests/research/test_lookahead_vs_greedy.py tests/research/test_lookahead_vs_greedy_verdict.py \
  tests/poc/test_lookahead_vs_greedy_config.py tests/poc/test_lookahead_vs_greedy_scenario.py \
  tests/scripts/test_run_lookahead_vs_greedy.py tests/e2e/test_lookahead_vs_greedy.py \
  tests/docs/test_sidecar_config_hash.py -v -m "not gpu_required and not fem_required"
# skfem half (CI test-extras): tests/e2e/test_lookahead_vs_greedy.py -m fem_required
```

> **Amended 2026-10-08, after the runs:** the verdict suite
> (`tests/research/test_lookahead_vs_greedy_verdict.py`) and the sidecar-hash guard
> (`tests/docs/test_sidecar_config_hash.py`) were missing from this list. Adding test files to
> the regression command changes no question, testbed, arm, budget, metric or threshold above.

## Out of Scope

- T3 (moving front) — the decisive testbed; needs a time-dependent substrate.
- Goal-oriented refinement (dual-weighted greedy baseline).
- A trained evaluator; tuning the prior temperature, `c_puct`, noise or budgets after any result.
- Changing `MCTS`'s default `search_mode` off `ZERO_SUM`.
- Any charter evidence, novelty or deviations row (owned elsewhere; proposed in the run report).
