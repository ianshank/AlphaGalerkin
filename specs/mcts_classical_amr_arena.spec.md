# Spec: MCTS vs classical AMR arena

> **Status:** Implemented
> **Owner:** pde-solver / research
> **Primary module(s):** `src/research/mcts_classical_amr_arena.py`, `src/poc/scenarios/mcts_classical_amr_arena.py`
> **Config class:** `src.poc.scenarios.mcts_classical_amr_arena_config.MCTSClassicalAMRArenaConfig`
> **Tracking:** `openspec/changes/mcts-classical-amr-arena/`, `docs/FOCUS.md`

## Context

The cycle thesis is that **MCTS multi-step look-ahead beats classical greedy
marking for error-driven AMR**. That thesis was previously measured only on a
tensor-product substrate where adaptive Dörfler is *worse* than uniform
(`results/lshape_adaptive_vs_uniform.*`) and MCTS loses to Dörfler
(`results/lshape_mcts_vs_dorfler.csv`, median ratio 1.0996). Those numbers are
honest for a defective discretisation and **non-informative for element-local
policy**.

`SkfemTriSubstrate` plus the classical adequacy gate made the substrate
measurable. This spec pre-registers the **policy** comparison on that substrate
before any headline is written. Adequacy rates (`N^-1.31`, ~10× at θ=0.5 over
DOF ∈ (200, 4000)) are **gate evidence**, not a look-ahead win.

## User Story

**As a** researcher,
**I want** a pre-registered, shared-substrate MCTS vs Dörfler comparison on
`skfem_tri`,
**so that** a win or an honest negative can lift the FOCUS freeze without
re-poisoning the measurement.

## Hypothesis and falsifiers

**Hypothesis.** Untrained MCTS (residual-weighted legal prior + error-per-DOF
leaf value) produces a **lower** quadrature L2 than Dörfler bulk marking at
matched DOF on `SkfemTriSubstrate` / L-shaped Poisson, θ=0.5.

| Verdict | Meaning |
|---|---|
| **Win** | median `l2_error_ratio_at_matched_dof` `< 1.0` |
| **Null / honest negative** | median ratio `>= 1.0` (MCTS does not beat Dörfler at matched DOF) |
| **Inconclusive** | arms share no DOF, or `n_seeds < 1` after failures |
| **Abort** | `gate_violations()` nonempty on this exact substrate/θ/adequacy window — **do not retune thresholds** |

Freeze lifts on Win **or** Null. Smoke passing is not a freeze lift.

> **CORRECTED (2026-10-08), annotation only — the table above is the pre-registration
> and is unchanged.** A **Win** on this metric does not, by itself, evidence the
> hypothesis: MCTS / Dörfler mixes *marking granularity* (one element per step vs a
> bulk fraction) with *search*. The single-element greedy control
> (`include_greedy_control`, below) separates them — MCTS / greedy isolates what the
> search adds — and on the committed artifact it shows the search added nothing
> (see *Measured result*).

## Data Contract

Configured by `MCTSClassicalAMRArenaConfig`. Headline YAML:
`config/scenarios/mcts_classical_amr_arena.yaml`. CI smoke:
`config/scenarios/mcts_classical_amr_arena_ci.yaml` (`require_adequacy_precondition: false`).

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `substrate.kind` | `Literal` | `skfem_tri` | `skfem_tri`/`tensor_grid` | Shared substrate. Adequacy abort requires `skfem_tri`. |
| `substrate.element_type` | `Literal` | `P1` | P1–P3 | Lagrange order (skfem). |
| `operator_name` | `Literal` | `lshape_poisson` | poisson / lshape_poisson | Exact-solution operator. |
| `marking_fraction` | `float` | `0.5` | `(0, 1)` | Dörfler θ. Quote with every rate. |
| `max_dof` | `int` | `600` | `ge=10` | **Policy** DOF budget (not the adequacy window). |
| `max_steps` | `int` | `12` | `ge=1` | MCTS single-element refine cap. |
| `n_simulations` | `int` | `8` | `ge=1` | PUCT simulations per accepted step. |
| `top_k_actions` | `int` | `8` | `ge=0` | Residual-ranked legal set. |
| `max_action_space` | `int` | `16384` | `ge=1` | Must exceed `n_units` in the window. |
| `n_seeds` | `int` | `3` | `ge=1` | Median-over-seeds headline. |
| `seed_stride` | `int` | `1009` | `ge=1` | Offset between seeds. |
| `add_noise` | `bool` | `False` | locked False | Scored step. |
| `temperature` | `float` | `0.0` | locked 0 | Scored step. |
| `search_mode` | `Literal` | `single_agent` | locked | Engine default is `ZERO_SUM` (F0). |
| `evaluator_name` | `Literal` | `ResidualPriorErrorValueEvaluator` | locked | Headline leaf evaluator. |
| `use_intermediate_rewards` | `bool` | `False` | locked False | Leaf-value bootstrap only. |
| `require_adequacy_precondition` | `bool` | `True` | — | Abort if the adequacy gate fails. |
| `include_greedy_control` | `bool` | `True` | — | Run the single-element greedy control arm (added 2026-10-08, peer-review Gate 0.1). `False` writes the legacy uniform/dorfler/mcts CSV rows and columns, unperturbed by the control. The sidecar is **not** the legacy one either way: the config records this field, so `config_hash` differs from a sidecar written before it existed, and `decisions_diverging_from_greedy` is recorded either way. Divergence is recorded, not gated, and is excluded from the CLI's regression baseline (it has no better direction). |
| `max_l2_ratio_at_matched_dof` | `float` | `1.0` | `gt=0` | Sole gated threshold. |

Named constants: `HEADLINE_EVALUATOR_NAME`, `DEFAULT_SEED_STRIDE=1009`,
`DEFAULT_MARKING_FRACTION=0.5`, `DEFAULT_MAX_ACTION_SPACE=16384`.

**Headline leaf evaluator.** `ResidualPriorErrorValueEvaluator`
(`src/research/substrates/residual_evaluator.py`): softmax of residual
indicators over legal actions; leaf value from the trailing error-per-DOF slot
of `SubstrateRefinementGame.to_tensor`. **Forbidden as the published arm:**
`EncodedValueEvaluator` (reads `state[0]`, which is element-0's indicator on
this encoding) and `RandomEvaluator` (value 0.0).

**Caches.** One `FingerprintSolveCache` per MCTS seed. Classical arms call
`substrate.solve` through `run_refinement_sweep` and do **not** share that
cache. Cache **misses** count unique meshes; `apply_action` replay is
`n_cache_hits`.

**Adequacy window** (abort only): `AdequacyGateConfig.rate_fit_dof_range` =
`(200, 4000)` at the same θ. Policy comparison quotes `max_dof` / `max_steps`,
which may be smaller. Do not present policy ratios as if they were fitted over
`(200, 4000)`.

## Acceptance Criteria

### AC1: Adequacy abort
- **Given** `require_adequacy_precondition=True` and a host that fails `gate_violations`
- **When** the harness starts
- **Then** it raises `AdequacyPreconditionError` and writes no policy headline

### AC2: Shared substrate / θ / dof_convention
- **Given** the pinned YAML
- **When** uniform, Dörfler, and MCTS run
- **Then** they share `kind`, θ, operator, and the manifest records `dof_convention`

### AC3: Named evaluator and search flags
- **Given** the MCTS arm
- **When** `MCTS` is constructed
- **Then** `search_mode` is `SINGLE_AGENT`, `add_noise=False`, `temperature=0`,
  `use_intermediate_rewards=False`, evaluator is `ResidualPriorErrorValueEvaluator`,
  and `max_action_space >= n_units`

### AC4: Metric hierarchy
- **Given** finished trajectories
- **When** ratios are computed
- **Then** the gated metric is matched-DOF quadrature L2; matched-solves (cache
  misses) and wall-clock are recorded ungated

### AC5: Artifact contract
- **Given** a proposal-grade run
- **When** `*.run.json` is claim-committed
- **Then** `assert_proposal_grade` rejects `dirty is not False` and
  `config_hash == "unknown"`. Collectors in `run_manifest.py` still never raise.

### AC6: AQA
- **Given** `get_default_thresholds()`
- **When** compared to this spec's Thresholds table
- **Then** they agree (one `<` gate on `l2_error_ratio_at_matched_dof`)

### AC7: README/charter guard
- **Given** a new MCTS-vs-Dörfler AMR ratio in README or the charter
- **When** the alignment tests run
- **Then** the claim cites `results/*` and a sibling `.run.json` unless the
  citation is a documented pre-manifest golden labelled non-informative

## Thresholds

| Metric | Operator | Value | Meaning |
|---|---|---|---|
| `l2_error_ratio_at_matched_dof` | `<` | `max_l2_ratio_at_matched_dof` (default 1.0) | MCTS quadrature L2 / Dörfler at matched DOF. Honest FAIL is a valid outcome. |

Matched-solves and wall-clock ratios are **not** gated.

## Measured result (Phase 2, committed)

Proposal-grade run on `19609d4` (`results/mcts_classical_amr_arena.{csv,run.json}`,
`dirty: false`). θ=0.5, policy `max_dof=600`, `max_steps=12`, `n_simulations=8`,
`top_k_actions=8`, `ResidualPriorErrorValueEvaluator`, `search_mode=single_agent`,
`add_noise=False`, `temperature=0`, seeds `{42, 1051, 2060}` (identical
trajectories). *(Re-recorded 2026-10-08 on `f2c65c4` with the greedy control, and again on
`4574475` after the config-hash fix, rows unchanged; see the correction below.)*

| Metric | Value | Gated? |
|---|---|---|
| `l2_error_ratio_at_matched_dof` | **0.9532** (matched DOF 287) | yes (`< 1`) — **Win** · **CORRECTED (2026-10-08):** the verdict stands, but the win is single-element greedy marking, not look-ahead — **search contributed no decisions** |
| `l2_error_ratio_at_matched_solves` | 9.23 | no |
| `error_per_dof_ratio_mcts_over_dorfler` | ~30.8 (32.10 and 32.65 in the two 2026-10-08 re-records; wall-clock, moves with load) | no |

Adequacy abort did not fire (adaptive `N^-1.31` vs uniform `N^-0.67` on
`(200, 4000)`). Those rates are **gate evidence**, not a look-ahead win. Do not
present the 0.9532 ratio as if it were fitted over that adequacy window.

### Correction (2026-10-08): the Win is greedy marking

Re-recorded from a clean tree with `--proposal-grade` and the greedy control on
(sidecar `dirty: false`, SHA `f2c65c4`; `docs/business/COMMERCIALIZATION_PEER_REVIEW.md`
§1, Gate 0.1). Every `uniform` / `dorfler` / `mcts` row is identical to the
`19609d4` record in every column except `wall_time_seconds`; the matched-DOF and
matched-solves ratios are bit-identical. Re-recorded once more the same day on `4574475`:
the earlier `config_hash` folded the nested `SubstrateConfig.created_at` timestamp into the
hash, so no re-run could reproduce it. Every row again matched except `wall_time_seconds`,
and the sidecar's `config_hash` (`d6993a0dfd1ab778`) now recomputes from its recorded config.

| Metric | Value | Gated? |
|---|---|---|
| `l2_error_ratio_greedy_over_dorfler_at_matched_dof` | **0.9532** | no |
| `l2_error_ratio_mcts_over_greedy_at_matched_dof` | **1.0** | no |
| `decisions_diverging_from_greedy` / `_max` | **0 / 0** | no |

The MCTS arm makes the greedy control's identical 12 decisions on every seed, so
the 0.9532 is single-element greedy marking against Dörfler bulk marking — a
marking-granularity effect — and **search contributed no decisions**. The
hypothesis (look-ahead beats greedy marking) is **untested** by this artifact;
Gate 1 tests it. Two further readings are corrected:

- **The binding limit is `max_steps=12`, not `max_dof=600`.** Both game-driven arms
  stop after 12 refinements at 287 DOF, which is what sets matched DOF; the
  600-DOF policy budget never binds.
- **9.23 at matched solves is not the price of a quality edge.** The greedy control
  reaches the same 287-DOF mesh in 13 solves; the MCTS arm spends 43.

## Regression Surface

```bash
pytest tests/pde/games/test_substrate_refinement_game.py tests/research/substrates/test_residual_evaluator.py tests/research/test_mcts_classical_amr_arena.py tests/poc/test_mcts_classical_amr_arena_config.py tests/poc/test_mcts_classical_amr_arena_scenario.py tests/scripts/test_run_mcts_classical_amr_arena.py -v -m "not gpu_required"
```

## Out of Scope

- Trained evaluator / FNet checkpoint
- PETSc / MFEM
- Frozen tracks (`src/video_compression/`, `dashboard/`, `hf_space/`)
- Wiring scikit-fem into `lshape_amr_compare` (superseded; golden stays)
- Octree-on-SDF, certificates, Noyron v2.3 / v3.1
- Quoting adequacy rates as MCTS / look-ahead progress
- Changing `MCTS` default `search_mode` off `ZERO_SUM`
