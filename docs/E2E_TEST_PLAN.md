# E2E Test Plan — covering the work landed since the 2026-07 core refocus

> **Status:** plan, not delivery. Nothing in this document is implemented yet; every test
> named below is a proposal with an owner-visible acceptance bar. Measured numbers are marked
> *(measured)*; everything else is an estimate and says so.
>
> **Scope authority:** `docs/FOCUS.md` and `config/focus.yaml`. This plan adds tests only on
> the active surfaces (`src/refinement/`, `src/pde/`, `src/mcts/`, `src/research/`, the
> governance layer, and the PoC/agents CLIs that drive them). The two frozen tracks (`codec`,
> `interactive-surfaces`) are explicitly out of scope — an E2E suite for `dashboard/` or
> `video_compression/` landing alongside refinement work is exactly the split-attention diff
> `scripts/check_focus.py` exists to reject.

## 0. The finding that reorders everything else

**`tests/e2e/` is not run by CI, and has not been for as long as the fast lane has existed.**
Adding E2E tests to it without first changing that would add tests that pass on the author's
machine and are never executed anywhere else.

The evidence, all in the tree today:

| Fact | Where |
| --- | --- |
| The fast lane passes `--ignore=tests/e2e/` **and** `-m "not slow and not e2e and not gpu_required"` — belt and braces, both excluding the directory | `.github/workflows/ci.yml`, `test-fast` job, "Run fast unit tests" step |
| Exactly **one** of the 13 files in `tests/e2e/` is named by any workflow step: `test_chess_training_e2e.py`, in the `test-chess` job | `ci.yml`, "Run chess E2E smoke tests" step |
| The workflow *says why* it is not wired: `test_train_physics_minimal` "shells out with a fixed 120 s timeout and fails on a loaded machine" | `ci.yml`, comment above "Run demo and notebook suites" |
| **That reason is stale.** The test now passes `--n-train-samples 16 --n-eval-samples 8` (the run was never minimal — `--train-size` is the grid side, not the sample count — so it built 5000 samples and needed ~1.7 h; no timeout was ever the problem) and the budget is `E2E_TRAINING_TIMEOUT_S`, scaled by `E2E_TIMEOUT_SCALE` | `tests/e2e/test_quick_validation_journey.py:68-78`, `tests/e2e/conftest.py:19-42` |
| `make test-e2e` runs only `tests/e2e/test_user_journey_*.py` — **3 of ~70** tests. `make pre-pr` therefore certifies a PR against three E2E tests | `Makefile`, `test-e2e` target |
| Two files carry **no `e2e` marker** at all (`test_chess_training_e2e.py`, `test_cli_journey.py`), so a `-m e2e` selection cannot find them; only the `--ignore` keeps them out of the fast lane | `grep -L "pytest.mark.e2e" tests/e2e/test_*.py` |
| The existing centaur E2E asserts `returncode in (0, 1, 2)` on `poc.cli info` — every exit code argparse can produce. That is not an assertion | `tests/e2e/test_centaur_e2e.py:148` |

This is the same defect class the repo has already recorded three times for other directories
(`tests/demos/` and `tests/notebooks/` — 226 tests never executed in CI until `18f533d`;
`src/backend` — 213 passing tests measured nowhere). Each was found by a person reading the
workflow. Phase 0 below fixes the wiring **and adds the hermetic guard that would have caught
it**, so the fourth instance cannot recur silently.

The measured cost of the suite today, which is what decides whether wiring it is safe:

| Selection | Result *(measured 2026-09-04, this container, loadavg ≈1)* |
| --- | --- |
| `pytest tests/e2e/ -m "not gpu_required" -q` | see §7 — filled from the run this plan was written against |

## 1. What "relevant work" means here, and what already covers it

The work landed since the 2026-07-22 refocus, with the test tier that reaches each item
today. "In-process `main()`" means a test imports the script and calls `main(argv)` — it
exercises the logic but not the entry point (`python -m …`), import-time side effects, the
`if __name__ == "__main__":` line, or the exit code as a shell sees it.

| Work item (date) | User-facing entry point | Unit / integration today | E2E today | Gap |
| --- | --- | --- | --- | --- |
| Element-local refinement substrate, Slices A–D (2026-09) | `scripts/run_adaptive_vs_uniform.py` → `results/lshape_adaptive_vs_uniform.{csv,run.json}`; `src/research/substrates/sweep.py::run_refinement_sweep`; substrate registry | `tests/scripts/test_run_adaptive_vs_uniform.py` (11, in-process `main()`), `tests/research/test_substrates_sweep.py`, `test_tensor_grid_substrate.py`, `test_skfem_substrate.py`, `test_amr_arena_interpretability.py` (AC7 gate) | none | No test runs the script as a process, checks the `.run.json` sidecar it writes against the charter's evidence row, or drives the skfem substrate from the *registry* (the user's lookup path) rather than by direct construction |
| L-shape AMR MCTS-vs-Dörfler harness + retraction (2026-07/08) | `scripts/run_lshape_amr.py`; `poc.cli run --config config/scenarios/lshape_amr_compare_cpu.yaml` | `tests/scripts/test_run_lshape_amr.py::test_end_to_end_small_run` (in-process), `tests/poc/test_lshape_amr_compare_scenario.py` | none | Shipped CPU YAML is never run through the CLI process; CSV/PNG artifacts never checked from outside the process |
| Honest transfer benchmark (2026-07-22) | `scripts/run_transfer_baseline_compare.py` `--record-baseline` / `--baseline`; CI job `transfer-baseline-regression` (soft) | `tests/scripts/test_run_transfer_baseline_compare.py::test_record_and_diff_roundtrip` (in-process) | CI job runs the real script, **soft-gated**, no assertion on its output beyond exit code | The one E2E that exists is a CI step, not a test — it cannot be run locally by `pre-pr` and its exit code is informational |
| Stochastic Galerkin NKE layer (2026-07-23) | `scripts/run_stochastic_galerkin_compare.py`; `config/scenarios/stochastic_galerkin_compare_ci.yaml` | `tests/scripts/test_run_stochastic_galerkin_compare.py::test_micro_run_exits_zero`, `::test_record_baseline_then_self_diff` (in-process) | none | Same shape as transfer |
| Noyron v2.2 basis selection (2026-07-01) | `poc.cli run --config config/scenarios/noyron_basis_cpu.yaml` | `tests/poc/test_noyron_basis_scenario.py` (real CPU micro-run, in-process) | none | Shipped CPU YAML never dispatched through the CLI process |
| PoC baseline harness: `record-baseline` / `diff` (2026-06-12), `eval-harness` (2026-08) | `python -m src.poc.cli {run,record-baseline,diff,eval-harness}` | `tests/poc/test_cli_baselines.py` (13), `test_cli_eval_harness.py` (5) — all call `cmd_*` in-process | `test_poc_scenario_journey.py` covers `list/info/run --help` only | The `run → record-baseline → diff` chain, exit 0 on self-diff and exit 1 on regression, has never been observed as three processes sharing an output directory — which is how CI and a human use it |
| Agents scaffold CLI (2026-07-01) | `python -m src.agents.cli scaffold <name> [--dry-run]` | `tests/agents/test_scaffold_cli.py` (10; Typer runner in-process) | `test_centaur_e2e.py` checks `--help` only | Overwrite refusal's exit code, and "the generated agent runs under `src.agents.cli`", unobserved as processes |
| Governance gates: `scripts/audit_abstractions.py` (B18), `scripts/check_focus.py`, `scripts/check_doc_links.py` | Run by CI's `lint` job and pre-commit | `tests/scripts/test_audit_abstractions.py` (18), `test_check_focus.py` (43, 3 via subprocess) | none | `audit_abstractions` is never run with **CI's own argv**; a drift between the workflow's four roots and the test's assumptions is invisible |
| Run-provenance sidecars (`src/research/run_manifest.py`) | Written by `run_adaptive_vs_uniform`; read by `tests/docs/test_charter_alignment.py::test_evidence_artifacts_carry_run_provenance` | unit tests on the module | charter guard checks the committed sidecar **exists** | Nothing regenerates an artifact and checks the sidecar it produces round-trips through `load_run_manifest` / `migrate_run_manifest` with the fields the charter guard reads |
| Trained-evaluator path + checkpoint safety (2026-08-21) | `scripts/train.py` → checkpoint → `scripts/inspect_checkpoint.py` → `AlphaGalerkinSolver(evaluator="trained")` | `tests/alphagalerkin/test_trained_evaluator.py` (round trip), `tests/security/test_checkpoint_safety.py`, `tests/scripts/test_cli_pickle_flags.py` | `test_user_journey_pde_solving.py` uses the default evaluator only | No journey produces a checkpoint with the shipped trainer and consumes it with the shipped inspector and solver |
| Dockerfile (2026-08-16, first built 2026-09-02) | `make docker-build` / `make docker-test` | `tests/docs/test_dockerfile_context.py` (hermetic) | none, deliberately | A real build is a CI job decision, not a test — see §6 |
| `scripts/evaluate_model.py`, `scripts/demo_pde_solver.py`, `scripts/export_helix_stl.py` | direct | **none at all** (no test file names them) | none | Untested entry points on active surfaces; `--help` smoke at minimum |

## 2. Design rules for the new tier

These are drawn from what the repo already enforces elsewhere, not invented here.

1. **Through the real entry point, as a process.** Use the `cli_runner` fixture
   (`tests/e2e/conftest.py`) and `python -m <module>`. The in-process `main()` tests in
   `tests/scripts/` stay where they are; E2E adds the process boundary, not a second copy of
   their assertions.
2. **Exact exit codes.** `assert result.returncode == 0` or `== 1`, never a set. The one
   existing `in (0, 1, 2)` is corrected in Phase 0.
3. **Every knob from argv or a shipped YAML; every output under `tmp_path`.** No test writes to
   `results/` or `outputs/`. Shipped configs are used *as shipped* wherever a CPU/CI variant
   exists (`*_cpu.yaml`, `*_ci.yaml`), so a config that drifts from its schema fails here.
4. **Timeouts only from the three tiers** (`E2E_TRIVIAL_TIMEOUT_S`, `E2E_BENCHMARK_TIMEOUT_S`,
   `E2E_TRAINING_TIMEOUT_S`). A new literal timeout is a review blocker.
5. **No wall-clock ratio assertions.** CLAUDE.md's Next Steps record seven such assertions
   already in the blocking lane and one failing under load; E2E adds zero.
6. **Every test cites what it guards** — a spec AC (`specs/*.spec.md`), a charter row
   (`openspec/specs/project-charter/spec.md`), or a CLAUDE.md Regression Surface row — in its
   docstring. A test that guards nothing nameable is not written.
7. **Mutation-checked before merge.** Each test's PR description names the planted defect it
   fails on (a dropped CSV column, an exit code forced to 0, a flag ignored). This is the
   repo's standing convention (`tests/docs/*`, the substrate D1–D5 work), and it is the only
   defence against the `returncode in (0, 1, 2)` class of non-test.
8. **Markers:** `e2e` on every file in `tests/e2e/` (guarded, §3); `slow` on anything above
   the benchmark tier; `fem_required` on anything importing `scikit-fem`; `gpu_required`
   never appears in this plan (GPU journeys stay manual, see §6).
9. **Determinism:** seeds pinned via argv/YAML; assertions are on structure, keys, ranges and
   monotone properties, never on a specific floating-point value unless the artifact is a
   committed baseline with its own `tolerance_pct`.
10. **Subprocess over in-process for anything touching a singleton registry.** `ScenarioRegistry`
    and the substrate registry are process-global and `tests/poc/*` autouse fixtures `clear()`
    them; the charter guard already reads them in a subprocess for that reason.

## 3. Phase 0 — make the directory visible, and make invisibility impossible

**Prerequisite for every later phase. Its own PR.** Estimated size: ~80 lines of workflow and
Makefile, one new test file, four one-line test edits.

| Change | Detail |
| --- | --- |
| New CI job `test-e2e` | `needs: test-fast`, `timeout-minutes: 30`, runs `pytest tests/e2e/ -m "not gpu_required and not fem_required" --tb=short -q`. Same shape as `test-integration`. |
| `fem_required` E2E in `test-extras` | That is the only job that installs `[fem]`; the `fem_required` E2E of Phase 1 runs there, after the existing two `[fem]` gate steps. |
| Blocking from day one | Add to `ci-success.needs` in the **same** PR. The workflow's stated reason for not wiring it — the 120 s timeout — is already fixed (§0). If the first CI run shows a genuine load failure, the fallback is the `focus`/`secrets` convention: a named comment in `ci-success` saying why it is advisory, not a silent omission. Do **not** widen a timeout to get green; scale `E2E_TIMEOUT_SCALE` in the job's `env:` and record the measured reason. |
| Retire the stale comment | Replace the "not currently safe to wire" paragraph in `ci.yml` with a pointer to the job and the commit that fixed `test_train_physics_minimal`. |
| `make test-e2e` | Run the whole directory with the same `-m` filter as CI, not `test_user_journey_*.py`. `pre-pr` then means what it says. |
| Markers | Add `pytestmark = pytest.mark.e2e` to `test_chess_training_e2e.py` and `test_cli_journey.py`. |
| Fix the non-assertion | `test_poc_cli_info_scaling_law`: `returncode == 0` and the stdout contains the scenario's description string, not just the substring "scaling". Mutation: rename the scenario in the registry → must fail. |
| **New guard: `tests/docs/test_e2e_visibility.py`** (hermetic, parses `ci.yml` and the test files, runs nothing) | (a) every `tests/e2e/test_*.py` carries the `e2e` marker at module or function level; (b) at least one non-`--ignore` step in `ci.yml` selects `tests/e2e/` **and** its `-m` expression does not exclude `e2e`; (c) that step's job is in `ci-success.needs` or carries an "advisory" comment naming a reason; (d) `Makefile`'s `test-e2e` target selects the directory, not a glob subset. Same idiom as `test_coverage_gate_integrity.py` — falsifiable exemptions only. Mutations to kill before merge: delete the job; add `not e2e` to its `-m`; remove the marker from one file; restore the old `test_user_journey_*.py` glob. |
| CLAUDE.md Regression Surface row | "E2E tier visibility" row pointing at the guard, per the repo's rule that a gate not mirrored there decays. |

**Acceptance for Phase 0:** one green CI run with `test-e2e` in the gate; the four listed
mutations each fail a *named* test; `make pre-pr` runs the full directory.

## 4. Phase 1 — the refinement thesis, end to end (active-surface priority)

These are the journeys the cycle's thesis lives on. They are first because `docs/FOCUS.md`
says attention goes here, and because the 2026-08-16 retraction shows what an unmeasured
substrate costs.

### 4.1 `tests/e2e/test_adaptive_vs_uniform_journey.py` — script → CSV → provenance sidecar

Guards: charter evidence row *"L-shape adaptive Dörfler vs uniform at matched DOF"*;
`specs/refinement_substrate.spec.md` AC5 (DOF convention recorded in the manifest), AC7
(rate separation reported with the range pinned in config).

| Test | Journey | Asserts | Tier |
| --- | --- | --- | --- |
| `test_script_writes_csv_and_sidecar` | `python -m scripts.run_adaptive_vs_uniform --output <tmp>/x.csv --initial-side 4 --max-dof 120 --marking-fraction 0.5` | exit 0; CSV exists with both arms (`uniform`, `dorfler`) and the columns the charter guard reads; `manifest_path_for(csv)` exists | BENCH |
| `test_sidecar_round_trips_and_echoes_argv` | same run, then `load_run_manifest` | `harness == "scripts.run_adaptive_vs_uniform"`; `config` equals the argv values (every knob, not a sample); `artifacts.csv` points at the CSV *relative to the manifest*; `migrate_run_manifest(raw)` is idempotent on the freshly written file | BENCH |
| `test_sidecar_metrics_have_the_charter_shape` | same run | keys `uniform_convergence_exponent`, `dorfler_convergence_exponent`, `dorfler_over_uniform_{min,max}` present and finite; `matched_dof_min <= matched_dof_max <= max_dof` | BENCH |
| `test_dof_budget_is_honoured_across_the_process_boundary` | run with `--max-dof 60` | no CSV row exceeds 60 DOF in either arm | BENCH |
| `test_unknown_flag_exits_two` | `--no-such-flag` | exit 2, usage on stderr | TRIVIAL |

Mutations: drop the sidecar write → test 1; write `config_hash` but not `config` → test 2;
ignore `--max-dof` → test 4.

### 4.2 `tests/e2e/test_lshape_amr_journey.py` — shipped CPU YAML through the CLI process

Guards: `specs/lshape_amr_compare.spec.md` (three ratios recorded, one gated); CLAUDE.md
row *"L-shape AMR MCTS-vs-Dörfler baseline"*.

| Test | Journey | Asserts | Tier |
| --- | --- | --- | --- |
| `test_script_runs_shipped_cpu_yaml` | `python -m scripts.run_lshape_amr --config config/scenarios/lshape_amr_compare_cpu.yaml --output-dir <tmp> --max-dof 120 --n-simulations 2 --seed 1` | exit 0 **or** 1 is *not* acceptable here — the config's own threshold decides; assert exit `== 0 if ratio < 1 else 1` by reading the ratio back from the persisted result JSON, so the test checks the exit-code *contract* rather than a research outcome | TRAINING |
| `test_all_three_ratios_are_persisted` | same run | `l2_error_ratio_at_matched_dof`, `l2_error_ratio_at_matched_solves`, `error_per_dof_ratio_mcts_over_dorfler` in the result JSON, all finite; per-seed spread keys present | TRAINING |
| `test_csv_and_png_artifacts_land_in_output_dir` | same run | both files under `<tmp>`, CSV has a `method` column with both arms | TRAINING |
| `test_poc_cli_dispatches_the_same_yaml` | `python -m src.poc.cli run --config config/scenarios/lshape_amr_compare_cpu.yaml --output-dir <tmp>` with the YAML copied to `tmp` and budgets reduced by editing the copy (documented as the only place a test edits a shipped config, and why: `poc.cli run` has no per-scenario override flags) | exit 0; a result JSON under `<tmp>/results/` whose `scenario_name == "lshape_amr_compare"` | TRAINING |

The two `TRAINING`-tier runs share one `module`-scoped fixture so the search runs once.
Mark `slow`. Mutation: make `export_plot` a no-op → test 3; drop one ratio from
`ScenarioResult.metrics` → test 2.

### 4.3 `tests/e2e/test_refinement_substrate_journey.py` — registry → sweep → adequacy verdict

Guards: `specs/refinement_substrate.spec.md` AC5, AC7, AC8 (the *tripwire* half: a uniform
rate that is too good must fail the gate); substrate D5 (registry has registrants, read in a
subprocess).

This is a Python-level journey run **in a subprocess** (`python -c` via `cli_runner`'s
`sys.executable`), because the substrate registry is a process-global singleton that two
suites `clear()`.

| Test | Journey | Asserts | Marker |
| --- | --- | --- | --- |
| `test_tensor_grid_from_registry_reproduces_the_committed_defect_direction` | registry lookup by `kind="tensor_grid"` → `run_refinement_sweep` (uniform + Dörfler, θ=0.5, small DOF range) → `measure_rate_separation` | Dörfler is *worse* than uniform (ratio > 1), matching the sign of the committed `.run.json`; **no** numeric closeness to the committed artifact (different budget) | none |
| `test_skfem_from_registry_passes_the_adequacy_gate_on_a_small_range` | registry lookup by `kind="skfem_tri"` → same sweep | `gate_violations()` empty on the reduced range **and** `describe()["dof_convention"]` is the FEM one; `n_dof_free <= n_dof` at every level | `fem_required` |
| `test_gate_is_not_vacuous` | feed the tensor-grid sweep result into the skfem gate predicate | non-empty violations — the AC7 clause "a gate that passes on both substrates is not a gate", but now driven from the registry path a user would take | none |

The in-process `test_amr_arena_interpretability.py` already proves the gate on directly
constructed substrates; this file proves the **registry** hands a user the same objects.
Mutation: register `TensorGridSubstrate` under both kinds → tests 2 and 3 fail.

Found while planning: **`gate_violations()` is defined in the test file**
(`tests/research/test_amr_arena_interpretability.py:125`), not in `src/`. A user who runs the
sweep has no way to ask for the adequacy verdict the spec calls "the gate that makes any
comparison meaningful"; only pytest can. Phase 1 promotes it to
`src/research/substrates/sweep.py` next to `measure_rate_separation` (the test file keeps a
one-line re-export so its existing `TestGatePredicate` suite is untouched), which is what
lets a subprocess call it at all. The predicate's thresholds already live in config; only the
function moves.

## 5. Phase 2 — the evidence pipeline: run → record-baseline → diff, as three processes

Guards: CLAUDE.md row *"PoC baseline harness (WS2)"*; `specs/headline_runs.spec.md`;
CI job `transfer-baseline-regression` (this is the local, hard-asserting twin of that soft
CI step).

### 5.1 `tests/e2e/test_baseline_gate_journey.py` — parametrised over the three harness scripts

Parametrise over `(module, shipped_yaml, tiny_overrides)`:

| Module | Shipped config | Overrides (all existing flags) |
| --- | --- | --- |
| `scripts.run_transfer_baseline_compare` | `config/scenarios/transfer_baseline_compare_ci.yaml` | `--n-epochs 1 --n-seeds 1 --n-train-samples 8 --target-resolution 13` |
| `scripts.run_stochastic_galerkin_compare` | `config/scenarios/stochastic_galerkin_compare_ci.yaml` | `--n-epochs 1 --n-seeds 1 --grid-n 8` |
| `scripts.run_lshape_amr` | `config/scenarios/lshape_amr_compare_cpu.yaml` | `--max-dof 120 --n-simulations 2` (no `--record-baseline` on this script today — see gap below) |

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_record_then_self_diff_exits_zero` | run with `--record-baseline <tmp>/b.json`, then run again with `--baseline <tmp>/b.json --tolerance-pct 50` | first exit 0, baseline JSON parses with `POC_BASELINE_*_SCHEMA_VERSION`; second exit 0 |
| `test_tightened_baseline_exits_one` | edit `b.json`: halve every `lower_is_better` entry's value and set `tolerance_pct` to 0 | exit 1; stdout names the regressed metric |
| `test_missing_baseline_file_exits_nonzero_with_message` | `--baseline <tmp>/absent.json` | exit ≠ 0, the path appears in stderr |

Found while planning, to be fixed in the same phase: **`scripts/run_lshape_amr.py` has no
`--record-baseline` / `--baseline` flags**, unlike its two siblings, so the L-shape headline
cannot be regression-gated from its own CLI. Adding them is ~40 lines mirroring
`run_transfer_baseline_compare.py:155-173`; the third parametrisation above is what makes
the omission a red test rather than a note.

### 5.2 `tests/e2e/test_poc_baseline_cli_journey.py` — the generic CLI, same chain

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_run_record_diff_chain` | `poc.cli run --config <tmp copy of config/scenarios/noyron_basis_cpu.yaml> --output-dir <tmp>` → parse the run id from `<tmp>/results/<run_id>/` → `poc.cli record-baseline --output-dir <tmp> --run-id <id> --out <tmp>/b.json --tolerance-pct 10` → `poc.cli diff --baseline <tmp>/b.json --output-dir <tmp> --run-id <id>` | three exit 0s; `b.json` has one entry per numeric metric in the result JSON |
| `test_diff_against_a_tampered_baseline_exits_one` | as above, then tighten one entry | exit 1 |
| `test_record_with_unknown_run_id_exits_nonzero` | `--run-id nope` | exit ≠ 0, message names the id |

Mark `slow` (the `noyron_basis` CPU run is the cost; §7 records it).

## 6. Phase 3 — agents and governance CLIs as processes

### 6.1 `tests/e2e/test_agents_scaffold_journey.py`

Guards: CLAUDE.md row *"Agents hardening (lifecycle hooks + timeout + scaffold)"*.

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_dry_run_prints_plan_and_writes_nothing` | `python -m src.agents.cli scaffold demo_probe --root <tmp> --dry-run` | exit 0; three planned paths in stdout; `<tmp>` empty |
| `test_scaffold_then_second_run_refuses` | same without `--dry-run`, twice | first exit 0 + three files; second exit ≠ 0 and the existing path named |
| `test_generated_agent_is_importable_from_a_fresh_interpreter` | `python -c "import <generated module>"` with `PYTHONPATH=<tmp>` | exit 0 |
| `test_list_agents_sees_research` | `python -m src.agents.cli list-agents` | exit 0, `research` in stdout |

### 6.2 `tests/e2e/test_governance_cli_journey.py`

Guards: CLAUDE.md rows *"Abstraction audit (F0/F1 screen)"* and *"Scope containment (frozen
tracks)"*. The point is to run each gate with **the argv CI runs**, parsed out of `ci.yml`,
so the test cannot drift from the workflow.

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_audit_abstractions_with_ci_argv_is_clean` | read the `lint` job's `python -m scripts.audit_abstractions … --fail-on-missing` line from `ci.yml`; run it | exit 0; the four roots named in CLAUDE.md appear in the parsed argv (a fifth or a missing one fails, so CLAUDE.md's row and the workflow cannot diverge) |
| `test_audit_report_only_root_exits_zero` | `python -m scripts.audit_abstractions src/backend` (report-only, as CI) | exit 0 even though findings exist; findings printed |
| `test_check_focus_on_an_empty_diff_is_clean` | `python -m scripts.check_focus --base HEAD --head HEAD --fail-on-violation` | exit 0 |
| `test_check_doc_links_on_this_plan` | `python scripts/check_doc_links.py docs/E2E_TEST_PLAN.md` | exit 0 — the plan's own links resolve |

`test_check_focus.py` already runs three subprocess cases with synthetic repos; this adds
only the live-repo, CI-argv invocation.

### 6.3 `tests/e2e/test_untested_entry_points_smoke.py`

The three scripts with no test at all. `--help` smoke is the floor, not the ceiling; each gets
one real invocation where a CPU one exists.

| Script | Smoke | Real run |
| --- | --- | --- |
| `scripts/evaluate_model.py` | `--help` exit 0 | deferred: needs a checkpoint — chain it behind Phase 4's trainer fixture |
| `scripts/demo_pde_solver.py` | `--help` exit 0 | smallest argv the parser accepts, exit 0, output under `tmp` |
| `scripts/export_helix_stl.py` | `--help` exit 0 | `--n-turns 1` plus its output flag to `tmp`; file exists and starts with `solid` (ASCII STL) or has the 84-byte binary header |

## 7. Phase 4 — checkpoint lifecycle through the shipped tools

Guards: CLAUDE.md rows *"Trained evaluator"* and *"Checkpoint deserialization safety"*.

`tests/e2e/test_checkpoint_lifecycle_journey.py`, one `module`-scoped fixture that runs
`python -m scripts.train --config-name=train_fast training.total_steps=2 hydra.run.dir=<tmp>
…` once (exact override set to be confirmed against `config/train_fast.yaml` when writing;
the fixture fails loudly, not skips, if no checkpoint appears).

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_trainer_writes_a_versioned_checkpoint` | fixture | one `*.pt` under `tmp`; `load_torch_checkpoint` reads it with `weights_only=True`; version key present |
| `test_inspect_checkpoint_reads_it` | `python -m scripts.inspect_checkpoint <pt>` | exit 0; the checkpoint's top-level keys (what the script prints today) include the version key the migration registry reads |
| `test_inspect_refuses_a_marker_payload_without_the_hatch` | write the same marker pickle `tests/security/test_checkpoint_safety.py` uses | exit ≠ 0; the `allow_unsafe_pickle` hatch is **not** exercised here (that is the security suite's job) |
| `test_solver_consumes_it_as_trained_evaluator` | `AlphaGalerkinSolver(AlphaGalerkinConfig(evaluator="trained", checkpoint_path=<pt>, n_mcts_simulations=4))` on Poisson, in a subprocess | `SolverResult` with finite `l2_error`; a checkpoint whose action space does not match fails loudly under `checkpoint_strict_load=True` |
| `test_evaluate_model_runs_on_it` | `python -m scripts.evaluate_model --checkpoint <pt> …` smallest argv | exit 0 |

Mark `slow`. This is the only phase whose runtime is a bet on host speed; it lives behind
`E2E_TRAINING_TIMEOUT_S` and nothing else.

## 8. Explicitly not in this plan

| Item | Why |
| --- | --- |
| Frozen tracks (`video_compression`, `dashboard`, `hf_space`) | `docs/FOCUS.md`; substantive test additions there alongside this work would trip `check_focus`. When the freeze lifts, the same template applies. |
| GPU / LM Studio journeys (`llm_prior_*`, `scaling_law_demo`, `noyron_hx` voxel-FDM) | Already manual, `gpu_required`, documented in CLAUDE.md's Next Steps; CI has no CUDA. Nothing to add without hardware. |
| A real `docker build` | A CI job decision (`make docker-build` exists; the hermetic guard exists). A test that needs a daemon is a job, not a test. Recommend a `workflow_dispatch`-only job later; out of scope here. |
| Wall-clock or throughput assertions | §2 rule 5. |
| Re-testing what `tests/scripts/` already asserts in-process | E2E adds the process boundary and the shipped configs; it does not duplicate `apply_overrides` unit tests. |
| A `.run.json` for `results/lshape_mcts_vs_dorfler.csv` | Predates the manifest module; the charter guard already records this as the accepted gap. Regenerating it is a `run-provenance` task, not a test. |

## 9. Budget, placement and order

| Phase | New files | Est. tests | Est. runtime (CPU, cold) | CI job | Marker set |
| --- | --- | --- | --- | --- | --- |
| 0 | `tests/docs/test_e2e_visibility.py` + wiring | ~8 | < 1 s (hermetic) | `coverage-gates` or `lint` (hermetic, cheap) | — |
| 1 | 3 | ~12 | 4.1: ~30 s; 4.2: ~2–4 min; 4.3: ~20 s (+ ~30 s fem) | `test-e2e`; 4.3 fem half in `test-extras` | `e2e`, `slow` on 4.2, `fem_required` on 4.3b |
| 2 | 2 | ~9 | ~3–6 min (three harness scripts × 2 runs + one noyron run) | `test-e2e` | `e2e`, `slow` |
| 3 | 3 | ~13 | ~1 min (process startup dominates; torch import ≈ 3–5 s per process) | `test-e2e` | `e2e` |
| 4 | 1 | ~5 | ~2–4 min | `test-e2e` | `e2e`, `slow` |

Estimates are from the existing in-process tests' durations and the per-process torch import
cost; they are replaced by `--durations` output in each phase's PR. The `test-e2e` job's
30-minute cap has ~2× headroom over the sum of the upper estimates. If a phase pushes the job
past 15 minutes, split `slow` E2E into `test-slow` (which already has the
`workflow_dispatch` hatch) rather than widening the cap.

Order is 0 → 1 → 2 → 3 → 4, one PR per phase. Phase 1 before 2 because the thesis surfaces
are the cycle's focus; 3 and 4 are independent of each other and can be parallel PRs after
Phase 1 lands.

## 10. Definition of done, per test

A test in this plan is done when all of the following are true, and the PR says so:

- [ ] Runs via `python -m …` (or `python -c` for registry journeys) through `cli_runner`; never `patch.object(sys, "argv")`.
- [ ] Asserts an exact exit code.
- [ ] Every timeout is one of the three tier constants.
- [ ] Every output path is under `tmp_path`; `git status` is clean after the run.
- [ ] Docstring names the spec AC / charter row / Regression Surface row it guards.
- [ ] PR description names the planted mutation and the test that failed on it.
- [ ] `e2e` marker present; `slow`/`fem_required` where §9 says.
- [ ] Appears in CLAUDE.md's Regression Surface (one row per phase, not per test).
- [ ] `ruff check`, `ruff format --check`, and `mypy --strict` clean on the file (E2E files are under `tests/`, which CI lints).

## 11. Measured baseline this plan was written against

Filled from the run executed while drafting; see the PR that lands this document.

<!-- e2e-baseline:start -->
_pending — the full `tests/e2e/` run was in progress when this section was written; the
numbers are appended in the same commit once it completes._
<!-- e2e-baseline:end -->
