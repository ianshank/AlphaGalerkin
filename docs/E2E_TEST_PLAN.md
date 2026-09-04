# E2E Test Plan — covering the work landed since the 2026-07 core refocus

> **Status: IMPLEMENTED (2026-09-04).** Phases 0–4 are landed; see §12 for what shipped, what
> the implementation found wrong in this plan (again), and what is deliberately still open.
> The plan text below is kept as written so the corrections stay auditable.
>
> Numbers marked *(measured)* were produced on this branch on 2026-09-04; everything else was an
> estimate and says so. **v1 of this plan was peer-reviewed against the code and retracted in
> part** — §0.1 lists what was wrong, in the repo's usual form: the wrong claim stays on the
> page, marked, next to the corrected one. §12.2 does the same for v2.
>
> **Scope authority:** `docs/FOCUS.md` / `config/focus.yaml`. Tests are added only on the
> active surfaces (`src/refinement/`, `src/pde/`, `src/mcts/`, `src/research/`, the governance
> layer, and the PoC/agents CLIs that drive them). The two frozen tracks (`codec`,
> `interactive-surfaces`) are out of scope: an E2E suite for `dashboard/` or
> `video_compression/` landing alongside refinement work is the split-attention diff
> `scripts/check_focus.py` exists to reject.

## 0. The finding that reorders everything else

**`tests/e2e/` is not run by CI.** Adding E2E tests to it without first changing that adds
tests that pass on the author's machine and run nowhere else.

| Fact | Where |
| --- | --- |
| The fast lane passes `--ignore=tests/e2e/` **and** `-m "not slow and not e2e and not gpu_required"`; the `coverage` job repeats both | `.github/workflows/ci.yml` `test-fast` "Run fast unit tests"; `coverage` job |
| Exactly **one** of the 11 test files in `tests/e2e/` is named by any step in any workflow: `test_chess_training_e2e.py`, in `test-chess` | `ci.yml`, "Run chess E2E smoke tests" (grep over all of `.github/workflows/*.yml`) |
| The workflow *says why*: `test_train_physics_minimal` "shells out with a fixed 120 s timeout and fails on a loaded machine" | `ci.yml`, comment above "Run demo and notebook suites" |
| **That reason is half stale, and the repo contradicts itself about it.** The test now passes `--n-train-samples 16 --n-eval-samples 8` — the run was never minimal, `--train-size` is the grid side, so it built 5000 samples and needed ~1.7 h; the test's own comment says "no timeout was ever the problem". But `tests/e2e/conftest.py:21-27` *still* attributes the failure to the 120 s budget, and the default budget is still 120 s. Measured now: **15.3 s** *(measured)*. | `tests/e2e/test_quick_validation_journey.py:68-78`, `tests/e2e/conftest.py:19-42` |
| `make test-e2e` runs only `tests/e2e/test_user_journey_*.py` — **3 of 81** collected tests. `make pre-pr` certifies a PR against three E2E tests. (~~v1 said "3 of ~70"~~ — 81 collect *(measured)*.) | `Makefile` `test-e2e`, chained by `pre-pr` |
| Two files carry **no `e2e` marker** (`test_chess_training_e2e.py`, `test_cli_journey.py`; 5 tests), so `-m e2e` cannot select them; only the `--ignore` keeps them out | `grep -L "pytest.mark.e2e" tests/e2e/test_*.py` |
| **Seven** set-valued exit-code assertions exist in the directory (~~v1 said one~~): `test_quick_validation_journey.py:81` `[0,1]`; `test_centaur_e2e.py:148` `(0,1,2)` and `:160` `(0,2)`; `test_poc_scenario_journey.py:49` `[0,1,2]`, `:78` `[0,2]`, `:95` `[0,1]`, `:112` `[0,2]`. `(0,1,2)` is every code argparse can produce; that is not an assertion | as cited |
| Every existing `tests/e2e` test **passes on CPU today**: 81/81 *(measured)*. The full directory took **308 s** in one run and the three disjoint subsets summed to **≈130 s** when run separately — the first run overlapped a stray process, so treat 130–310 s as the range | §11 |

This is the same defect class the repo has recorded three times (`tests/demos/` + `tests/notebooks/`,
226 tests never executed until `18f533d`; `src/backend`, 213 passing tests measured nowhere).
Each was found by a person reading the workflow. Phase 0 fixes the wiring **and adds the
hermetic guard that would have caught it**.

### 0.1 What peer review found wrong in v1 of this plan

Recorded so the corrections are auditable, and because several are findings about the code,
not the plan.

| v1 said | Actually | Consequence for v2 |
| --- | --- | --- |
| §4.1 "no CSV row exceeds `--max-dof`" | `run_uniform_arm` appends the row *then* checks `n_dof >= max_dof` (`scripts/run_adaptive_vs_uniform.py:97-99`): the budget is a **stopping rule**, the last row always overshoots (`--max-dof 120` → 208 DOF *(measured)*) | Assert the documented semantics (§4.1); the v1 tests would have failed unmutated |
| §4.1 sidecar `artifacts.csv` "relative to the manifest" | Written verbatim from argv (`:238`); absolute when `--output` is absolute | Assert it equals argv |
| §1 `tests/scripts/test_run_adaptive_vs_uniform.py` "calls `main()` in-process" | It never imports `main`; only `build_parser`/`run_uniform_arm`/`compare`/`export_csv`. **The entry point and the sidecar write (`:217-251`) are exercised by nothing** | Gap is larger than stated; §4.1 is now the only cover |
| §7 `inspect_checkpoint` "exits ≠ 0 on a marker payload" | `inspect()` catches every exception, prints `Error:` and returns; `main()` returns `None` → **exit 0 on every failure** (`scripts/inspect_checkpoint.py:33-37,69-72`) | Script fix in Phase 4; test asserts the fixed contract |
| §5.1 `--n-train-samples 8` | Schema floor is `ge=64` (`transfer_baseline_compare_config.py:63`) → `ValidationError` before any run | 64 |
| §6.1 `list-agents` "shows `research`" | It lists `coupling`/`decomposition`/`meta`/`solver` *(measured)*; `research` is a *subcommand*, not a registered agent | Test asserts the four; the asymmetry is recorded, not papered over |
| §4.2 "read the ratio from the persisted result JSON" | `scripts/run_lshape_amr.py` writes CSV+PNG and **prints** metrics; only `poc.cli run` persists JSON | Script test parses the printed `Metrics:` block; JSON assertions move to the CLI path |
| §7 `evaluate_model --checkpoint` | Flag is `--model` (required); it is a **Go** evaluator (`--n-games`, plays vs random), not a PDE-solver consumer; `--device` defaults to CUDA-if-available at argv time | Moves to Phase 4 with the Go checkpoint the trainer writes; `--device` always passed |
| §1 `test_check_focus.py` "43 tests, 3 via subprocess" | 45 (parametrised); **zero** run the script as a process — `subprocess` there is for `git` helpers | v1's "already has three subprocess cases" was wrong; still not added here (§6.2 explains) |
| §4.3 "the predicate's thresholds already live in config" | `UNIFORM_RATE_BAND`, `ADAPTIVE_RATE_MIN`, `ADAPTIVE_VS_UNIFORM_MAX_RATIO`, `RATE_FIT_DOF_RANGE` are module constants **in the test file** (`test_amr_arena_interpretability.py:51-76`) | Promotion moves the constants into a typed config too (§4.3) |
| §1 `test_poc_scenario_journey.py` "covers `list`/`info`/`run --help` only" | It also runs `poc.cli run --tier unit` as a real process (`:83-95`) | Row corrected |
| Counts: `test_cli_baselines.py` 13, `test_scaffold_cli.py` 10, "13 files" | 12, 16, 11 test files | Corrected |
| §9 runtime estimates (2–4 min, `slow`) | `run_lshape_amr` at the plan's argv: **6 s**; `poc.cli run noyron_basis_cpu.yaml`: **8 s**; `run_adaptive_vs_uniform`: **< 5 s** *(measured)* — off by 20–50× | No `slow` marker on Phases 1–3; it would have deselected them from the job Phase 0 creates |
| §2 rule 9 "determinism: seeds pinned" | The existing transfer round-trip uses `--tolerance-pct 100000` because of "CPU-matmul run-to-run drift" (`tests/scripts/test_run_transfer_baseline_compare.py:134-138`); `src/seeding.py:52-54` claims GPU determinism but sets no cuDNN flag (`:64-65`) | §2 rule 9 rewritten; §5 uses an exact self-diff, not a re-run |
| §6.2 `check_focus --base HEAD --head HEAD`, `check_doc_links` on this file | Empty diff by construction; self-referential | Deleted |
| §6.1 import the generated agent with `PYTHONPATH=<tmp>` | It does `from src.agents.base import …` at `<root>/src/agents/<name>.py` with no `__init__.py`; the existing test uses `exec()` for this reason | Deleted |
| §4.3 `cli_runner` runs `python -c` | It always builds `[sys.executable, "-m", module]` | `py_runner` fixture added (§2) |
| §7 `hydra.run.dir=<tmp>` relocates checkpoints | `checkpoint_dir` resolves against cwd → writes `checkpoints/…` into the tree | `checkpoint_dir=<tmp>` override |
| §2 rule 10 "`tests/poc/*` fixtures clear the substrate registry" | `tests/poc` clears `ScenarioRegistry`; the substrate registry (`src/refinement/substrate_registry.py`, not under `src/research/substrates/`) is cleared by `tests/refinement/test_substrate.py` and `tests/research/test_skfem_substrate.py` | Corrected |

## 1. Device contract — the property "GPU/CPU agnostic" actually requires

"Agnostic" is three properties, and the code today satisfies none of them for an E2E test:

1. **The test picks the device from the environment, never from a literal.** Today every GPU
   smoke hardcodes `device="cuda"` and relies on the `gpu_required` skip
   (`tests/pde/stochastic/test_gpu_smoke.py:47`, `tests/poc/test_scaling_law_smoke.py:29,51`,
   `tests/agents/test_research_loop_smoke.py:33,60`); every workflow is `runs-on: ubuntu-latest`,
   so all 22 `gpu_required` sites have skipped on every CI run ever, and the root hook does not
   even report the count (`conftest.py:44-48`; the fem hook next to it does). No env-var
   convention for a test device exists anywhere in `tests/` or `src/`.
2. **The artifact can prove where the run executed.** It cannot: `ScenarioResult.device`
   ("Computation device used", `src/poc/config.py:333`) is populated at
   `src/poc/registry.py:327` and `:414` as `"cuda" if torch.cuda.is_available() else "cpu"` —
   **host availability, not execution device**. A `device: cpu` YAML on a CUDA host persists
   `"device": "cuda"`. This is the exact inversion of the property, in the one field an E2E
   would naturally read.
3. **No silent fallback between "asked for" and "ran on".** The repo has five device policies:
   `src/poc/device.py` (`cuda` raises, `auto` silent, `cpu` forces — used by all four scenarios
   in this plan); `src/alphagalerkin/solver.py:114-121` (**`cuda` silently falls back**, warn-once,
   and `AlphaGalerkinConfig.device` defaults to `"cuda"` at `:186`, so
   `test_user_journey_pde_solving.py` already runs the fallback path on every CI host);
   `src/training/base_trainer.py:228-231` (`auto` silent, anything else fails at the first
   `.to()`); `src/backend/torch_backend.py:94-101` (enum, `AUTO` silent);
   `scripts/evaluate_model.py:26` (argv-time availability default).

The mechanism, consistent with `tests/e2e/conftest.py`'s existing one-env-var convention:

| Piece | Specification |
| --- | --- |
| **`E2E_DEVICE`** | Env var read in `tests/e2e/conftest.py` beside `E2E_TIMEOUT_SCALE`. Values are exactly `resolve_device`'s four forms: `cuda`, `cuda:N`, `cpu`, `auto`. Default **`auto`**. No fifth vocabulary. |
| **Resolve once, at conftest import** | `E2E_RESOLVED_DEVICE = str(resolve_device(E2E_DEVICE, context="tests/e2e"))`. So `E2E_DEVICE=cuda` on a CPU box is a **collection error for the whole directory**, not a skip — the same outcome class as `ALPHAGALERKIN_REQUIRE_EXTRAS=1`. `cuda` *is* the require form (`device.py:10-12`); no separate `E2E_REQUIRE_GPU`. `auto` on CPU CI runs unchanged; `auto` on a CUDA host resolves to `cuda`; `cpu` on a CUDA host forces CPU for bisecting. |
| **Never forward `auto`** | Tests pass the *resolved concrete string* to every child. The silent-fallback resolvers are then never handed an ambiguous input; the only way a run lands on the wrong device is a script ignoring its flag, which the artifact assertion below catches. |
| **`e2e_device` fixture** | Session-scoped, returns the resolved string. A `pytest_terminal_summary` line (`e2e device: cuda` / `e2e device: cpu (E2E_DEVICE=auto, CUDA unavailable)`) — the fem-hook idiom — so a GPU host that lost its driver is visible in the summary. |
| **Flow (a): scripts with `--device`** | `run_lshape_amr.py:116`, `run_transfer_baseline_compare.py:153`, `run_stochastic_galerkin_compare.py:127`, `evaluate_model.py:68`: append `["--device", e2e_device]`. Overrides apply only when non-`None`, so the flag is authoritative. |
| **Flow (b): `poc.cli run`** (no `--device`) | The plan already copies YAMLs to `tmp` to shrink budgets. One shared helper `pin_scenario_yaml(src, dst, *, device, **overrides)` in `tests/e2e/conftest.py` sets `device: <e2e_device>` in the copy and **asserts the key existed before the edit** (nesting differs across configs), so a config with no `device` field cannot make the pin a silent no-op. Do **not** add `--device` to `poc.cli run`: not every scenario config has the field; a flag that applies to some scenarios is a new silent path. |
| **Flow (c): registry journeys** | Numpy-only (`src/research/substrates/*` and `src/research/lshape_amr_compare.py` import no torch). Pass nothing; the docstring says the surface is device-irrelevant. **Do not fabricate a device assertion there.** |
| **Flow (d): `scripts/train.py`** | Hydra override `device=<e2e_device>` (`train.py:125`; `config/train_fast.yaml:51` pins `cpu`). `inspect_checkpoint` reads at `map_location="cpu"` (`src/training/checkpoint.py:601,840`) — a real cross-device property on a CUDA host: trained on `cuda`, inspected on `cpu`. The solver step passes `device=e2e_device` explicitly and asserts `next(model.parameters()).device.type == e2e_device.split(":")[0]` inside the subprocess, as `tests/alphagalerkin/test_trained_evaluator.py:556-576` does — otherwise `solver.py:186` defaults to `cuda` and `:114-121` falls back silently. |
| **`py_runner` fixture** | `cli_runner` cannot run `python -c`. Add a sibling that builds `[sys.executable, "-c", code]` with the same env/timeout/cwd handling, for the registry journeys. |
| **Src fix: `ScenarioResult.device`** | `src/poc/registry.py:327,414` populate `device` from the scenario's resolved device when it has one (`getattr(self, "_device", None)`), falling back to the availability expression only for scenarios with no device concept. Then every E2E reading a result JSON asserts `payload["device"] == e2e_device`. **Mutation**: revert `:414` → fails on a CUDA host under `E2E_DEVICE=cpu`, and fails on *any* host via the negative test below. Additive; unit-tested in `tests/poc/`. |
| **The negative test that runs identically on both host types** | `cli_runner(..., env={"CUDA_VISIBLE_DEVICES": ""})` with `--device cuda` (or a pinned YAML) → exit 1, `status == "error"`, `error_message` contains `requested device='cuda' but CUDA is not available` (`device.py:69-73`, wrapped by `registry.py:310-327`). Proves the fail-loud contract without a GPU, and on a host that has one proves the child honoured the flag. Parametrised over the three `--device` scripts and one `poc.cli run` pinned YAML. The trainer path fails with torch's message instead (`trainer.py:205`), so that row asserts non-zero exit plus a CUDA substring, not the `resolve_device` text. Fixture-level proof first: a child with that env reports `torch.cuda.is_available() is False`. |
| **Sentinel** | One test, the **only** `gpu_required` name allowed in `tests/e2e/`: `e2e_device.startswith("cuda") == torch.cuda.is_available()` under `E2E_DEVICE=auto`. Gives a GPU host a red line if resolution and availability disagree, and `-rs` a legible count on CPU. The visibility guard (§3) allowlists exactly this name. |
| **Marker rule** | No other E2E file carries `gpu_required` — not because "GPU journeys stay manual" (v1's reason) but because the marker *skips*, and a device-agnostic tier must not be skippable on device. Root hook extended to report the gpu skip count like the fem hook. |
| **Honesty rule for numpy-only surfaces** | The L-shape scenario resolves a device (`lshape_amr_compare.py:62`) and then places nothing on it. Its E2E asserts that the device *is recorded* and says the harness is numpy-only. It is never described as "exercising the GPU". |
| **`src/seeding.py` docstring** | Claims "deterministic on CPU and GPU alike" and sets no cuDNN flag. Correct the docstring in Phase 0 (a docstring fix, not a behaviour change — enabling `cudnn.deterministic` repo-wide is a separate decision). No E2E pins a float on GPU on the strength of it. |

Which surfaces the device actually touches — so no test claims more than it can:

| Entry point | Torch on the hot path? | Device source | Agnostic via |
| --- | --- | --- | --- |
| `scripts/run_adaptive_vs_uniform.py`, `src/research/substrates/*`, `scripts/demo_pde_solver.py`, `scripts/export_helix_stl.py` | **No** (numpy/scipy) | none | nothing to do; say so |
| `scripts/run_lshape_amr.py` / `lshape_amr_compare` scenario | **No** (numpy solve); device resolved and logged only | `--device` / YAML | record-only assertion |
| `scripts/run_transfer_baseline_compare.py`, `scripts/run_stochastic_galerkin_compare.py`, `noyron_basis` via `poc.cli` | Yes | `--device` / YAML | flow (a)/(b) + `payload["device"]` |
| `scripts/train.py` → `inspect_checkpoint` → `AlphaGalerkinSolver(evaluator="trained")` → `evaluate_model` | Yes | Hydra `device=` / `map_location` / config `device` / `--device` | flow (d) |

## 2. Design rules for the new tier

Drawn from what the repo enforces elsewhere; each is a review blocker when violated.

1. **Through the real entry point, as a process** — `cli_runner` (`python -m …`) or
   `py_runner` (`python -c`). The in-process `main()` tests in `tests/scripts/` stay; E2E adds
   the process boundary, the shipped configs, and the exit code as a shell sees it.
2. **Exact exit codes**, and — because `1` means *both* "threshold failed" and "errored"
   (`src/poc/cli.py:221-223`, `run_lshape_amr.py:138`) — **`status` alongside the code**
   (`passed`/`failed`/`error`, `src/poc/config.py:309`). Never a set. `2` is argparse.
3. **Every knob from argv or a shipped YAML; every output under `tmp_path`**; `git status`
   clean after the run. Shipped `*_cpu.yaml`/`*_ci.yaml` are used as shipped except for the
   `pin_scenario_yaml` copy, which is the *only* place a test edits a config.
4. **Timeouts only from the three tiers.** A new literal is a blocker.
5. **No wall-clock or throughput assertions.**
6. **Every test names what it guards** (spec AC, charter row, Regression Surface row) in its
   docstring.
7. **Mutation-checked before merge**; the PR names the planted defect and the test that failed.
8. **Markers:** `e2e` on every file (guarded); `slow` only where §9 says (Phase 4);
   `fem_required` on the skfem half; `gpu_required` only on the sentinel.
9. **Determinism is not assumed.** Structure, keys, ranges, monotone properties, and
   `passed`-derived exit codes are asserted. A specific float is asserted only through a
   recorded baseline with its own `tolerance_pct`, or via an **exact self-diff** on the *same*
   run id. **No byte comparison to any committed `results/*.csv`** (`.8e` formatting is
   device-sensitive). A tighter rtol may be asserted **only** under `e2e_device == "cpu"`, with
   the reason in the docstring.
10. **Subprocess for anything touching a process-global registry** (`ScenarioRegistry`, the
    substrate registry).
11. **Nothing device-agnostic is skippable on device** (§1).

## 3. Phase 0 — make the directory visible, and make invisibility impossible

Prerequisite for every later phase. Its own PR. Also carries the two small src changes the
device contract needs (`ScenarioResult.device`, `seeding.py` docstring) because Phase 1's
assertions cannot be written without them.

| Change | Detail |
| --- | --- |
| New CI job `test-e2e` | `needs: test-fast`, `timeout-minutes: 30`, `pytest tests/e2e/ -m "not gpu_required and not fem_required" --tb=short -q`, `env: E2E_DEVICE=cpu` (explicit on CPU CI so a runner that grows a GPU cannot silently change what the job measures). |
| `fem_required` E2E in `test-extras` | A **named step** that positively selects `pytest tests/e2e/ -m fem_required` with `ALPHAGALERKIN_REQUIRE_EXTRAS=1` — otherwise the fem half runs nowhere, the exact invisibility §0 describes. |
| Blocking from day one | In `ci-success.needs` in the same PR. Measured cost is 130–310 s (§11), well inside the cap. If the first run shows a real load failure, scale `E2E_TIMEOUT_SCALE` in the job `env:` and record why; never widen a literal. |
| Retire the stale comments | Both: the `ci.yml` paragraph and `tests/e2e/conftest.py:21-27`. |
| `make test-e2e` | The whole directory with CI's `-m` filter. |
| Markers | `pytestmark = pytest.mark.e2e` on `test_chess_training_e2e.py` and `test_cli_journey.py`. |
| **Fix all seven set-valued exit assertions** | Each becomes an exact code plus, where a result exists, `status`. Mutation per site: force the CLI to exit 0 → the test fails. |
| Device pieces | `E2E_DEVICE`, resolve-at-import, `e2e_device`, `py_runner`, `pin_scenario_yaml`, terminal-summary line, gpu skip-count reporting, `ScenarioResult.device` fix + unit test, `seeding.py` docstring. |
| **Guard: `tests/docs/test_e2e_visibility.py`** (hermetic) | (a) every `tests/e2e/test_*.py` carries `e2e`; (b) a non-`--ignore` step in `ci.yml` selects `tests/e2e/` whose `-m` does not exclude `e2e`; (c) that job is in `ci-success.needs` — a comment is **not** accepted (a stale one passes forever); exemptions live in a test-side constant with a reason **and a self-expiry check**, the `_OMIT_WITHOUT_A_CI_GATE` idiom; (d) `Makefile` `test-e2e` selects the directory, not a glob subset; (e) no `gpu_required` in `tests/e2e/` except the allowlisted sentinel; (f) a positively-selecting `fem_required` step exists. Mutations to kill: delete the job; add `not e2e` to its `-m`; drop one marker; restore the glob; mark a journey `gpu_required`; delete the fem step. |
| **Guard: `-m` vocabulary** | `--strict-markers` rejects unknown markers on *tests*, not unknown identifiers inside a `-m` expression: `-m "not gpu_requried"` silently selects everything. Hermetic test parsing every `-m` in `ci.yml` and `Makefile`, asserting each identifier is in `pyproject.toml [markers]`. This is what makes Phase 0's own new `-m` string trustworthy. |
| CLAUDE.md Regression Surface rows | "E2E tier visibility" and "E2E device contract". |

**Acceptance:** one green run with `test-e2e` in the gate; every listed mutation fails a named
test; `make pre-pr` runs the full directory; `E2E_DEVICE=cuda pytest tests/e2e` on this CPU
box errors at collection with the `resolve_device` message.

## 4. Phase 1 — the refinement thesis, end to end

### 4.1 `tests/e2e/test_adaptive_vs_uniform_journey.py` — script → CSV → provenance sidecar

Guards: charter row *"L-shape adaptive Dörfler vs uniform at matched DOF"*;
`specs/refinement_substrate.spec.md` AC5, AC7 (rate separation reported); and the fact that
**nothing exercises this script's `main()` or its sidecar write today** (§0.1). Numpy-only;
no device.

| Test | Journey | Asserts | Tier |
| --- | --- | --- | --- |
| `test_script_writes_csv_and_sidecar` | `python -m scripts.run_adaptive_vs_uniform --output <tmp>/x.csv --initial-side 4 --max-dof 120 --marking-fraction 0.5` | exit 0; CSV has both arms in the column the charter guard reads (`method`/`arm`); `manifest_path_for(csv)` exists | BENCH |
| `test_budget_is_a_stopping_rule` | same run | per arm: every row but the last has `n_dof < max_dof`, the last has `n_dof >= max_dof`; `matched_dof_max` equals the smaller of the two arms' final DOF. This is the **documented** semantics (`:97-99`); the plan also asks that the script docstring say "stopping rule, not a cap" | BENCH |
| `test_sidecar_round_trips_and_echoes_argv` | `load_run_manifest` | `harness == "scripts.run_adaptive_vs_uniform"`; `config` equals every argv value **compared as parsed floats**, not strings; `artifacts["csv"] == <argv --output>` verbatim; `migrate_run_manifest(raw)` idempotent | BENCH |
| `test_sidecar_metrics_have_the_charter_shape` | same | `uniform_convergence_exponent`, `dorfler_convergence_exponent`, `dorfler_over_uniform_{min,max}` finite; `matched_dof_min <= matched_dof_max`. **No sign assertion** — at this budget the ratio's direction is a research outcome; the pinned-range assertion lives in `test_amr_arena_interpretability.py` | BENCH |
| `test_hardware_tag_is_recorded_or_declared_unknown` | same | `hardware_tag` is a non-empty string. Today it is `UNKNOWN` (`run_manifest.py:130`, volatile at `:52`) and the harness never sets it; the plan asks the harness to populate it with the host/device string so the charter's provenance record carries the one field a device-agnostic tier needs. If the owner decides provenance deliberately excludes device, the test asserts `UNKNOWN` and the spec says why | BENCH |
| `test_unknown_flag_exits_two` | `--no-such-flag` | exit 2 | TRIVIAL |

Mutations: drop the sidecar write → test 1; write `config_hash` but not `config` → test 3;
break the stopping check → test 2.

### 4.2 `tests/e2e/test_lshape_amr_journey.py` — shipped CPU YAML through both entry points

Guards: `specs/lshape_amr_compare.spec.md`; CLAUDE.md row *"L-shape AMR MCTS-vs-Dörfler
baseline"*. Measured cost at the argv below: **6 s** per run *(measured)*; two runs, no `slow`.

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_script_exit_code_tracks_the_verdict` | `python -m scripts.run_lshape_amr --config config/scenarios/lshape_amr_compare_cpu.yaml --output-dir <tmp> --max-dof 120 --n-simulations 2 --seed 1 --device <e2e_device>` | parse the printed `summary()` line for the verdict and the `Metrics:` block for the three ratios; `returncode == (0 if passed else 1)`. **Never** `== 0`: the gate is `ratio < 1.0`, which the honest headline (median 1.0996) fails and which reduced budgets happen to pass (0.9627 *(measured)*) — asserting 0 would encode a research outcome |
| `test_all_three_ratios_are_printed_and_finite` | same | `l2_error_ratio_at_matched_dof`, `l2_error_ratio_at_matched_solves`, `error_per_dof_ratio_mcts_over_dorfler` present, finite. Per-seed spread keys present but **not** asserted non-zero — at this budget all seeds are identical (`l2_ratio_seed_std = 0` *(measured)*) |
| `test_csv_and_png_land_in_output_dir` | same | both under `<tmp>`; CSV `method` column has both arms |
| `test_poc_cli_persists_json_with_verdict_and_device` | `pin_scenario_yaml(lshape_amr_compare_cpu.yaml, device=e2e_device, max_dof=120, n_simulations=2)` → `python -m src.poc.cli run --config <copy> --output-dir <tmp>` | `returncode == (0 if payload["passed"] else 1)`; `ScenarioResult.model_validate(payload)` succeeds in a fresh process; `payload["device"] == e2e_device` (record-only — the docstring says the solve is numpy) |
| `test_cuda_request_fails_loud_without_cuda` | script with `--device cuda` and `env={"CUDA_VISIBLE_DEVICES": ""}` | exit 1, the `resolve_device` message on stdout/stderr |

stdout contains structlog `debug` lines even at `--log-level WARNING` *(measured)*; parsers
anchor on the `Metrics:` header, not on line position.

### 4.3 `tests/e2e/test_refinement_substrate_journey.py` — registry → sweep → adequacy verdict

Guards: `specs/refinement_substrate.spec.md` AC5, AC7, AC8 (the tripwire half); substrate D5
(registry has registrants, read in a subprocess). Numpy-only; run through `py_runner`.

**Prerequisite src change, found by review:** `gate_violations()` **and its four thresholds**
live in `tests/research/test_amr_arena_interpretability.py:51-125`. A user who runs the sweep
cannot ask for the verdict the spec calls "the gate that makes any comparison meaningful".
Phase 1 promotes both to `src/research/substrates/sweep.py` as a typed `AdequacyGateConfig`
(`uniform_rate_band`, `adaptive_rate_min`, `adaptive_vs_uniform_max_ratio`, `rate_fit_dof_range`
as `Field`s with the current values as defaults — the `surface-hardcoded-value` skill's
value-identity check applies) plus `gate_violations(separation, gate)`. The test file keeps a
re-export so `TestGatePredicate` is untouched.

| Test | Journey | Asserts | Marker |
| --- | --- | --- | --- |
| `test_tensor_grid_from_registry_produces_a_complete_sweep` | `RefinementSubstrateRegistry().get("tensor_grid")` → `run_refinement_sweep` (uniform + Dörfler, θ=0.5, `max_dof` small) → `measure_rate_separation(adaptive, uniform, dof_range)` | both arms present; finite exponents; `error_ratio_at_matched_dof` finite; `describe()["dof_convention"]` recorded. **No sign assertion** (see §0.1) | none |
| `test_skfem_from_registry_passes_the_pinned_gate` | `get("skfem_tri")` → the same sweep over the **pinned** `rate_fit_dof_range` from `AdequacyGateConfig` — not a "reduced range", which review flagged as likely to fail `adaptive_rate_min` | `gate_violations()` empty; `dof_convention == "fem_basis_dofs"`; `n_dof_free <= n_dof` at every level | `fem_required` |
| `test_gate_is_not_vacuous` | tensor-grid sweep result into the same predicate | non-empty — AC7's "a gate that passes on both substrates is not a gate", driven from the registry path | none |

Cost of the fem half is **unverifiable here** (scikit-fem absent); the in-process AC7 test
already runs that range, so it is bounded by that test's CI duration in `test-extras`.
Mutation: register `TensorGridSubstrate` under both kinds → tests 2 and 3 fail.

## 5. Phase 2 — the evidence pipeline as separate processes

Guards: CLAUDE.md row *"PoC baseline harness (WS2)"*; `specs/headline_runs.spec.md`; the
local, hard-asserting twin of the soft CI job `transfer-baseline-regression`.

### 5.1 `tests/e2e/test_baseline_gate_journey.py` — the three harness scripts

Parametrise over `(module, shipped_yaml, overrides)`, always with `--device <e2e_device>`:

| Module | Shipped config | Overrides |
| --- | --- | --- |
| `scripts.run_transfer_baseline_compare` | `transfer_baseline_compare_ci.yaml` | `--n-epochs 1 --n-seeds 1 --n-train-samples 64 --target-resolution 13` (64 is the schema floor) |
| `scripts.run_stochastic_galerkin_compare` | `stochastic_galerkin_compare_ci.yaml` | `--n-epochs 1 --n-seeds 1 --grid-n 8` |
| `scripts.run_lshape_amr` | `lshape_amr_compare_cpu.yaml` | `--max-dof 120 --n-simulations 2` — **`xfail(strict=True)`** until the flags land (below) |

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_record_baseline_exits_zero_and_document_validates` | `--record-baseline <tmp>/b.json` | exit 0 (the verdict is ignored under record, `run_transfer_baseline_compare.py:214`); the JSON loads through `ScenarioBaselineRegistry.load` + `migrate_baseline_document`; only **stable** metrics are recorded (`_stable_metrics`), so the assertion is "every recorded entry is a metric of the run", not "one entry per metric" |
| `test_rerun_against_own_baseline_is_within_the_documented_drift` | second run with `--baseline <tmp>/b.json --tolerance-pct <T>` | exit 0. **`T` is not a free choice**: the in-process round-trip uses `100000` because of measured run-to-run drift (`tests/scripts/test_run_transfer_baseline_compare.py:134-138`). This test uses the tolerance the **shipped** `config/baselines/*_ci.json` declares per entry, so if that tolerance cannot absorb one-epoch drift on CPU, the test is red and the finding is that the CI regression gate is not a gate at that budget — which is worth knowing, and is why this test exists |
| `test_tightened_baseline_exits_one_and_names_the_metric` | edit **one** strictly-positive entry (`transfer_mse_ratio_13x13`, `stochastic_density_mse`): halve its value, `tolerance_pct: 0` — using the document's real key `direction` (`"lower_better"`), not v1's `lower_is_better`; zero-valued metrics (`*_win_fraction` at one seed) are never chosen because halving 0 regresses nothing | exit 1; stdout names that metric |
| `test_missing_baseline_exits_nonzero` | `--baseline <tmp>/absent.json` | exit ≠ 0, path in the message. Note the baseline is loaded **after** the run (`:188-222`), so this pays a full run; keep it in the parametrisation rather than adding a fourth |
| `test_cuda_request_fails_loud_without_cuda` | `--device cuda`, `CUDA_VISIBLE_DEVICES=""` | exit 1, `resolve_device` message |

**Gap written into the phase rather than noted:** `scripts/run_lshape_amr.py` has no
`--record-baseline` / `--baseline` / `--tolerance-pct`, unlike its two siblings, so the L-shape
headline cannot be regression-gated from its CLI. ~40 lines mirroring
`run_transfer_baseline_compare.py:155-173`. The strict xfail flips visibly when they land.

### 5.2 `tests/e2e/test_poc_baseline_cli_journey.py` — the generic CLI, exact self-diff

Measured cost of the `noyron_basis_cpu.yaml` run: **8 s** *(measured)*; no `slow`.

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_run_record_diff_chain_is_exact` | `pin_scenario_yaml(noyron_basis_cpu.yaml, device=e2e_device)` → `poc.cli run --config <copy> --output-dir <tmp>` → run id from `<tmp>/results/<id>/` → `record-baseline --output-dir <tmp> --run-id <id> --out <tmp>/b.json --tolerance-pct 10` → `diff --baseline <tmp>/b.json --output-dir <tmp> --run-id <id>` | three exit 0s. This is the **true** self-diff (same run id, no re-execution) and is exact on any device; 4 entries recorded *(measured)*; `payload["device"] == e2e_device` |
| `test_diff_against_a_tampered_baseline_exits_one` | tighten one strictly-positive `lower_better` entry | exit 1, metric named |
| `test_record_with_unknown_run_id_exits_one` | `--run-id nope` | exit 1, id in message *(measured)* |
| `test_result_json_round_trips` | the persisted JSON | `ScenarioResult.model_validate(payload)` in a fresh process (`results.py:110` writes `model_dump(mode="json")` with `default=str`, which is exactly the kind of write that can stop round-tripping silently) |

## 6. Phase 3 — agents and governance CLIs as processes

### 6.1 `tests/e2e/test_agents_cli_journey.py`

Guards: CLAUDE.md row *"Agents hardening (lifecycle hooks + timeout + scaffold)"*.

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_scaffold_dry_run_writes_nothing` | `python -m src.agents.cli scaffold demo_probe --root <tmp> --dry-run` | exit 0; three planned paths printed; `<tmp>` empty |
| `test_scaffold_then_rerun_refuses_with_exit_one` | same without `--dry-run`, twice | first exit 0 + three files; second exit **1** (`FileExistsError` → `typer.Exit(1)`, `scaffold.py:220-221`) naming the path |
| `test_list_agents_names_the_four_builtins` | `list-agents` | exit 0; `coupling`, `decomposition`, `meta`, `solver`. Docstring records that `research` is a subcommand (`AgentType.RESEARCH`) with **no registered agent** — an asymmetry the owner may want to close, but not by a test pretending otherwise |
| `test_research_subcommand_rejects_a_missing_config_with_exit_two` | `research --config <tmp>/absent.yaml` | exit 2 |

The v1 "import the generated agent from a fresh interpreter" test is dropped (§0.1);
`tests/agents/test_scaffold_cli.py::test_generated_agent_imports_and_runs` already proves it
via `exec()`, which is the only sound way given the generated module's imports.

### 6.2 `tests/e2e/test_governance_cli_journey.py`

Guards: CLAUDE.md row *"Abstraction audit (F0/F1 screen)"*.

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_audit_abstractions_with_the_ci_gated_argv_is_clean` | parse the `lint` job for the **first** `python -m scripts.audit_abstractions … --fail-on-missing` line — the one with explicit roots (the second step expands `$(ls -d src/*/ …)` and is not hermetically parseable, so it is deliberately not the source); run it | exit 0; the parsed root set equals the four CLAUDE.md names — a fifth or a missing one fails, so the row and the workflow cannot diverge |
| `test_audit_report_only_root_exits_zero_with_findings` | `python -m scripts.audit_abstractions src/backend` | exit 0 *(measured)*; findings printed |

Dropped from v1: `check_focus --base HEAD --head HEAD` (empty diff by construction — nothing
short of a crash can fail it) and `check_doc_links` on this file (self-referential; the
pre-commit gate already covers `docs/`). A real `check_focus` process test needs a synthetic
repo with a two-track diff; `tests/scripts/test_check_focus.py` builds those in-process and
adding a process boundary there is a small follow-up, not an E2E journey.

### 6.3 `tests/e2e/test_untested_entry_points.py`

The scripts with no test at all. `--help` is not coverage; each gets a real run.

| Script | Real run | Asserts |
| --- | --- | --- |
| `scripts/demo_pde_solver.py` | `--pde-type poisson --n-episodes 1 --mcts-sims 2 --output-dir <tmp> --no-plots --seed 1` (numpy-only, zero torch references) | exit 0; output under `<tmp>` |
| `scripts/export_helix_stl.py` | `--n-turns 1 --output <tmp>/h.stl` | exit 0; file is **binary STL**: 80-byte header, `uint32` triangle count matching `(len − 84) / 50` — the script writes binary only (`:131-132`); v1's "starts with `solid`" alternative was dead |
| `scripts/evaluate_model.py` | Phase 4, on the Go checkpoint the trainer writes | — |

## 7. Phase 4 — checkpoint lifecycle through the shipped tools

Guards: CLAUDE.md rows *"Trained evaluator"* and *"Checkpoint deserialization safety"*. The
only phase whose runtime tracks host speed; `slow`, `E2E_TRAINING_TIMEOUT_S`.

**Prerequisite src change, found by review:** `scripts/inspect_checkpoint.py` exits 0 on every
failure (§0.1). Fix: `inspect()` returns an int, `main()` returns it, `SystemExit(main())`.
`tests/scripts/test_cli_pickle_flags.py` is in-process, so the shell-visible exit code is the
new thing this phase asserts — the docstring says exactly that and nothing else, or it reads
as a copy of the security suite.

`tests/e2e/test_checkpoint_lifecycle_journey.py`, one module-scoped fixture running
`python -m scripts.train --config-name=train_fast training.total_steps=2 checkpoint_dir=<tmp> device=<e2e_device>`
(`checkpoint_dir` is what relocates the write; `hydra.run.dir` does not — `config/train.yaml:136`
resolves against cwd). A final checkpoint is saved regardless of `checkpoint_interval`
(`trainer.py:1093`). The fixture fails, never skips, if no `*.pt` appears.

| Test | Journey | Asserts |
| --- | --- | --- |
| `test_trainer_writes_a_versioned_checkpoint` | fixture | one `*.pt`; `load_torch_checkpoint(path, map_location="cpu")` reads it (explicit `map_location` — on a CUDA host the tensors were saved on `cuda`); version key present |
| `test_inspect_checkpoint_exits_zero_and_prints_keys` | `python -m scripts.inspect_checkpoint <pt>` | exit 0; `Keys:` line includes the version key |
| `test_inspect_exits_nonzero_on_a_marker_payload` | the marker pickle `tests/security/test_checkpoint_safety.py` uses | exit ≠ 0 after the script fix; `allow_unsafe_pickle` **not** exercised here |
| `test_solver_consumes_it_on_the_requested_device` | `py_runner`: `AlphaGalerkinSolver(AlphaGalerkinConfig(evaluator="trained", checkpoint_path=<pt>, device=e2e_device, n_mcts_simulations=4))` on Poisson | finite `l2_error`; **inside the subprocess** `next(model.parameters()).device.type == e2e_device.split(":")[0]` — without the explicit `device=`, the config defaults to `cuda` and falls back silently |
| `test_strict_load_rejects_an_action_space_mismatch` | a **second** config with `checkpoint_strict_load=True` against a deliberately mismatched game (default is `False`, `solver.py:223`; v1 asserted both outcomes on one config) | loud failure naming the mismatch |
| `test_evaluate_model_runs_on_it` | `python -m scripts.evaluate_model --model <pt> --n-games 1 --board-size 9 --device <e2e_device>` — a Go evaluator; the `train_fast` checkpoint is a Go model, so this is the right consumer | exit 0 |

## 8. Explicitly not in this plan

| Item | Why |
| --- | --- |
| Frozen tracks | `docs/FOCUS.md`; `check_focus` would flag it alongside this work. |
| LM Studio journeys | Need a live server (`LM_STUDIO_URL`); already manual in CLAUDE.md Next Steps. Not a device question. |
| A real `docker build` | A CI job decision, not a test (`make docker-build` and the hermetic guard exist). |
| Enabling `cudnn.deterministic` repo-wide | Behaviour and performance change; only the false docstring is corrected here. |
| Byte-comparing regenerated artifacts to `results/*.csv` | Device-sensitive formatting; the charter guard checks provenance, not bytes. |
| A `.run.json` for `results/lshape_mcts_vs_dorfler.csv` | Predates the manifest module; the charter guard records the accepted gap; regenerating it is a `run-provenance` task. |

## 9. Budget, placement and order

Measured where a measurement exists; otherwise per-process torch import ≈ 3–5 s dominates.

| Phase | Files | Tests | Runtime (CPU) | CI job | Markers |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 guards + wiring + 7 assertion fixes + 2 src fixes | ~14 | < 2 s hermetic; existing suite 130–310 s *(measured)* | `test-e2e` (new), guards in `lint` | — |
| 1 | 3 | ~14 | 4.1 < 5 s/run; 4.2 6 s/run × 3; 4.3 tensor < 5 s, fem unmeasured | `test-e2e`; 4.3 fem in `test-extras` | `e2e`; `fem_required` on one |
| 2 | 2 | ~10 | ~1 min transfer/stochastic pairs (est.); 5.2 8 s/run *(measured)* | `test-e2e` | `e2e` |
| 3 | 3 | ~8 | ~40 s, process startup (est.) | `test-e2e` | `e2e` |
| 4 | 1 | ~6 | one training run + 4 processes (est. 1–3 min) | `test-e2e` | `e2e`, `slow` |

Order 0 → 1 → 2 → 3 → 4, one PR per phase; 3 and 4 can be parallel after 1. If `test-e2e`
passes 15 minutes, move Phase 4 to `test-slow` (which has the `workflow_dispatch` hatch)
rather than widening the cap.

## 10. Definition of done, per test

- [ ] Runs via `cli_runner` or `py_runner`; never `patch.object(sys, "argv")`.
- [ ] Exact exit code, and `status` where a result exists.
- [ ] Device comes from `e2e_device`; no `"cuda"`/`"cpu"` literal in a journey (the negative test's `--device cuda` + `CUDA_VISIBLE_DEVICES=""` is the one allowed pair, and it is a fixture).
- [ ] Every timeout is one of the three tier constants.
- [ ] Every output under `tmp_path`; `git status` clean afterwards.
- [ ] Docstring names the spec AC / charter row / Regression Surface row it guards, and states whether the surface is torch or numpy-only.
- [ ] PR names the planted mutation and the test that failed on it.
- [ ] Markers per §2 rule 8.
- [ ] Appears in CLAUDE.md's Regression Surface (one row per phase).
- [ ] `ruff check`, `ruff format --check`, `mypy --strict` clean.

## 11. Measured baseline this plan was written against

Container: 4 CPUs, no CUDA, torch 2.13.0+cu130, scikit-fem absent. All 81 tests pass.

| Run | Result |
| --- | --- |
| `pytest tests/e2e/ -m "not gpu_required" -q --deselect tests/e2e/test_user_journey_go_training.py::test_user_journey_go_training` | 80 passed, **308.6 s** — overlapped a stray earlier process; upper bound. ~~An earlier draft of this row wrote the command without the `--deselect` and attributed the 80/81 split to the marker filter~~ — **corrected 2026-09-04 (Copilot review, PR #144)**: *no* test in `tests/e2e/` carries `gpu_required`, so `-m "not gpu_required"` alone selects all 81. The deselection was an explicit flag, dropped when the command was transcribed, which is exactly the kind of silently-wrong provenance this plan's §2 rule 6 exists to prevent. |
| `test_quick_validation_journey` + `test_sbir_demo` + `test_centaur_e2e` + `test_chess_training_e2e` + `test_poc_scenario_journey` (38) | 38 passed, **60.7 s** |
| `test_chess_engine_e2e` + `test_config_validation` + `test_cli_journey` (40) | 40 passed, **63.1 s** |
| three `test_user_journey_*` (3) | 3 passed, **7.0 s** |
| Slowest single tests | `test_train_physics_minimal` 15.3 s; `test_engine_vs_random_game` 14.0 s; `test_user_journey_go_training` 4.6 s; every `poc.cli`/`agents.cli` subprocess ≈ 3.2–3.5 s (torch import) |
| Plan-argv runs (by the reviewer, same container) | `run_lshape_amr … --max-dof 120 --n-simulations 2`: exit 0, 6 s, ratio 0.9627, seed std 0; `poc.cli run --config noyron_basis_cpu.yaml`: exit 0, 8 s, 4 baseline entries; `run_adaptive_vs_uniform --max-dof 120`: < 5 s, final uniform DOF 208 |

---

## 12. Implementation record (2026-09-04)

### 12.1 What shipped

**Phase 0 — the tier is now CI-visible and device-agnostic.**

| Change | Where |
| --- | --- |
| New blocking CI job `test-e2e` (`-m "not gpu_required and not fem_required"`, `E2E_DEVICE: cpu`, 30-min cap), in `ci-success.needs` with a hard `exit 1` gate | `.github/workflows/ci.yml` |
| Positively-selecting `fem_required` E2E step in `test-extras`, the only job that installs `[fem]` | `.github/workflows/ci.yml` |
| `make test-e2e` runs the whole directory with CI's filter (was **3 of 81** tests via a glob) | `Makefile` |
| `E2E_DEVICE` (default `auto`), resolved **once at conftest import** through `resolve_device`; `e2e_device` / `e2e_device_type` fixtures; terminal-summary device line | `tests/e2e/conftest.py` |
| `py_runner` (`python -c`), `pin_scenario_yaml` (refuses to pin an undeclared key), `NO_CUDA_ENV`, `CLIResult.output`, shared `_run_subprocess`; `cli_runner`/`py_runner`/`pin_scenario_yaml` made session-scoped so a module fixture can run an expensive harness once | `tests/e2e/conftest.py` |
| `gpu_required` skips now **report a count**, like the `fem_required` hook next to them | `conftest.py` |
| `ScenarioResult.device` reports the **execution** device via `BaseScenario.execution_device_label()`; scenarios with no device concept keep the old expression | `src/poc/registry.py` |
| `set_global_seeds` docstring no longer claims GPU determinism it does not implement | `src/seeding.py` |
| All **seven** set-valued exit assertions replaced with exact codes (each measured first) | `tests/e2e/*` |
| `e2e` marker added to the two files that lacked it — `-m e2e` now selects **81**, was 76 | `tests/e2e/*` |
| Guards: `tests/docs/test_e2e_visibility.py` (49 tests) + `tests/docs/test_marker_vocabulary.py` (45), sharing `tests/support/workflows.py` and `tests/support/marker_expr.py` | `tests/docs/`, `tests/support/` |

**Phases 1–4 — the journeys.** Nine new files under `tests/e2e/`:
`test_adaptive_vs_uniform_journey.py`, `test_lshape_amr_journey.py`,
`test_refinement_substrate_journey.py`, `test_baseline_gate_journey.py`,
`test_poc_baseline_cli_journey.py`, `test_agents_cli_journey.py`,
`test_governance_cli_journey.py`, `test_untested_entry_points.py`,
`test_checkpoint_lifecycle_journey.py`.

**Src changes the journeys required** (each additive and separately tested):

- `AdequacyGateConfig` + `gate_violations` + `measure_adequacy` + `default_adequacy_gate`
  **promoted** from `tests/research/test_amr_arena_interpretability.py` into
  `src/research/substrates/{config,sweep}.py`. Byte-identical defaults; the test file keeps
  re-exports so `TestGatePredicate` exercises the promoted function rather than a copy.
  A caller who runs a sweep can now ask for the adequacy verdict — previously only pytest could.
- `src/poc/baselines/cli_support.py`: shared `add_baseline_arguments` / `handle_baseline_flags`.
  `scripts/run_lshape_amr.py` gained `--record-baseline` / `--baseline` / `--tolerance-pct`,
  closing the gap where the L-shape headline could not be regression-gated from its own CLI.
  Policy (which metrics are stable, which are higher-better) stays per-harness; only the
  mechanism is shared.
- `scripts/inspect_checkpoint.py` now returns a real exit code. It previously exited **0 whether
  the checkpoint deserialized or not**.
- `collect_hardware_tag()` in `src/research/run_manifest.py`, wired into
  `scripts/run_adaptive_vs_uniform.py`: sidecars recorded `hardware_tag: "unknown"` because no
  harness ever set it. Now e.g. `x86_64-4cpu`, plus the CUDA device name when one is visible.

### 12.2 What the implementation found wrong in *this* plan (v2)

Same convention as §0.1: the wrong statement stays, marked.

| v2 said | Actually | Fix |
| --- | --- | --- |
| §3 guard clause (b): "a step selects `tests/e2e/` whose `-m` does not exclude `e2e`" | **Defeated by its own mutation.** The `test-extras` fem step (`-m "fem_required and not gpu_required"`) satisfies that literally — it selects the directory and never mentions `e2e` — so *deleting the entire `test-e2e` job left the guard green* | Strengthened to `expression_selects_plainly`: a qualifying step must also not narrow the run to some other positively-required marker. Mutation 1 only goes red under the strengthened form |
| §6.3: assert STL `len >= 84` and `uint32@80 == (len - 84) // 50` | **Passes on a zero-triangle file**: an empty exporter writes exactly the 84-byte preamble and `0 == 0` holds. Verified — under the planted mutation the file *was* 84 bytes and the relation *did* hold | Added `triangle_count > 0` (the load-bearing half) and replaced floor division with an exact tiling check `len == 84 + count * 50`, since `//` also absorbs a truncated payload |
| §4.3: "only the thresholds move; the predicate is a straight promotion" | `max_sweep_dof` returned `float` (the field defaults became floats), which `run_refinement_sweep(max_dof: int)` rejects under `mypy --strict` | Returns `int`, which also restores exact value identity with the original int-pair constant |
| §4.3 mutation (d): "tests 2 and 3 fail" | Test 2 is `fem_required` and **skips on CPU CI**, so the aliased-registry mutation would have gone unkilled on every CI run | A registry-distinctness assertion was added to test 3 (the un-marked one), which is what kills it |
| §6.1: assert `list-agents` output contains the agent names | Vacuous as a substring check — `structlog` prints `item_registered` lines naming all four agents *before* the table renders, so it passes against an empty table | Parses the rendered table's first column and compares an exact set |
| §5.1: lshape row is `xfail(strict=True)` until the baseline flags land | The flags landed in the same change, so it is a **live parametrisation**, not an xfail | Row is real; all three harnesses are gated |
| §2 rule 9 / §5.1: re-run drift needs a wide tolerance (the in-process test uses `--tolerance-pct 100000`) | **Measured: zero.** All three harnesses reproduce their stable metrics *exactly* at `--tolerance-pct 0` with the seed pinned, on this CPU container | `RERUN_TOLERANCE_PCT_CPU = 1.0` with the measurement recorded; the wide value applies only off-CPU, where no cuDNN determinism flag is set |
| §9: Phase 2 ≈ 3–6 min, Phase 4 `slow` | Phase 2 measured **2:54**; Phase 4 **22 s** | Phase 4 keeps `slow` (it is the only phase whose cost tracks host speed); estimates corrected in §12.3 |

### 12.3 Measured runtimes (4-CPU container, `E2E_DEVICE=auto` → cpu)

| File | Result | Wall |
| --- | --- | --- |
| `test_baseline_gate_journey.py` | 14 passed | 2:54 |
| `test_poc_baseline_cli_journey.py` | 5 passed | 31 s |
| Phase 1 (3 files) | 21 passed, 1 skipped (`fem_required`) | 59 s |
| Phase 3 (3 files) | 10 passed | 34 s |
| `test_checkpoint_lifecycle_journey.py` | 4 passed, 1 xfailed | 22 s |
| `tests/docs/` (incl. both new guards) | 178 passed | ~2 s |

### 12.4 A memory finding the wiring surfaced

Running the tier **whole** — which nothing had ever done — exposed a pre-existing
leak that has nothing to do with the new journeys.

Traced with a 2-second RSS sampler over the full 136-test selection on a 4-CPU,
16 GB container:

| Point in the run | pytest **parent** RSS |
| --- | --- |
| t=0–113 s, through `test_adaptive_vs_uniform_journey`, `test_agents_cli_journey`, `test_baseline_gate_journey` (all new, all subprocess-driven) | **flat at 594 MB** |
| t=203 s, `test_chess_engine_e2e.py` starts (pre-existing, 29 in-process tests) | **2,125 MB** |
| t=219 s, `test_chess_training_e2e.py` (pre-existing) | 2,913 MB |
| ... monotonic from there, never releasing ... | |
| t=461 s, `test_quick_validation_journey.py` | **13,093 MB** |

Peak parent 13.2 GB; the run was OOM-killed (SIGKILL, exit 137) at ~84% on two
separate attempts, at slightly different tests — the signature of a cumulative
allocation plus whatever peak lands on top, not of one greedy test.

What this does and does not say:

- **The new journeys do not leak.** Run first, all three subprocess-driven files
  hold the parent flat at 594 MB for nearly two minutes. The substrate journey
  measured alone peaks at 1.1 GB and finishes in 8 s.
- **The growth begins exactly at the first in-process file** and continues
  regardless of what runs afterwards, which is consistent with the allocator not
  returning freed pages rather than with a single runaway test.
- **A first hypothesis was wrong and is recorded as such**: captured subprocess
  output. Measured, each child emits 4–20 KB, ~3 MB across the tier — three
  orders of magnitude short. Not the cause.
- **It was invisible because the directory was never run whole.** CI ran exactly
  one of these files, alone, in the chess job. This is the same shape as the
  finding in §0: the defect and the reason nobody saw it are the same fact.

Root-causing it is chess/MCTS work, not E2E-test work, so it is **recorded, not
fixed**. What matters for this change is the consequence: the `test-e2e` job is
blocking, and a runner with less headroom than the peak will fail it. If that
happens, the mitigation is to split the single pytest invocation into two steps
in the same job — two processes, so memory is released in between — and that is
a workaround to be labelled as one, not a fix.

### 12.5 Deliberately still open

- **No shipped command produces a PDE-consumable checkpoint.**
  `scripts/train.py +game=pde_basis` crashes in `src/modeling/embeddings.py` with a tensor-shape
  mismatch: `config/train_fast.yaml`'s `operator.input_channels` is Go-shaped (17) and nothing
  reconciles it with the selected game's encoding. Pinned as a **strict xfail**
  (`test_pde_game_training_is_a_known_gap`) so it flips visibly when fixed. Fixing it is
  PDE-model work, not E2E-test work.
- **CLAUDE.md's Multi-Game Commands section documents `python -m scripts.train game=go`**, which
  errors — `game` is not a key in `config/train.yaml`, so Hydra requires `+game=go`.
- **`tests/docs/test_coverage_gate_integrity.py` still carries its own workflow parser**, now the
  one remaining copy alongside `tests/support/workflows.py`. It is heavily mutation-tested;
  collapsing it belongs in its own change.
- **`src/research/baselines.py` and the three `-m` vocabularies** are untouched here; the
  `make pre-pr` / CI exclusion-list duplication (backlog B7) is unchanged.
