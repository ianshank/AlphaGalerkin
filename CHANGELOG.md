# Changelog

All notable changes to AlphaGalerkin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Codec Model Zoo Phase 2-D (`src/video_compression/zoo/sweep.py`, `scripts/train_compression_zoo.py`)

- **Manifest-level sweep orchestrator** — `ZooSweep` drives every entry in
  a manifest through a configurable `EntryRunner`. `should_skip(zoo, entry)`
  inspects the persisted entry hash so reruns of an unchanged entry skip
  cleanly. `EntryStatus` + `SweepReport` are frozen dataclasses; the
  default `default_entry_runner` runs `ZooTrainer` in-process.
- **Slice A — multi-entry CLI** — `scripts/train_compression_zoo.py` adds
  `dry-run` / `train` subcommands operating on a manifest. Shared
  primitives extracted into `src/video_compression/zoo/cli_helpers.py`
  (load / resolve_path / load_codec_config / resolve_entry /
  resolve_codec_config_for_entry / override_entry / resolve_device);
  the original `train_compression_zoo_entry.py` re-imports them as
  `_underscored` aliases so existing tests continue to monkeypatch
  through the script module.
- **Slice B — parallel dispatch + subprocess runner** — `ZooSweep.run_parallel()`
  groups entries by device and dispatches one worker thread per device,
  keeping same-device entries serialized inside their worker. Statuses
  return in manifest order regardless of completion order.
  `make_subprocess_entry_runner(...)` returns an `EntryRunner` that
  re-invokes the existing single-entry CLI with `CUDA_VISIBLE_DEVICES`
  pinned to the parent's `cuda:N` index (translating the child's
  `--device` to `cuda:0` because only one GPU is visible). After exit 0,
  the parent reads `metrics.json` + checkpoint back and reconstructs a
  `ZooTrainingReport`. Tests inject a fake `subprocess.run` via the
  `subprocess_runner` hook.
- **`ZooTrainer` persists wall-clock** — `train_wallclock_s` /
  `eval_wallclock_s` now land in `metrics.json` so the subprocess runner
  can reconstruct them across process boundaries.
- **Gap-analysis coverage closure** — branch-wide tech-debt scan confirmed
  zero hardcoded values (sole literal `"cuda:0"` at `sweep.py:510` is a
  CUDA ABI constant — the only-visible-GPU always presents as `cuda:0` in
  a `CUDA_VISIBLE_DEVICES=N` subprocess, documented inline). Discovered
  `cli_helpers.py` at 68% coverage; added 22 unit tests in
  `tests/video_compression/zoo/test_cli_helpers.py` covering every public
  helper across YAML/JSON/unsupported-suffix/empty/non-dict load paths,
  absolute/cwd-relative/manifest-relative path resolution, codec-config
  round-trip, entry lookup/KeyError, codec-config-ref precedence/fallback
  /no-ref raise, override short-circuit, device-preference cascade.
  Lifts `cli_helpers.py` from 68% → **100%**; zoo-subpackage total
  **98.44%**.
- **Test surface** — 22 cli_helpers tests + 15 Slice B tests
  (`tests/video_compression/zoo/test_sweep_parallel.py`) + 9 Slice A
  tests (`tests/scripts/test_train_compression_zoo.py`) + 4 entry-CLI
  tests (`tests/scripts/test_train_compression_zoo_entry.py`) + 11
  sweep-unit tests + zoo-trainer tests; **162-test full
  zoo+scripts+training regression** passes; mypy --strict + ruff clean.

### Added — Codec Model Zoo Phase 2-B (`src/video_compression/zoo/`)

- **Dual-GPU model zoo for the 8-point R-D Lagrangian sweep** — new
  `src/video_compression/zoo/` subpackage that schedules an arbitrary
  `λ`-grid across heterogeneous CUDA devices. Targets the reference rig
  (RTX 5060 Ti 16 GiB at `cuda:0` + RTX 5060 8 GiB at `cuda:1`) but is
  resolution-/SKU-agnostic: the planner consumes a `list[DeviceCapability]`
  produced at runtime by `scan_devices()`, so the same code runs on a
  laptop CPU, a single-GPU box, or a multi-node cluster.
- **Pydantic-validated schemas** (`config.py`, 100% coverage) — every
  measurement-/training-affecting knob is a validated field with bounds:
  `lambda_rd > 0`, `target_psnr_db > 0`, `train_steps ≥ 1`,
  `warmup_steps ≤ train_steps`, Adam betas in `(0, 1)`, entry IDs match
  `^[a-zA-Z0-9_\-\.]+$`, parent-entry-id resolution, dedupe enforcement.
  Schema versions are module-level constants
  (`PERF_ZOO_MANIFEST_SCHEMA_VERSION = 1`,
  `PERF_ZOO_ENTRY_SCHEMA_VERSION = 1`). Zero hardcoded magic numbers.
- **Forward-compatible manifest** (`manifest.py`, 100%) — JSON or YAML
  load/save dispatched by file suffix (`.yaml` / `.yml` → YAML, else
  JSON). `_migrate_manifest_document` promotes unversioned manifests to
  v1, fails loud on newer-than-binary, and rejects non-int schema
  versions. The shipped `config/video_compression/zoo/lambda_grid.yaml`
  (8 points: λ ∈ {0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045, 0.09, 0.18})
  loads cleanly through the same path as user-authored YAML.
- **Heterogeneous-VRAM device planner** (`device_planner.py`, 100%) —
  four assignment strategies: `VRAM_AWARE` (default; best-fit packing
  by current headroom, falls back to largest-total over-commit when no
  device has room), `ROUND_ROBIN`, `SINGLE_DEVICE`, `MANUAL` (per-entry
  pin via `device="cuda:N"` / `"cpu"` / `"cuda"`). Explicit pins are
  pre-resolved out before strategy dispatch, so all strategies compose
  with manual overrides. `scan_devices()` does a runtime `import torch`
  to remain monkeypatch-friendly. Module-level `CPU_DEVICE_LABEL`
  constant; no string literals leaked.
- **Filesystem `VideoCodecZoo` registry** (`storage.py`, 100%) —
  per-entry directory layout (`<root>/<entry_id>/{checkpoint.pt,
  entry.json, metrics.json}`), atomic write semantics, GCS backend gated
  via `importlib.import_module("src.vertex.storage")` (raises
  `NotImplementedError` until Phase D wires it). Constants
  `CHECKPOINT_FILENAME`, `ENTRY_FILENAME`, `METRICS_FILENAME` are
  module-level.
- **Coverage gate** — `pyproject.toml` `coverage.run.omit` rebalanced
  from a global `src/video_compression/*` blanket to a per-subpackage
  list. The `zoo/` subpackage is omitted from the project-wide 85%
  gate (CI's fast suite uses `--ignore=tests/video_compression/`, so
  including `zoo/` globally would 0%-tank the gate); a dedicated per-
  module gate enforces the zoo coverage floor instead. Achieved
  coverage on the zoo subpackage: **100% line + branch** across all
  five modules (`__init__.py`, `config.py`, `device_planner.py`,
  `manifest.py`, `storage.py`).
- **Test suite** (`tests/video_compression/zoo/`, 68 tests) —
  `test_config.py` (22 tests, schema + validator coverage),
  `test_manifest.py` (11 tests including YAML round-trip + shipped-grid
  smoke + Hypothesis property-based migration test),
  `test_device_planner.py` (14 tests across all four strategies +
  reference-rig fixture + CPU-only fixture),
  `test_storage.py` (8 tests, 1 GCS-skip),
  `test_edge_cases.py` (13 tests targeting all originally-uncovered
  branches: `DevicePlan.device_for` KeyError, bare-`cuda` resolution
  under MANUAL, CPU pin under VRAM_AWARE, `_resolve_run_target` cuda /
  cuda:N / invalid paths, MANUAL with missing pin, `list_entries` with
  removed root, non-dict checkpoint / metrics payloads).
- **E2E validation on live dual-GPU hardware** — `lambda_grid.yaml`
  loads, `scan_devices()` reports both cards correctly
  (`cuda:0=RTX 5060 Ti 16 GiB`, `cuda:1=RTX 5060 8 GiB`),
  `assign_devices` produces a deterministic plan with structured
  `structlog` events bound to `entry_id` / `device` / `strategy`.
  Phase 0 perf harness + Phase 1 runtime backends regress green
  (228 passed, pre-existing ONNX-runtime test failures on `cuda:0` are
  unrelated to this branch).

### Added — Codec Performance Benchmark Phase 0 (`src/video_compression/perf/`)

- **GPU-primary benchmark harness** — new `PerfBenchmark(BaseExecutable)` with `device_preference="cuda"` default, per-profile `cuda:N` pinning so a single sweep covers both cards of the reference dual-GPU rig (RTX 5060 Ti 16 GB at `cuda:0` + RTX 5060 8 GB at `cuda:1`). Indexed-CUDA resolver wraps `src/poc/device.resolve_device` without disturbing existing PoC scenarios.
- **Pydantic-validated config with zero hardcoded values** — `PerfBenchmarkConfig` surfaces every measurement-affecting knob (resolution / batch / phase / warmup / repeats / tolerance / track-VRAM / pattern / data-seed) as validated fields with bounds. `RuntimeProfile` and `ResolutionSpec` schemas pin labels and devices for stable cell keys. Schema versions are module-level constants (`PERF_BENCHMARK_CONFIG_SCHEMA_VERSION`, `PERF_BASELINE_DOCUMENT_SCHEMA_VERSION`, `PERF_BASELINE_ENTRY_SCHEMA_VERSION`).
- **Forward-compatible baseline registry** — `BaselineRegistry` load/save/diff with explicit JSON schema versioning, `extra="ignore"` for unknown future fields, and `_migrate_baseline_document` hook with an unversioned-to-v1 migration. Per-entry tolerance overrides allow tightening regression gates on critical cells without weakening the global threshold.
- **Three YAML configs** — `config/perf/smoke.yaml` (CPU CI gate, ~10 s), `config/perf/cuda0_headline.yaml` (single-card 16 GB primary), `config/perf/default.yaml` (dual-card sweep across `cuda:0` + `cuda:1`).
- **CLI `scripts/benchmark_codec.py`** — `run` / `record-baseline` / `diff` subcommands with structured `structlog` events bound to `benchmark_id` + `cell_key`. Argparse-based; no typer dep.
- **`BenchmarkSubject` Protocol** — runtime-agnostic interface for the timed object (`prepare` / `step` / `teardown`). Phase 1 (ONNX Runtime, TensorRT, FP16, `torch.compile`) drops new subjects in without touching the benchmark loop. Extended docstring includes a runnable Phase-1 example.
- **Coverage gate** — new `.github/workflows/codec-perf-coverage.yml` enforces ≥85% per-module coverage on `src/video_compression/perf/`. Inline-coveragerc heredoc with `include = src/video_compression/perf/*.py` (same pattern as `regression-surface.yml::noyron-hx-coverage-gate` on master). Achieved coverage: `__init__.py` 100% / `baseline.py` 99% / `benchmark.py` 100% / `config.py` 96% / `device.py` 90% / `metrics.py` 100% / `subjects.py` 97% — **TOTAL 98.42%**.
- **Defensive raise-paths covered** — three new tests in `TestDefensiveRaisePaths` (`tests/video_compression/perf/test_benchmark_smoke.py`) lock in: `fail_fast=True` propagates non-`NotImplementedError` exceptions; non-FP32 precision raises clean `NotImplementedError` (Phase-1 stub); `report_from_result` rejects `ExecutionResult` lacking the `"report"` artifact with a clear `KeyError`.

### Added — Noyron HX v1 Hardening (`src/pde/sdf.py`, `src/pde/geometry_picogk.py`, `src/poc/scenarios/noyron_hx.py`)

- **Voxel-FDM training consistency** — `NoyronHXScenario` now trains directly on the cached FDM solution when `ref_solver_kind="voxel_fdm"`. Previously the scenario trained on the harmonic surrogate but graded against FDM; the head-line `mse_low < 5e-4` / `mse_high < 1e-3` thresholds were unreachable in FDM mode. The cached solution is built lazily via `_voxel_fdm_reference()` and reused at evaluation, so reference and supervision come from the same field.
- **Surfaced scenario metrics** — `accept_rate` (from `PicoGKDomain.volume_accept_rate`), `train_time_s`, `eval_time_s`, and `train_loss_final` are now recorded in `ScenarioResult.metrics`. Timing values are captured via `ScenarioLogger.timed(...)` context-manager and propagated through the public metric dict.
- **Bisection / grid-search fallback for SDF projections** — `AnalyticalHelixSDF._nearest_t` gains a coarse-grid + Newton-refine fallback (opt-out via `enable_fallback=False`); `PicoGKDomain._project_to_surface` gains a bracketed bisection along the central-difference gradient (opt-out via `enable_bisection_fallback=False`). Both restore robustness on thin tubes (`r/R << 0.1`) where the original Newton iteration could stall.
- **`PicoGKDomain.volume_accept_rate`** — new read-only property that exposes the empirical interior acceptance rate computed at construction by the existing Monte-Carlo volume estimator. No re-sampling cost; the rate is cached on `_volume_accept_rate`.
- **`NoyronHXScenarioConfig.helix_n_turns`** default aligned to **5** across the Pydantic config, `config/scenarios/noyron_hx.yaml`, and `AnalyticalHelixSDF`. Previously the config-class default was 3 while the YAML used 5 — instantiating the config in code produced a different geometry than the headline run.
- **Module-level numerical-stability constants** (replace previously hardcoded literals):
  - `DEFAULT_TRANSFER_RATIO_FLOOR: float = 1e-12` — division floor for `mse_high / mse_low`.
  - `DEFAULT_NORMALIZE_EXTENT_FLOOR: float = 1e-9` — bbox-extent clamp in `_normalize`.
  - `EVAL_SEED_STRIDE: int = 9973` — prime offset between low- and high-density evaluation seeds.
- **`NoyronHXScenario._draw_pool_indices(n_pool, n_pts)`** — single helper for sampling indices from the cached FDM voxel pool. Replaces duplicated `randperm` / `randint` selection logic that previously appeared inline in both `_sample_voxel_fdm_batch` and `_evaluate`. Validates `n_pool > 0` and `n_pts > 0` and routes through-replacement sampling via `randint` when `n_pts > n_pool`.

### Added — Noyron HX Test Suite (`tests/pde/test_sdf.py`, `tests/pde/test_picogk_domain.py`, `tests/poc/test_noyron_hx_scenario.py`)

- **43 new test cases** covering: SDF fallback (disabled / param validation / grid scaling / pathological initial guess / no-Newton-refine branch), `PicoGKDomain` constructor validators (`grad_epsilon`, `max_oversample`, `projection_max_iters`, `min_grad_norm_sq`), bisection-fallback no-op when all-converged, projection-converged log branch, `accept_rate` / `train_time_s` / `eval_time_s` metric round-trip, voxel-FDM uses FDM (not harmonic) supervision, voxel-FDM cache cleared on teardown, `_draw_pool_indices` semantics, and module-constant invariants.
- **Per-module coverage**: `src/pde/sdf.py` **100%**, `src/pde/geometry_picogk.py` **100%**, `src/poc/config_noyron.py` **100%**, `src/poc/scenarios/noyron_hx.py` **97%** — all well above the project 85% gate.

### Added — Learned PDE Evaluator (`src/alphagalerkin/`)

- **`AlphaGalerkinConfig.evaluator="trained"`** — re-enables the network-backed evaluator literal that was removed in the DOE Genesis PR. The trained branch loads an `AlphaGalerkinModel` checkpoint via `create_model_from_checkpoint` and wraps it in the existing `FNetEvaluator`, providing learned policy/value priors to MCTS rather than the uniform prior of `RandomEvaluator`. Closes the only non-trivial entry under *Known Issues* in `CLAUDE.md`.
- **`checkpoint_path: Path | None`** Pydantic field with a `model_validator(mode="after")` that fails fast at config-construction time when `evaluator="trained"` is paired with a missing or non-existent checkpoint.
- **GPU-primary default** — `AlphaGalerkinConfig.device` default flipped from `"cpu"` to `"cuda"`. New module-level `_resolve_device_cached` helper (cached via `functools.cache`) falls back to CPU at runtime when `torch.cuda.is_available()` is False, emitting at most one downgrade warning per unique device string for the lifetime of the process. The random/uniform evaluator path skips device resolution entirely (it is device-agnostic) so CPU-only users do not see spurious `cuda_requested_but_unavailable` warnings under the new default. `config/train_pde.yaml` updated to `device: auto` (the `Trainer`'s native CUDA-availability fallback) — note that the solver's runtime fallback is solver-only and does not apply to the training pipeline.
- **Trained-evaluator instance cache** — `AlphaGalerkinSolver._build_trained_evaluator()` constructs the `FNetEvaluator` once per solver instance and reuses it across subsequent `solve()` calls so benchmark suites that iterate over many PDEs do not pay repeated disk I/O + model-init cost. `reset_cache()` invalidates the cache for callers that swap checkpoints during a long-running process.
- **New evaluator config fields** (replace previously hardcoded values):
  - `evaluator_temperature: float` (gt=0.0, default=1.0) — softmax temperature for trained-evaluator policy logits.
  - `evaluator_use_fast_path: bool` (default=True) — toggle the FNet fast-forward path inside `FNetEvaluator`.
  - `checkpoint_strict_load: bool` (default=False) — controls strict shape matching on `create_model_from_checkpoint`; the default tolerates policy-head shape mismatches across PDEs.

### Added — Trained Evaluator Tests (`tests/alphagalerkin/test_trained_evaluator.py`)

- **8 new test classes / parameterized cases** covering evaluator dispatch, action-space mismatch graceful degradation, device resolution caching, trained-evaluator instance caching, and config-field propagation. The GPU smoke test is gated on `@pytest.mark.gpu_required` and auto-skips on CPU CI via the root `conftest.py` hook.
- **Per-module coverage** on `src/alphagalerkin/` raised to **94%** (gate: 85%).

### Changed

- **`src/alphagalerkin/solver.py`** — module docstring rewritten to document the three evaluator modes; previously hardcoded `temperature=1.0` / `use_fast_path=True` / `strict=False` in `_build_mcts` removed in favour of the new Pydantic config fields.
- **`tests/alphagalerkin/test_solver.py`** — `test_trained_evaluator_rejected_by_config` flipped into `test_trained_evaluator_requires_checkpoint`, asserting both missing-path and non-existent-path failure modes surface as `ValidationError`.
- **`config/train_pde.yaml`** — `device: cpu` → `device: auto` to use the trainer's native CUDA-availability fallback (`BaseTrainer` resolves `auto` via `torch.cuda.is_available()`); the solver's `_resolve_device_cached` runtime fallback is solver-only and does not apply to the training pipeline.

### Documentation

- **`CLAUDE.md`** — drop the "trained-evaluator stub" entry from *Known Issues*; add the *Learned PDE Evaluator Wired* milestone for 2026-04-25.
- **`docs/architecture/c4_mermaid.md`** — extend the Container Diagram with the trained-evaluator path through `FNetEvaluator` and the on-instance evaluator cache.

### Added — E2E Dashboard (`dashboard/`)

- **`dashboard/app.py`** — Gradio Blocks application factory (`build_app()`) and CLI entry point (`main()`). Launches a tabbed UI exposing all AlphaGalerkin capabilities at `http://localhost:7860`. Accepts `--host`, `--port`, `--share`, `--debug` flags.

- **`dashboard/config.py`** — Full Pydantic v2 config hierarchy eliminating every hardcoded value:
  `AppConfig`, `GameConfig`, `PDEConfig`, `ComplexityRunConfig`, `StabilityRunConfig`,
  `TransferMilestone`, `PoCConfig`, `TrainingConfig`, `DashboardConfig`.
  `DEFAULT_CONFIG` singleton for zero-configuration startup.

- **`dashboard/utils.py`** — Shared utility module:
  - `fig_to_pil()` — always closes matplotlib figure (even on exception); `.copy()` detaches from buffer
  - `device_str()` — CUDA/CPU detection with graceful fallback
  - `format_exc()` — consistent exception formatting
  - `configure_structlog()` — idempotent structured logging setup

- **`dashboard/tabs/game_tab.py`** — Go AI tab. Thread-safe lazy model loading via `threading.Lock` (double-checked locking). Human vs AI and AI vs AI modes with 9×9/13×13/19×19 board support (zero-shot transfer). Config-injected via `GameConfig`.

- **`dashboard/tabs/pde_tab.py`** — Interactive Poisson equation solver. Five charge patterns (Point Charge, Dipole, Quadrupole, Ring, Random), multi-resolution comparison with zoom-upsampling MSE. Config-injected via `PDEConfig`.

- **`dashboard/tabs/poc_tab.py`** — PoC scenario runner. O(N) complexity benchmark, LBB stability monitoring, zero-shot transfer milestone display. Module-level optional imports for test patchability. Config-injected via `PoCConfig`.

- **`dashboard/tabs/training_tab.py`** — Architecture summary, simulated training curves (policy/value/LBB losses), and loss breakdown diagram. Config-injected via `TrainingConfig`.

### Added — Dashboard Test Suite (`tests/dashboard/`)

- **203 tests**, **89% line coverage** (gate: 85%), all passing with zero ruff violations.
- `conftest.py` — shared fixtures, `matplotlib.use("Agg")`, config fixture hierarchy, mock scenario results, charge-grid fixtures.
- `test_app.py` (24 tests) — CSS builder, arg parser, `build_app()`, `main()`.
- `test_config.py` (31 tests) — all Pydantic models, validation errors, JSON round-trip.
- `test_utils.py` (24 tests) — `fig_to_pil` (close on error, detached buffer), `device_str`, `format_exc`, `configure_structlog`.
- `test_pde_tab.py` (37 tests) — all charge patterns, Poisson solve integration, `solve_and_visualize`, `compare_resolutions` with shape-matching mock.
- `test_poc_tab.py` (32 tests) — `_parse_int_list`, `run_complexity`, `run_stability` (mocked), `show_transfer_milestone` (live).
- `test_training_tab.py` (28 tests) — model summary (fallback on import error), training curves, loss breakdown.
- `test_game_tab.py` (27 tests) — `autouse` fixture resetting module globals, fallback board, `_ensure_loaded` idempotency, human/AI move handlers.

- **Intercept Module** (`src/intercept/`)
  - `InterceptGame` implementing `GameInterface` protocol for MCTS-guided missile defense
  - 6-DOF rigid body dynamics (`dynamics.py`, `interceptor_dynamics.py`)
  - Proportional Navigation guidance (`guidance.py`)
  - `ExtendedKalmanFilter` for target tracking (`tracking.py`)
  - `RadarSensor`, `SensorFusion` for multi-sensor tracking (`sensors.py`)
  - `HungarianAssigner` for weapon-target assignment (`assignment.py`)
  - `ISAAtmosphere`, `WindModel` for atmospheric modeling (`atmosphere.py`)
  - `AeroModel`, `TabularAeroModel` for aerodynamic coefficients (`aero.py`)
  - `FrameTransform`, `QuaternionOps` for reference frame conversions (`frames.py`)
  - Pydantic-validated `InterceptorConfig`, `EngagementConfig`, `ThreatConfig`

- **Backend Abstraction** (`src/backend/`)
  - `BackendInterface` protocol for unified PyTorch/JAX operations
  - `TorchBackend`, `JaxBackend` implementations
  - `Array`, `Precision`, `DeviceType` type abstractions (`types.py`)
  - Random number generator abstraction (`rng.py`)
  - Backend-aware logging and debug utilities

- **Prototyping Module** (`src/prototyping/`)
  - `ModelBuilder`, `PrototypeModel` for rapid architecture iteration
  - `QuickTrainer`, `TrainResult` for fast experiment loops
  - `QuickEvaluator`, `EvalResult` for quick model evaluation
  - `DataGenerator`, `SyntheticData` for synthetic data creation
  - `Visualizer` with multiple plot types
  - `ExperimentTemplate`, `TemplateRegistry` for experiment patterns

- **Analysis Module** (`src/analysis/`)
  - `PositionEvaluator`, `EvaluationResult` for position evaluation
  - `GameReviewer`, `MoveAnalysis` for game review and move quality assessment
  - `PatternMatcher`, `PatternLibrary` for board pattern detection
  - `GameStatistics`, `StatisticsCollector` for game statistics aggregation
  - `AnalysisConfig`, `AnalysisMode` Pydantic configuration

- **Tournament Module** (`src/tournament/`)
  - `TournamentManager`, `TournamentState` supporting Round-Robin, Swiss, Elimination formats
  - `TournamentScheduler` for match scheduling
  - `EloRating`, `RatingSystem` for player rating computation
  - `Player`, `PlayerRegistry` for participant management
  - `Match`, `MatchResult`, `MatchStatus` for match tracking

### Changed

- **`pyproject.toml`** — Added `[[tool.mypy.overrides]]` for `dashboard.*` modules (relaxed strict checks for Gradio code). Added `[tool.coverage.report]` with `fail_under = 85` and `show_missing = true`. Added `dashboard` pytest marker.
- **Gradio 6 compatibility** — CSS argument moved from `Blocks()` constructor to `launch()`.

> **Branch and PR cleanup** — removed 28 stale remote branches and 6 open stale PRs.

## [0.3.0] - 2026-04-01

### Summary

Key highlights of this release:

- **Chess Self-Play Training Pipeline** — AlphaZero methodology, 4672-action dense policy, 119-channel state encoding
- **SBIR Readiness Infrastructure** — Navy N252-088, DOE ASCR, NSF SBIR, AFWERX proposal configs and benchmark suite
- **Advanced PDE Operators** — NavierStokes (Taylor-Green), L-shaped Poisson (singularity), enhanced Burgers (Cole-Hopf)
- **Domain Geometry & Time-Stepping module** — Rectangular, L-shaped, Cylinder domains; ForwardEuler, RK4, CrankNicolson
- **Multi-Agent Swarm Planning** — PettingZoo `ParallelEnv` adapter, potential field obstacle avoidance
- **Unified Loss Package & BaseTrainer consolidation** — `LossRegistry`, `get_loss()` factory, shared AMP/grad/LR in `BaseTrainer`
- **CI/CD hardening** — 85% coverage gates, nightly schedule, Stage 8 chess pipeline
- **218+ new tests** across PDE, research, training, and games modules

---

### Added

- **SBIR Readiness Infrastructure** (Navy N252-088, DOE ASCR, NSF, AFWERX)
  - `config/proposals/navy_n252_088.yaml`, `nsf_sbir.yaml` — SBIR-specific benchmark configs
  - `config/benchmarks/sbir_suite.yaml` — 3-problem benchmark suite (L-shaped Poisson, Burgers shock, NS Taylor-Green)
  - `src/research/baselines.py` — Classical PDE solver baselines: UniformFDMSolver, DorflerAMRSolver, SimplePINNSolver
  - `src/research/pde_benchmarks.py` — PDEBenchmarkRunner with JSON/Markdown report generation and convergence rate computation
  - `docs/proposals/templates/sbir_phase1.md` — Reusable SBIR Phase I proposal template
  - `docs/proposals/IP_STRATEGY.md` — 3 provisional patent claims, trade secret boundaries, publication plan

- **Advanced PDE Operators**
  - `NavierStokesOperator` — Taylor-Green vortex benchmark with analytical solution, configurable Re
  - `BurgersOperator` enhanced — Cole-Hopf exact solution, configurable shock params, convergence rate method
  - `LShapedPoissonOperator` — r^(2/3)*sin(2theta/3) singularity for AMR benchmarking

- **Domain Geometry Abstractions** (`src/pde/geometry.py`)
  - `RectangularDomain`, `LShapedDomain`, `CylinderFlowDomain` (DFG benchmark)
  - Rejection sampling for non-convex domains, proportional boundary sampling
  - `GeometryConfig` Pydantic schema and `create_geometry()` factory

- **Time-Stepping Module** (`src/pde/time_stepping.py`)
  - `ForwardEuler`, `RK4`, `CrankNicolson` (fixed-point iteration) with factory pattern
  - `TimeSteppingConfig` Pydantic schema, `integrate()` with snapshot saving

- **S500 Swarm Planning Game** (`src/pde/games/swarm_planning.py`)
  - `SwarmPlanningGame` with round-robin multi-agent control (7 actions per agent)
  - Potential field obstacle avoidance (Laplace equation connection), coverage rewards
  - `SwarmPlanningConfig` — fully Pydantic-validated with no hardcoded values

- **PettingZoo Adapter** (`src/games/pettingzoo_adapter.py`)
  - `PettingZooAdapter` wrapping `GameInterface` as PettingZoo `ParallelEnv`
  - Optional dependency with graceful degradation (`HAS_PETTINGZOO` flag)

- **Unified Loss Package** (`src/training/losses/`)
  - `LossRegistry` with decorator-based registration (`"alphagalerkin"`, `"l2_relative"`, `"h1"`, `"mse"`)
  - `get_loss()` factory function for config-driven loss instantiation
  - Backwards-compatible thin wrappers in `src/training/loss.py` and `src/training/physics_loss.py`

- **BaseTrainer Consolidation** (`src/training/base_trainer.py`)
  - Abstract `BaseTrainer[ConfigT]` with shared AMP, gradient clipping, LR scheduling, checkpoint save/load
  - `BaseTrainerConfig` Pydantic schema covering all shared hyperparameters
  - `StepResult` dataclass for structured step output

- **Checkpoint Migration System** (`src/training/checkpoint_migration.py`)
  - Version-aware migration with `@register_migration` decorator
  - Migration path: `0.0.0 -> 1.0.0 -> 1.1.0` (LBB config fields added)

- **Property-Based and Numerical Stability Tests**
  - `tests/training/test_loss_properties.py` — hypothesis tests: non-negativity, CE = log(n), gradient flow
  - `tests/training/test_numerical_stability.py` — extreme values, near-zero denominators, NaN propagation
  - `tests/pde/test_operator_properties.py` — PDE operator invariants, linearity, collocation in domain
  - `tests/modeling/test_attention_properties.py` — Galerkin attention shape, LBB positivity, resolution independence

- **Comprehensive Coverage Tests** (218 new tests)
  - `tests/pde/test_geometry.py` — 65 tests for domain geometries
  - `tests/pde/test_time_stepping.py` — 37 tests for time-stepping methods
  - `tests/research/test_baselines.py` — 39 tests for classical solver baselines
  - `tests/research/test_pde_benchmarks.py` — 38 tests for benchmark runner
  - `tests/training/test_base_trainer.py` — 39 tests for BaseTrainer
  - `tests/pde/test_swarm_planning.py` — 50 tests for swarm planning game
  - `tests/games/test_pettingzoo_adapter.py` — 11 tests for PettingZoo adapter

### Changed

- **CI/CD Hardening** (`.github/workflows/ci.yml`)
  - MyPy strict enforcement (`continue-on-error: false`)
  - Coverage gates raised: 75% -> 85% overall, 80% -> 85% per-module (pde, modeling, training)
  - Added `research` module coverage gate at 85%
  - Added nightly schedule (`cron: '0 4 * * *'`) and performance benchmark job on main merges

- **Config-Driven LBB Loss** (`config/schemas.py`)
  - Surfaced `lbb_loss_weight`, `lbb_target`, `lbb_eps`, `log_barrier_weight` as Pydantic fields
  - Added mathematical documentation (Babuska-Brezzi motivation) in field descriptions

- **Race Condition Fix** (`src/modeling/model.py`)
  - Removed `_training_resolution` mutation from `forward()` (DDP-unsafe)
  - Added explicit `set_training_resolution()` public method

### Fixed

- `advection_coeff` dimension mismatch in `PDEBenchmarkRunner._create_operator()` — was hardcoded `[0.0, 0.0]` for any dim

- **Chess Self-Play Training Pipeline** (AlphaZero methodology)
  - `ActionPolicyHead` for dense 4672-action policy output (`src/modeling/model.py`)
  - `StatefulGameWrapper` bridging stateless `GameInterface` to MCTS (`src/games/wrapper.py`)
  - Chess training CLI (`scripts/train_chess.py`) with Hydra config (`config/train_chess.yaml`)
  - `game_type` and `action_space_size` fields in `OperatorConfig` (`config/schemas.py`)
  - PRD and ADR documentation (`docs/prd/prd-chess-self-play.md`, `docs/architecture/ADR-chess-self-play.md`)

- **Chess Training Tests**
  - `tests/games/test_wrapper.py` — StatefulGameWrapper unit tests (10 tests)
  - `tests/modeling/test_chess_model.py` — ActionPolicyHead and chess model tests (12 tests)
  - `tests/training/test_chess_self_play.py` — Chess self-play integration tests (7 tests)
  - `tests/games/test_chess_exhaustive.py` — Exhaustive encode/decode roundtrip + edge cases (20 tests)
  - `tests/training/test_trainer_chess.py` — Checkpoint save/load/resume, engine eval, config tests (11 tests)
  - `tests/security/test_chess_security.py` — Invalid actions, OOB states, corrupted data (15 tests)
  - `tests/e2e/test_chess_training_e2e.py` — E2E training smoke tests (3 tests)

- **Stockfish Benchmark Evaluation**
  - Engine eval config fields in `TrainingConfig` (path, depth, games, movetime)
  - `Trainer._run_engine_evaluation()` with W&B Elo metric logging
  - Engine eval section in `config/train_chess.yaml`

- **CI/CD Chess Pipeline**
  - Stage 8: Chess Pipeline Tests in `.github/workflows/ci.yml`
  - Coverage gate `--cov-fail-under=80` for `chess.py` (97%) and `wrapper.py` (100%)
  - CI Success gate requires chess tests
### Changed

- **Game-agnostic self-play**: `SelfPlayWorker` now accepts optional `GameInterface` parameter
- **Game-agnostic trainer**: `Trainer.__init__()` accepts `game` parameter, forwarded to worker
- **Game-agnostic collator**: `VariableSizeCollator` and `SameSizeCollator` derive action mask size from `target_policy` tensor instead of hardcoded `board_size²+1`
- `AlphaGalerkinModel` and `AlphaGalerkinFast` auto-select policy head by `action_space_size`
### Fixed

- **Underpromotion encode/decode mismatch** (`src/games/chess.py`): `_decode_move` used `[-1, 0, 1]` but `_encode_move` used `straight=0, left=1, right=2` — straight promotion from column 0 decoded as `to_col=-1`. Fixed to `[0, -1, 1]`.
- **Collator action mask size** (`src/data/collate.py`): Both collators hardcoded `n_actions = board_size²+1` causing tensor size mismatch with chess's 4672-action policy. Fixed to detect per-experience policy encoding.

## [0.2.0] - 2026-01-26

### Milestones Achieved

- **Zero-Shot Transfer Validated**: Physics PoC demonstrated resolution-independence
  - Trained on 9x9 grids, achieved MSE 0.000209 on 19x19 grids (240x better than 0.05 threshold)
  - Validates core Galerkin approach for continuous operator learning

- **Training Pipeline Operational**: End-to-end training with self-play working on GPU
  - MCTS-based self-play generates training experiences
  - LBB stability monitoring integrated into training loop

### Added

- **W&B Integration for Physics PoC**
  - `--wandb` flag for `train_physics.py` to enable Weights & Biases logging
  - Logs training loss, evaluation MSE, transfer MSE, and learning rate
  - Final summary includes success status and best transfer MSE

- **GameInterface Protocol Implementation**
  - Added `apply_action()` method to `SimpleGoGame` class
  - Enables MCTS integration with Go game state

- **Security Tests** (`tests/security/`)
  - Input sanitization tests for GTP interface
  - DoS protection via input length limits

- **E2E Tests** (`tests/e2e/`)
  - CLI journey tests for help and train commands

### Changed

- Replaced Unicode checkmarks with ASCII `[PASS]`/`[FAIL]` for Windows compatibility
- Updated `.gitignore` with additional patterns:
  - `nul` (Windows device file)
  - `*.log`, `*.dist-info/`
  - `hydra_outputs/`

### Fixed

- Fixed `AttributeError: 'SimpleGoGame' object has no attribute 'apply_action'`
- Fixed unused loop variable warning in `BoardSizeBatchSampler`
- Fixed line length issue in W&B initialization

## [0.1.0] - 2026-01-26

### Added

- **Core Architecture**
  - `AlphaGalerkinModel`: Resolution-independent Go AI using continuous operators
  - `GalerkinLinearAttention`: O(N) complexity global influence modeling
  - `SoftmaxAttention`: Local tactical reading with injectivity preservation
  - `FNetBlock`: FFT-based mixing for fast MCTS rollouts

- **Mathematical Kernel**
  - Fredholm integral equation with Green's function formulation
  - Fourier features for positional encoding
  - Monte Carlo integral normalization (1/n) for Galerkin attention
  - LBB stability monitoring (dim(Key) >= dim(Query))

- **Training Infrastructure**
  - Self-play with MCTS for experience generation
  - Uniform and prioritized replay buffers
  - `AlphaGalerkinLoss`: policy_CE + value_MSE + LBB_regularization
  - Checkpoint management with best model tracking
  - Hydra configuration system

- **Physics PoC**
  - Poisson equation solver for synthetic data generation
  - `PhysicsOperator` neural network for influence field prediction
  - Zero-shot transfer verification scripts

- **PoC Scenario Framework**
  - Configuration-driven scenario execution
  - Built-in scenarios: transfer, complexity, stability
  - Pydantic-validated configs
  - Structured logging via structlog

### Documentation

- C4 architecture diagrams
- CLAUDE.md with project context and verification commands
