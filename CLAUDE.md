# CLAUDE.md - AlphaGalerkin Context

## Project Overview
AlphaGalerkin is a resolution-independent Go AI that uses Continuous Operator Learning
(Galerkin Transformers & FNet) instead of discrete CNNs, enabling zero-shot transfer
between board sizes (e.g., 9x9 to 19x19) and accelerating MCTS rollouts via FFT mixing.

## Mathematical Decisions
- [2026-01-26]: Chosen Kernel: Fredholm integral equation with Green's function formulation.
- [2026-01-26]: Basis function selection: Fourier Features for positional encoding.
- [2026-01-26]: Normalization scheme: Monte Carlo integral normalization (1/n) for Galerkin attention.
- [2026-01-26]: LBB Stability: dim(Key) >= dim(Query) to satisfy inf-sup condition.

## Architecture Decisions
- [2026-01-26]: Strategy Body uses GalerkinLinearAttention for O(N) global influence modeling.
- [2026-01-26]: Tactical Head uses SoftmaxAttention to preserve injectivity for local reading.
- [2026-01-26]: FNet mixing uses real-valued FFT (torch.fft.rfft2) for efficiency.
- [2026-01-26]: All tensor operations use einops for dimension clarity.

## Key Mathematical Operators

### GalerkinAttention
Implements Petrov-Galerkin projection with O(N) complexity:
- Projects values onto Key basis: K^T V (Monte Carlo integral)
- Reconstructs in Query basis: Q * Context
- Normalization: 1/n (not 1/sqrt(d))

### FNetBlock
FFT-based mixing for high-speed rollouts:
- FFT2D -> Spectral Mixing -> iFFT2D
- Enables batch MCTS leaf evaluation

### StabilityGuard
Monitors LBB condition during training:
- Computes singular values of Key-to-Value projection
- Ensures sigma_min > beta > 0

## Training Infrastructure
- [2026-01-26]: Added complete training pipeline with self-play, replay buffer, and trainer.
- [2026-01-26]: Loss = policy_CE + value_MSE + lbb_regularization for Galerkin stability.
- [2026-01-26]: Replay buffer supports uniform and prioritized experience replay.
- [2026-01-26]: Variable board size batching via padding and masking.
- [2026-01-26]: Checkpoint manager with best model tracking and rotation.

## Physics PoC (Supervised Learning Validation)
- [2026-01-26]: Added Poisson equation solver for synthetic data generation.
- [2026-01-26]: PhysicsOperator neural network for influence field prediction.
- [2026-01-26]: Zero-shot transfer validation: Train on 9x9 → Evaluate on 19x19.
- [2026-01-26]: Success criterion: MSE < 0.05 on 19x19 without retraining.
- [2026-01-26]: **MILESTONE ACHIEVED**: Zero-shot transfer MSE = 0.000209 (240x better than threshold)
- [2026-01-26]: Added W&B integration for experiment tracking (--wandb flag).

## PoC Scenario Framework
- [2026-01-26]: Added configuration-driven PoC scenario framework (src/poc/).
- [2026-01-26]: Three built-in scenarios: transfer, complexity, stability.
- [2026-01-26]: Pydantic-validated configs with no hardcoded values.
- [2026-01-26]: Structured logging via structlog throughout.
- [2026-01-26]: C4 architecture documentation in docs/architecture/c4_model.md.
- [2026-01-26]: Added comprehensive C4 architecture in Mermaid format (docs/architecture/c4_mermaid.md).

## Milestones
- [2026-01-26]: **Zero-Shot Transfer Validated** - Physics PoC achieved MSE 0.000209 on 19x19 (trained on 9x9)
- [2026-01-26]: **Training Pipeline Operational** - End-to-end GPU training with MCTS self-play working
- [2026-02-01]: **CI/CD Pipeline Added** - GitHub Actions workflow with lint, type check, tests, coverage
- [2026-02-01]: **Video Compression Hyperprior Fixed** - Proper z_bitstream encoding/decoding for entropy model
- [2026-02-01]: **Chess Game Implementation** - Full Chess rules with AlphaZero-style encoding (119 planes)
- [2026-02-04]: **P0 Critical Fixes** - Emergency checkpoint, GTP player assignment, parallel self-play
- [2026-02-04]: **PDE-MCTS Integration** - PDEGameAdapter bridges PDE games to MCTS search engine
- [2026-02-04]: **Physics Loss Wired** - Laplacian regularization via autodiff in PhysicsLoss
- [2026-02-04]: **Curriculum Config Schema** - curriculum_schedule field on TrainingConfig with transition logging
- [2026-03-30]: **CI Hardening** - MyPy strict enforcement, coverage gates raised to 75%/80%, nightly schedule
- [2026-03-30]: **Config-Driven LBB Loss** - Magic numbers surfaced as Pydantic fields with mathematical docs
- [2026-03-30]: **Race Condition Fixed** - _training_resolution mutation removed from forward(), explicit setter added
- [2026-03-30]: **Checkpoint Migration** - Version-aware migration system (0.0.0→1.0.0→1.1.0) with registry pattern
- [2026-03-30]: **Loss Package Unification** - src/training/losses/ with registry, backwards-compatible imports
- [2026-03-30]: **NavierStokesOperator** - Taylor-Green vortex benchmark with exact analytical solution
- [2026-03-30]: **BurgersOperator Enhanced** - Cole-Hopf exact solution, configurable shock params, convergence rates
- [2026-03-30]: **Domain Geometry Abstractions** - RectangularDomain, LShapedDomain, CylinderFlowDomain
- [2026-03-30]: **L-Shaped Poisson Operator** - r^(2/3)*sin(2θ/3) singularity for AMR benchmarking
- [2026-03-30]: **Time-Stepping Module** - Forward Euler, RK4, Crank-Nicolson with factory pattern
- [2026-03-30]: **SBIR Proposal Infrastructure** - Navy N252-088, AFWERX, DOE ASCR, NSF configs + benchmark suite
- [2026-03-30]: **IP Strategy Documented** - 3 provisional patent claims, publication plan, dual-licensing
- [2026-03-30]: **Property-Based Tests** - Hypothesis tests for loss, PDE operators, attention mechanisms
- [2026-03-30]: **Numerical Stability Tests** - Edge cases, extreme values, mixed precision, NaN propagation
- [2026-04-02]: **Physics Loss Fully Wired** - CombinedAlphaGalerkinPhysicsLoss passes lbb_constant, action_mask, model to trainer
- [2026-04-02]: **2D AMR Baseline** - DorflerAMRSolver extended to 2D with element-wise refinement and Dorfler marking
- [2026-04-02]: **Navier-Stokes FDM Solver** - Chorin projection method baseline for Taylor-Green vortex benchmark
- [2026-04-02]: **PDE GameInterface Bridge** - PDEGameInterface wraps PDEGame for GameRegistry registration
- [2026-04-02]: **PDE Games Registered** - pde_basis and pde_mesh registered in GameRegistry via src/pde/register_games.py
- [2026-04-02]: **PDE Training Config** - config/train_pde.yaml for MCTS-guided basis selection training
- [2026-04-02]: **ROI Implementation Plan** - Tiered next-steps plan in docs/ROI_IMPLEMENTATION_PLAN.md
- [2026-04-07]: **Physics Loss Tests** - 52 comprehensive tests for physics-informed training (config toggle, gradient flow, property-based)
- [2026-04-07]: **SBIR Benchmark Demo** - End-to-end sbir_demo.py with HTML/JSON/Markdown report generation
- [2026-04-07]: **Loss Balancing Audit** - Fixed NaN/Inf propagation bugs in ReLoBRaLo/SoftAdapt, 96 property-based tests
- [2026-04-07]: **PDE-MCTS Self-Play Wired** - PDE games auto-register, create_trainer() accepts game parameter, 40 tests
- [2026-04-07]: **Visualization Module** - PlotRegistry with 5 plot types, HTMLReportGenerator with themed templates
- [2026-04-07]: **Coverage Expansion** - 390+ new tests across training, PDE, games, curriculum, modeling modules
- [2026-04-07]: **BaseTrainer Refactor** - Extracted shared AMP, grad clip, LR scheduling into BaseTrainer base class
- [2026-04-07]: **Distributed Trainer Tests** - 35 new tests for DistributedTrainer (metrics, checkpoints, multi-process)
- [2026-04-07]: **Test Speed Fixes** - Mocked MCTS self-play in all trainer tests to prevent hanging (chess, physics, pipeline)
- [2026-04-07]: **Coverage Sprint** - 115 new tests: statistics significance (52), tuning sampler/tuner (33), ONNX integration (30)
- [2026-04-07]: **GPU Skip Hook** - Root conftest.py auto-skips gpu_required tests when CUDA unavailable; 0 spurious failures
- [2026-04-07]: **Gumbel MCTS Search Tests** - 38 integration tests for search(), _sequential_halving(), _simulate(), get_improved_policy(), factory
- [2026-04-25]: **Leap 71 / PicoGK Integration v1** - SDFEvaluator Protocol + AnalyticalHelixSDF (closed-form helical-tube SDF), PicoGKDomain(DomainGeometry) with Newton-projected boundary sampling, HelicalHeatOperator on the LShapedPoissonOperator override pattern, voxel-FDM reference solver, NoyronHXScenario for zero-shot 3D heat-transfer transfer (GPU-preferred device handling), 48 new tests covering SDF, domain adapter, operator, and scenario smoke. The PicoGK .NET dependency is gated behind the optional `[picogk]` extra; CI runs entirely on the analytical helix surrogate.
- [2026-04-25]: **PhysicsOperator 3D-Aware** - FourierBasis/FourierFeatures parameterized by `input_dim` (default 2 preserves all existing 2D callers); PhysicsOperator gained an `input_dim` constructor arg. Enables the Noyron HX 3D scenario without touching any 2D code path.
- [2026-04-25]: **Leap 71 / PicoGK Integration v2** - Added `HelicalStokesOperator` (steady incompressible Stokes flow on a helical SDF — Noyron RP coolant channels, v2.3 expansion), `HelicalMagnetostaticsOperator` (vector-potential magnetostatics — Noyron EA actuators, v3.1 expansion), and `HelicalBasisSelectionInterface` (MCTS basis selection on any helical operator — v2.2 expansion). All three operators registered in `PDEOperatorRegistry` and wired through the `pde_basis_helical` game. 35 new tests; 100% coverage on the new flow/EM operators. Module-level constants (Newton iters, gradient epsilon, oversample max, voxel-FDM iter cadence, harmonic wave number) surfaced as constructor / Pydantic fields for tunability. Shared `src/poc/device.py` helper extracted with explicit GPU-preferred / CPU-fallback / fail-loud semantics. `tests/pde/conftest.py` adds reusable helix-param fixtures.
- [2026-04-27]: **Noyron HX v1 Hardening** - Closed the gap-report deviations on the Leap 71 v1 milestone: (1) `voxel_fdm` mode now trains on the cached FDM solution itself (zero source, operator-Dirichlet boundary) instead of the harmonic surrogate, so the headline `mse_low < 5e-4` / `mse_high < 1e-3` thresholds are reachable in FDM mode; (2) `accept_rate`, `train_time_s`, and `eval_time_s` are recorded in `ScenarioResult.metrics`; (3) `helix_n_turns` default aligned across `NoyronHXScenarioConfig`, the YAML scenario, and `AnalyticalHelixSDF` (now 5); (4) bisection / grid-search fallback wired into both `AnalyticalHelixSDF._nearest_t` and `PicoGKDomain._project_to_surface`, opt-out via `enable_fallback` / `enable_bisection_fallback`. `PicoGKDomain.volume_accept_rate` exposes the MC rejection-rate without recomputation. 27 new tests across SDF, domain adapter, and scenario; 0 mypy regressions on the changed surface.
- [2026-04-27]: **Noyron HX Tech-Debt Scrub** - Hardcoded numerical-stability literals surfaced as named module constants (`DEFAULT_TRANSFER_RATIO_FLOOR=1e-12`, `DEFAULT_NORMALIZE_EXTENT_FLOOR=1e-9`, `EVAL_SEED_STRIDE=9973`); duplicated `randperm`/`randint` voxel-pool sampling between `_sample_voxel_fdm_batch` and `_evaluate` extracted into a single `_draw_pool_indices` helper with explicit input validation. Per-module coverage on the Leap 71 v1 hardening surface: `sdf.py` 100%, `geometry_picogk.py` 100%, `config_noyron.py` 100%, `noyron_hx.py` 97% (16 additional tests for SDF / domain validators + `_draw_pool_indices` semantics). Noyron HX scenario added to the `Regression Surface` table and the C4 PoC-framework component diagram.
- [2026-04-27]: **Self-Hosted Transcoder Phase 0** — `src/video_compression/perf/` benchmark harness landed: GPU-primary (default `device_preference="cuda"`), per-profile `cuda:N` pinning so a single sweep covers the reference dual-GPU rig (RTX 5060 Ti 16 GB at `cuda:0` + RTX 5060 8 GB at `cuda:1`), Pydantic-validated `PerfBenchmarkConfig` with no hardcoded values (resolution / batch / runtime / phase / warmup / repeats / tolerance all surfaced as fields), `BaselineRegistry` with explicit JSON schema versioning (`PERF_BASELINE_DOCUMENT_SCHEMA_VERSION`), unversioned-file migration, and `extra="ignore"` forward-compat. Three YAML configs shipped (`config/perf/smoke.yaml` for CPU CI, `config/perf/cuda0_headline.yaml` for the 16 GB primary, `config/perf/default.yaml` for the dual-card sweep). 115 unit + integration tests pass, 12 GPU-only tests auto-skip when CUDA is unavailable via `@pytest.mark.gpu_required`. CLI `scripts/benchmark_codec.py` exposes `run` / `record-baseline` / `diff` subcommands with structured `structlog` events bound to `benchmark_id` and `cell_key`.
- [2026-04-30]: **Self-Hosted Transcoder Phase 1** — Four decoder runtime backends in `src/video_compression/runtime/`: `pytorch-eager` (baseline), `pytorch-compiled` (`torch.compile` with inductor, CUDA graphs via `reduce-overhead`), `onnx-cuda` (in-memory ONNX export + `CUDAExecutionProvider`), `tensorrt` (`torch_tensorrt.compile` Dynamo IR for max throughput). Registry-based discovery via `@register_runtime` decorator; `DecoderRuntime` Protocol + `DecoderRuntimeContext` dataclass; `CompiledArtifactMetadata` provenance tracking with `extra_tags`. FP32/FP16/BF16 precision dispatch via `_dtype_for_precision()` + `_runtime_name_for_profile()` in the benchmark loop (`NotImplementedError` gates removed). CUDA environment configured: PyTorch 2.11.0+cu126 + torch_tensorrt 2.11.0+cu126 on GTX 1660 Ti. Full regression: 244 passed, 17 skipped, 0 failed. `ruff` clean, `mypy --strict` clean.
- [2026-05-01]: **Self-Hosted Transcoder Phase 2-B** — `src/video_compression/zoo/` model-zoo subpackage for the 8-point R-D Lagrangian sweep across the dual-GPU rig. Pydantic schemas (`ModelZooEntryConfig` / `ModelZooManifestConfig` / `OptimizerConfig` / `SchedulerConfig`) with zero hardcoded values; JSON+YAML manifest persistence dispatched by file suffix; forward-compat migration via `_migrate_manifest_document` with explicit `PERF_ZOO_MANIFEST_SCHEMA_VERSION=1` and `PERF_ZOO_ENTRY_SCHEMA_VERSION=1`; `scan_devices()` + `assign_devices()` with four strategies (`VRAM_AWARE` best-fit pack / `ROUND_ROBIN` / `SINGLE_DEVICE` / `MANUAL` per-entry pin); `VideoCodecZoo` filesystem registry with GCS-backend stub gated for Phase D. Shipped `config/video_compression/zoo/lambda_grid.yaml` (λ ∈ {0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045, 0.09, 0.18}) loads through the same path as user manifests. **100% line + branch coverage** on all five zoo modules (68 tests including Hypothesis property-based migration tests). E2E validated on live `cuda:0=RTX 5060 Ti 16 GiB` + `cuda:1=RTX 5060 8 GiB`. `ruff` clean, `mypy --strict` clean.
- [2026-05-01]: **Self-Hosted Transcoder Phase 2-D** — Manifest-level sweep orchestrator (`src/video_compression/zoo/sweep.py`) with two slices stacked on Phase 2-B/C. **Slice A**: `scripts/train_compression_zoo.py` adds `dry-run` / `train` subcommands operating on a manifest; shared CLI primitives extracted into `src/video_compression/zoo/cli_helpers.py` and re-exported as `_underscored` aliases from `train_compression_zoo_entry.py` for back-compat. **Slice B**: `ZooSweep.run_parallel()` dispatches one worker thread per device (same-device entries serialized inside their worker); `make_subprocess_entry_runner(...)` returns an `EntryRunner` that re-invokes the single-entry CLI with `CUDA_VISIBLE_DEVICES=<idx>` pinning and translates the child's `--device` to `cuda:0`, then reads `metrics.json` + checkpoint back to reconstruct a `ZooTrainingReport`. `ZooTrainer` now persists `train_wallclock_s` / `eval_wallclock_s` so the subprocess runner can reconstruct them across process boundaries. 35 new tests (11 sweep + 9 multi-entry CLI + 15 parallel/subprocess); 140-test full zoo+scripts+training regression passes. `ruff` clean, `mypy --strict` clean.
- [2026-05-03]: **SBIR P40 Benchmark Hardening** — closed every gap surfaced by the post-run report (NS-FDM L2 ≈ 0.5 floor, Dörfler AMR stuck at 18 DOF on Burgers, no GPU telemetry, hard-coded CPU PINN, no extreme-resolution Poisson level). (1) Single-line numpy/torch parity fix at `src/pde/operators.py:1189` corrects the FDM IC, FDM L2 reference, and PINN L2 evaluation simultaneously. (2) `AMRConfig` defaults raised (`marking_fraction` 0.3→0.5, `max_refinements` 10→30, `max_initial_points_1d` 8→256, `initial_dof_divisor` 4→2) plus target-aware `n_start = max(min(n_dof//2, max_initial_points_1d), min_initial_points)` lets 1D Burgers refinement scale with the request instead of saturating at 18. (3) `PINNConfig` gains `device: str = "auto"` (GPU-preferred) and `vector_pde: bool | None = None`; `SimplePINNSolver.solve` honours both, builds 2-channel networks for NS, wraps the training loop in `GpuUtilizationProfiler`, and emits `device` / `vector_pde` / `gpu_profile` in metadata. (4) New `src/research/gpu_profiler.py` wraps `nvidia-smi dmon` as a context manager (mean SM-util %, mean memory-util %, peak FB-MiB); column indices and termination timeout surfaced as named constants. (5) `PDEBenchmarkRunner(heavy=False)` opt-in pipes `heavy_refinement_levels` from the YAML through `--heavy` so the 65 536-DOF Poisson level is gated off CI. (6) `scripts/run_sbir_p40.py` rewritten as a config-driven argparse driver loading `config/benchmarks/sbir_p40.yaml` with PINN profiles for `p40` / `cpu` rows; CLI flags for every override (`--config`, `--output-dir`, `--device`, `--n-epochs`, `--n-collocation`, `--refinement-levels`, `--skip-cpu`, `--require-cuda`); helpers individually unit-tested. **Coverage**: `gpu_profiler.py` 96%, `baselines.py` 95%, `pde_benchmarks.py` 94%, `poc/device.py` 100%; 1131 tests pass; `ruff` + `ruff format` clean.
- [2026-05-12]: **LLM-Prior MCTS Basis-Selection Integration (LM Studio + Qwen-14B)** — new `src/integrations/` namespace plus `llm_prior_ablation` PoC scenario benchmarking three MCTS evaluators (`RandomEvaluator`, `FNetEvaluator`, new `LMStudioEvaluator`) on Poisson (ID) and Burgers (OOD). `src/integrations/lm_studio/` ships `LMStudioConfig` (Pydantic; every knob — base_url, model, timeout_ms, max_retries, backoff_base_s, temperature, max_tokens, fallback_to_uniform_on_parse_error, min_free_vram_gib, preflight_on_construct, enabled — surfaced as a typed field), `LMStudioPolicyResponse` + typed exception hierarchy (`LMStudioError` → `LMStudioParseError` / `LMStudioActionSpaceMismatchError` / `LMStudioConnectionError` / `LMStudioPreflightError`), `LMStudioClient` (synchronous `openai`-SDK wrapper using `response_format={"type":"json_object"}` and `seed=...`, bounded exponential-backoff retries with corrective user-turn on action-size mismatch, `lm_studio_call` / `lm_studio_retry` structlog events carrying `prompt_hash`, `latency_ms`, `tokens_in`, `tokens_out`, `parse_ok`, `retries_used`), deterministic `prompt.build_policy_prompt` + sha256-truncated `prompt_hash`, `preflight.check_lm_studio_server` (server reachable + model in `/v1/models` + free-VRAM floor via `torch.cuda.mem_get_info`), and `LMStudioEvaluator` implementing `src/mcts/evaluator.py::Evaluator` structurally with illegal-action `-inf` masking + temperature softmax mirroring `FNetEvaluator._process_policy`. **GPU-only**: scenario `setup()` calls `src.poc.device.resolve_device(config.device, context=...)` which raises if CUDA is unavailable. Arm gating is **graceful** — when LM Studio preflight fails or no trained checkpoint is configured the affected arm is disabled *and* its acceptance thresholds are removed from `self.config.thresholds` so absent metrics don't auto-FAIL the run (`BaseScenario._evaluate_thresholds` only knows `bool`). Per-seed reproducibility via `np.random.seed`/`torch.manual_seed` before each `MCTS(...)` (no `seed` kwarg on `MCTS.__init__`) plus `LMStudioClient.complete_policy(seed=...)`. New optional dep `[lm-studio] = ["openai>=1.40,<2.0"]`; the openai SDK is imported lazily so the base install never pulls it in. CPU CI mocks the SDK via `tests/integrations/conftest.py::FakeOpenAIModule`; GPU smoke tests carry `@pytest.mark.gpu_required` and additionally gate on `LM_STUDIO_URL`. Headline acceptance thresholds shipped (`id_rollout_reduction_pct ≥ 25%`, `ood_llm_residual ≤ 1e-2`, `ood_trained_residual > 1e-1`, `llm_call_p95_latency_ms ≤ 3000` — recalibrated from 300 ms after empirical Qwen-14B Q4 latency review). **Coverage** (post-tech-debt + post-Copilot-review): `src/integrations/lm_studio/*` 91% (line+branch combined: `client.py` 91%, `evaluator.py` 95%, `preflight.py` 97%, `prompt.py` 100%, `config.py`/`schema.py`/`__init__.py` 100%), `llm_prior_ablation.py` 86%, `llm_prior_config.py` 100%; 96 new tests; full project regression green; `ruff` + `ruff format` clean; `mypy --strict` zero new errors on the changed surface.
- [2026-06-12]: **Operationalising the Headline Runs (OpenAI-compatible LLM backends + PoC baseline harness)** — additive, backwards-compatible follow-up to PR #86 that turns the merged scaling-law / research-loop machinery into reproducible, regression-guarded results. **(WS1) Multi-backend LLM**: new `src/integrations/openai_compat/` backend-profile registry (`BackendProfile`, `register_backend`/`get_backend`/`apply_backend_defaults`) lets `LMStudioConfig.backend ∈ {lm_studio, vllm, llama_cpp}` auto-fill each server's endpoint/model/`vram_check_mode` for fields left unset (explicit YAML always wins; `lm_studio` profile == historical defaults so it is a guaranteed no-op). Two additive `LMStudioConfig` fields (`backend`, `vram_check_mode`); preflight skips the *local* free-VRAM probe when `vram_check_mode="off"` (remote server). All existing `LMStudio*` imports/configs unchanged. **(WS2) Baseline harness**: `src/poc/baselines/` mirrors the perf `BaselineRegistry` — schema-versioned (`POC_BASELINE_*_SCHEMA_VERSION`, `extra="ignore"`, `migrate_baseline_document`) record/load/save + direction-aware `compare` (per-entry higher/lower-better + tolerance, so one code path gates residual / `solved_fraction` / latency). New poc-CLI `record-baseline` / `diff` subcommands (diff exits non-zero on regression → CI-usable); research-loop result now persisted to `outputs/agents/research/<run_id>/result.json` via `agents.cli research --output-dir`. **(WS3) DRY**: duplicated `_median` extracted to `_centaur_common.median_of` (back-compat `_median` aliases kept). **(WS4) Docs**: tested-server matrix in `src/integrations/AGENT.md`; backend-switch + baseline sections in `docs/PR86_HEADLINE_RUNS.md`. **Coverage**: `openai_compat` 94%, `poc/baselines` 100%; 60+ new tests (backend registry 19, baseline harness 31, poc-CLI 7, research persistence 3) incl. Hypothesis migration property test; full integrations/poc/agents regression green; `ruff` + `ruff format` clean; `mypy --strict` zero new errors on the changed surface.
- [2026-06-07]: **AI-for-Physics Scaling Themes (OOD operators + scaling law + centaur research loop)** — three additive, backwards-compatible deliverables operationalising Adam Brown's "held-out generalisation / bitter-lesson scaling / billions of cloneable Einsteins" themes. (1) **OOD operators**: `HelmholtzOperator` (∇²u + k²u = f; oscillatory zeroth-order term, wavenumber resolves arg → `reaction_coeff` → `DEFAULT_HELMHOLTZ_WAVENUMBER`) and `BiharmonicOperator` (∇⁴u = f; fourth-order via two autodiff Laplacian passes) with manufactured `∏ sin(π x_d)` solutions, registered in `PDEOperatorRegistry` and added to `PDEType` + the `llm_prior_ablation` `ood_pde` Literal — held-out residual structures the FNet evaluator never trained on. (2) **Shared centaur primitives**: `src/poc/scenarios/_centaur_common.py` extracts the canonical `PDE_TYPE_MAP`, operator/game construction, arm evaluator factory (`build_arm_evaluator`), and the inner MCTS rollout (`run_basis_selection_cell`); `llm_prior_ablation` refactored to delegate (regression suite stays green). (3) **Scaling-law scenario**: `ScalingLawScenario` (`@scenario("scaling_law")`) sweeps MCTS-simulation budget and fits a log-log residual curve per arm (`residual_scaling_exponent`/`residual_fit_r2` thresholds), with a `StatisticalAnalyzer` arm-vs-arm comparison and HTML report. (4) **Centaur research-loop harness** extends `src/agents/` with `AgentType.RESEARCH`, `ResearchLoopConfig`/`ResearchProblemSpec`, and `ResearchLoopOrchestrator(BaseExecutable)` that sweeps MCTS+evaluator across a problem manifest (sequential or one-thread-per-problem) and aggregates a per-problem discovery ledger (`solved_fraction`, `arm_wins_*`); exposed via the agents CLI `research` subcommand. Every knob is a typed Pydantic field (no hardcoded budgets/tolerances/wavenumbers); arm gating mirrors `llm_prior_ablation` (preflight/checkpoint). Demo configs: `config/scenarios/scaling_law_demo.yaml`, `config/agents/research_loop_demo.yaml`. **Coverage**: `_centaur_common.py` 89%, `scaling_law.py` 86%, `scaling_law_config.py` 98%, `agents/config.py` 98%, `agents/research_loop.py` 86%; 100+ new tests (OOD operators 20, scaling-law 35, centaur-common 12, research-loop 16) plus 4 `gpu_required` real-server smoke tests; full pde/poc/agents/integrations/mcts regression green (1780 passed); `ruff` + `ruff format` clean; `mypy --strict` zero new errors on the changed surface.
- [2026-06-22]: **Game-Review Real Evaluator + Capture-Correct Reconstruction (Phase 1 of unfinished-feature roadmap)** — closed two latent defects in `src/analysis` that a peer review surfaced (the original gap report misdiagnosed both). (1) `GameReviewer`/`create_game_reviewer` now accept an injected `model_evaluator` and, when `AnalysisConfig.model_checkpoint_path` is set, auto-wire a checkpoint-backed evaluator via the new `src/analysis/go_adapter.py::build_checkpoint_model_evaluator` (→ `create_model_from_checkpoint` + `FNetEvaluator`, device resolved through `src/poc/device.py::resolve_device`); previously **no** model was ever attached, so every position fell through to the uniform `win_rate=0.5` dummy. The no-model case now logs `review_no_model_evaluator` instead of being silent, and a checkpoint load failure degrades gracefully (`review_checkpoint_load_failed` warning → dummy fallback). (2) `GameReviewer._create_board_state` delegates to `go_adapter.reconstruct_board`, replaying moves through the real `GoGame` engine so **captures and ko are honoured** (the prior naive replay left captured stones on the board); coordinate mapping (`(x,y)` ↔ `row*size+col`) is centralised in `move_to_action`/`action_to_move` and round-trip tested. New config fields `model_checkpoint_path`/`device`/`evaluator_temperature` (all typed, defaulted, no hardcoded values). **Coverage**: `src/analysis` 91% overall (`go_adapter.py` 93%, `reviewer.py` 82%, `config.py` 98%) — new dedicated `--cov-fail-under=85` CI gate added; 162 analysis tests pass (16 new across capture removal, coordinate round-trip via Hypothesis, model-signal wiring, checkpoint fallback); `ruff` + `ruff format` clean; `mypy --strict` clean on the changed surface.

## Next Steps (post-PR #58)

The Leap 71 v1 demo is now production-grade for the headline scenario; the
roadmap below is gated on user demand and reviewer signal, not on additional
hardening of the HX path.

| Phase | Scope | Notes |
|---|---|---|
| **v2.1** | Octree-on-SDF AMR via `MeshRefinementGame` | Currently incompatible (structured grid only); needs an octree backend on top of `PicoGKDomain` |
| **v2.2** | Plug `BasisSelectionGame` into `HelicalHeatOperator` | First MCTS-on-Noyron result; `HelicalBasisSelectionInterface` (v2 milestone) already provides the plumbing |
| **v2.3** | Noyron RP — `NavierStokesOperator` on a copper-nozzle SDF | `HelicalStokesOperator` is the linear stepping stone (v2 milestone); needs convective term + nozzle SDF |
| **v3.1** | Noyron EA — `MagnetostaticsOperator` for actuators | `HelicalMagnetostaticsOperator` (v2 milestone) is in place; needs full vector-potential coupling |
| **v3.2** | Closed-loop Noyron coupling — surrogate-in-the-loop parametric search | Combine all three operators with a parametric outer-loop (e.g. coil pitch optimization) |
| **PicoGK STL ingestion** | Replace `AnalyticalHelixSDF` surrogate with the real Leap 71 STL via the `[picogk]` extra | `PicoGKSDFEvaluator` constructor stub already raises `NotImplementedError` with a clear message; needs voxel-grid loading + bbox extraction |
| **GPU headline run** | Run `python -m src.poc.cli run --config config/scenarios/noyron_hx.yaml` on a CUDA box and capture the headline `mse_low` / `mse_high` / `transfer_ratio` numbers | CI runs the CPU smoke test only; the headline GPU run is a manual reviewer step gated by hardware availability |
| **SBIR P40 GPU rerun** | Run `python -u -m scripts.run_sbir_p40` on the Tesla P40 with the corrected NS-FDM baseline and the 2000-epoch GPU PINN; verify `mean_sm_util_pct` is populated in `outputs/sbir_p40/results.json` for every `pinn_p40` row | Validates the bug fixes against real GPU hardware; CI runs only the helper-function unit tests because PyTorch's stock wheels don't ship sm_61 kernels for the P40 |
| **SBIR demo `--heavy` rerun** | Run `python -m scripts.run_sbir_demo --heavy --output-dir outputs/sbir_demo_v2` to capture the 65 536-DOF Poisson L-shaped row demonstrating the P40's 24 GiB VRAM advantage | Manual reviewer step; CI default keeps the heavy levels off |
| **LLM-prior MCTS GPU run** | Run `python -m src.poc.cli run --config config/scenarios/llm_prior_demo.yaml` on a CUDA host with LM Studio serving Qwen-14B; verify the four headline metrics (`id_rollout_reduction_pct ≥ 25%`, `ood_llm_residual ≤ 1e-2`, `ood_trained_residual > 1e-1`, `llm_call_p95_latency_ms ≤ 3000`). | CI runs only the mocked-CPU surface (`tests/integrations/`); the headline GPU run is a manual reviewer step gated by `LM_STUDIO_URL` + CUDA. Per-GPU baselines for the latency threshold should be recorded in `outputs/poc/llm_prior_ablation/`. |
| **LLM-prior OOD coverage expansion** | Add `helmholtz` and `biharmonic` operators to `_PDE_TYPE_MAP` in `src/poc/scenarios/llm_prior_ablation.py` and run them as additional OOD families. Compare LLM-prior vs trained on each. | The trained `FNetEvaluator` was never trained on these residual structures; the LLM should retain a meaningful advantage. New operators land in `src/pde/registry.py` first; the scenario then auto-picks them up via the `Literal` enum. |
| **LLM-prior alternative backends** | Add adapters for vLLM and llama.cpp-server alongside LM Studio (all OpenAI-compatible). Single `LMStudioConfig.base_url` already points at the endpoint; verify zero code change beyond a `model` rename. | Validates the `[lm-studio]` extra as the canonical OpenAI-compatible client and not LM-Studio-specific. Document tested-server matrix in `src/integrations/AGENT.md`. |

## SBIR Positioning
- **Verified Novelty Gap**: No published papers combine MCTS with Galerkin methods for PDE/mesh refinement
- **Target Solicitations**: Navy N252-088, DOE ASCR C59-01, NSF SBIR, AFWERX Open Topic
- **TRL Level**: 3-4 (advancing to 5-6 with benchmark demonstrations)
- **Key Differentiators**: Multi-step look-ahead (vs myopic RL), provable convergence (vs PINNs/FNO), no training data needed

## Next-Phase Infrastructure (v2.0)

### Distributed Training (src/distributed/)
- [2026-01-26]: Multi-node training via PyTorch DDP with NCCL backend.
- [2026-01-26]: Gradient synchronization with accumulation and compression support.
- [2026-01-26]: Distributed self-play coordination across nodes.
- [2026-01-26]: Model zoo for checkpoint management and curriculum learning.
- [2026-01-26]: Support for torchrun, SLURM, and custom launchers.

### ONNX Export (src/deployment/)
- [2026-01-26]: PyTorch to ONNX conversion with dynamic shape support.
- [2026-01-26]: Quantization support (dynamic/static) for edge deployment.
- [2026-01-26]: ONNX Runtime inference wrapper with multi-provider support.
- [2026-01-26]: Model validation against PyTorch outputs.

### Multi-Game Support (src/games/)
- [2026-01-26]: Abstract GameInterface for game-agnostic architecture.
- [2026-01-26]: Game registry with decorator-based registration.
- [2026-01-26]: Go implementation with full rules (Chinese scoring, superko).
- [2026-01-26]: 8-fold symmetry support for data augmentation.
- [2026-02-01]: Chess implementation with full rules (castling, en passant, promotion).
- [2026-02-01]: Chess uses 119-plane AlphaZero encoding with horizontal symmetry.

### Advanced MCTS (src/mcts/)
- [2026-01-26]: Gumbel AlphaZero implementation with sequential halving.
- [2026-01-26]: Improved policy targets via completed Q-values.
- [2026-01-26]: Gumbel-Top-k sampling for exploration.

### Enhanced PoC Framework (src/poc/tuning/, src/poc/statistics/)
- [2026-01-26]: Hyperparameter tuning with TPE, grid, and random samplers.
- [2026-01-26]: Statistical significance testing (t-test, Mann-Whitney, bootstrap).
- [2026-01-26]: Effect size calculations (Cohen's d, Hedges' g, Cliff's delta).
- [2026-01-26]: Multiple comparison corrections (Bonferroni, Holm, FDR).

### Google Vertex AI Training (src/vertex/)
- [2026-01-31]: Complete Vertex AI training integration for cloud-based training.
- [2026-01-31]: Pydantic configuration schemas for machine types, regions, and accelerators.
- [2026-01-31]: GCS checkpoint manager with local caching and atomic operations.
- [2026-01-31]: Multi-node distributed training setup with automatic environment detection.
- [2026-01-31]: Spot instance preemption handling with signal-based detection.
- [2026-01-31]: Cost tracking and estimation with GCP pricing data.
- [2026-01-31]: Vertex-aware trainer wrapper integrating all Vertex AI features.
- [2026-01-31]: Docker container infrastructure for training jobs.
- [2026-01-31]: CLI tools for job launching, monitoring, and management.
- [2026-01-31]: Full test suite with 124 tests (100% passing).

## Module Development Templates (src/templates/)
- [2026-01-27]: Reusable infrastructure for building new AlphaGalerkin modules.
- [2026-01-27]: Pydantic-based configuration with validation and no hardcoded values.
- [2026-01-27]: Thread-safe singleton registries with decorator-based registration.
- [2026-01-27]: Structured logging with context binding and timing utilities.
- [2026-01-27]: Base executable classes with result tracking and error handling.
- [2026-01-27]: CLI utilities with common options and error handling.
- [2026-01-27]: C4 architecture template in Mermaid format.
- [2026-01-27]: Full test suite with 107 tests (100% passing).

## PDE Game Framework (src/pde/)
- [2026-01-27]: PDEGame abstraction for treating PDE solving as sequential decision-making.
- [2026-01-27]: PDEState and PDEResult dataclasses for state representation and metrics.
- [2026-01-27]: PDE operator definitions with automatic differentiation (Poisson, Burgers, Advection-Diffusion, Heat).
- [2026-01-27]: Basis selection game for MCTS-guided Galerkin approximation.
- [2026-01-27]: Mesh refinement game for adaptive h/p-refinement strategies.
- [2026-01-27]: PDEOperatorRegistry with decorator-based registration.
- [2026-01-27]: Comprehensive test suite for all PDE components.
- [2026-02-04]: PDEGameAdapter bridging PDE games to MCTS GameInterface protocol.

## Adaptive Loss Balancing (src/training/loss_balancing.py)
- [2026-01-27]: ReLoBRaLo (Relative Loss Balancing with Random Lookback) for physics-informed training.
- [2026-01-27]: GradNorm gradient normalization for multi-task learning.
- [2026-01-27]: Uncertainty weighting with learnable log-variance parameters.
- [2026-01-27]: SoftAdapt rate-based adaptation for improving slower losses.
- [2026-01-27]: Factory function for creating balancers with Pydantic configuration.

## Physics-Informed Loss Components (src/training/physics_loss.py)
- [2026-01-27]: ResidualLoss for PDE residual minimization via autodiff.
- [2026-01-27]: BoundaryLoss for Dirichlet/Neumann/Robin BC enforcement.
- [2026-01-27]: InitialConditionLoss for time-dependent PDEs.
- [2026-01-27]: ConservationLoss for integral conservation properties.
- [2026-01-27]: PhysicsInformedLoss combining all physics terms with adaptive balancing.
- [2026-01-27]: CombinedAlphaGalerkinPhysicsLoss integrating policy/value with physics.

## Multi-Scale Fourier Features (src/modeling/multiscale_fourier.py)
- [2026-01-27]: MultiScaleFourierFeatures with multiple frequency bands to overcome spectral bias.
- [2026-01-27]: AdaptiveFourierFeatures with attention-based frequency selection.
- [2026-01-27]: ProgressiveFourierFeatures for curriculum-based frequency introduction.
- [2026-01-27]: SpatialPositionalEncoding for 2D grid data.
- [2026-01-27]: Configurable via Pydantic FourierFeaturesConfig.

## Neural Video Compression (src/video_compression/)
- [2026-01-30]: Resolution-independent neural video codec using Galerkin attention and FNet mixing.
- [2026-01-30]: Analysis transform (encoder) with O(N) Galerkin attention and O(N log N) FFT mixing.
- [2026-01-30]: Synthesis transform (decoder) with temporal cross-attention for P/B frames.
- [2026-01-30]: Scale hyperprior entropy model (Ballé et al.) for learned compression.
- [2026-01-30]: Differentiable quantization: noise, STE, and soft quantization modes.
- [2026-01-30]: MCTS-based rate control for GOP-level bit allocation with MuZero-style learned models.
- [2026-01-30]: Quality metrics: PSNR, SSIM, MS-SSIM, BD-rate computation.
- [2026-01-30]: R-D training with MSE, MS-SSIM, and perceptual (VGG) losses.
- [2026-01-30]: GOP manager for I/P/B frame scheduling and reference management.
- [2026-01-30]: Range encoder/decoder for lossless entropy coding.

### Key Architecture Decisions
- **Resolution Independence**: All encoder/decoder layers accept arbitrary (H, W) divisible by downsample factor.
- **Galerkin Attention**: Q(K^T V) formula with O(N) complexity, no softmax.
- **FNet Mixing**: torch.fft.fft2() for O(N log N) spatial mixing, no learnable parameters.
- **GDN/IGDN**: Generalized Divisive Normalization for density modeling.

## Known Issues
- SGF variation parsing in `tests/games/sgf/test_sgf.py::TestSGFParser::test_parse_variations` is skipped pending full tree-structured parsing support.
- MCTS rate-control tests in `tests/video_compression/unit/test_mcts_rate_control.py` are skipped until a trained MCTS model is available and the rate controller is enabled in the default codec path.

## Regression Surface

When changing the AlphaGalerkin solver or evaluator wiring, the following test surfaces must remain green:

| Surface | Command | What it guards |
|---|---|---|
| Solver wiring (config validation, dispatch) | `pytest tests/alphagalerkin/test_solver.py -v` | Pydantic field defaults/validators, evaluator Literal, mesh/basis dispatch |
| Trained evaluator | `pytest tests/alphagalerkin/test_trained_evaluator.py -v` | `evaluator="trained"` round-trip, `_resolve_device_cached` LRU semantics, on-instance evaluator cache, action-space mismatch under `strict=False`, GPU smoke |
| PDE end-to-end | `pytest tests/integration/test_pde_e2e.py -v` | `pde_basis`/`pde_mesh` registration, full self-play episode, trainer integration |
| MCTS evaluator protocol | `pytest tests/mcts/test_evaluator.py -v` | `RandomEvaluator` and `FNetEvaluator` `Evaluator` protocol compliance |
| Per-module coverage | `pytest tests/alphagalerkin/ --cov=src/alphagalerkin --cov-fail-under=85` | Coverage gate; current 94% |
| Noyron HX scenario (SDF, domain, scenario) | `pytest tests/pde/test_sdf.py tests/pde/test_picogk_domain.py tests/poc/test_noyron_hx_scenario.py -v` | `AnalyticalHelixSDF._nearest_t` grid-search fallback, `PicoGKDomain._project_to_surface` bisection fallback, `volume_accept_rate` property, voxel-FDM training supervision (not harmonic), `accept_rate`/`train_time_s`/`eval_time_s` metric round-trip, `_draw_pool_indices` helper, surfaced numerical-stability constants (`DEFAULT_TRANSFER_RATIO_FLOOR`, `DEFAULT_NORMALIZE_EXTENT_FLOOR`, `EVAL_SEED_STRIDE`) |
| Noyron HX per-module coverage | `pytest tests/pde/test_sdf.py tests/pde/test_picogk_domain.py tests/poc/test_noyron_hx_scenario.py --cov=src/pde/sdf --cov=src/pde/geometry_picogk --cov=src/poc/scenarios/noyron_hx --cov=src/poc/config_noyron --cov-fail-under=85` | Coverage gate on the Leap 71 v1 hardening surface; current sdf 100% / geometry_picogk 100% / config_noyron 100% / noyron_hx 97% |
| Codec perf benchmark (Phase 0) | `pytest tests/video_compression/perf/ -v` | Guards the perf benchmark harness: config validation, metrics, baseline registry, device pinning, regression diffing, CLI, and GPU test execution. |
| Decoder runtime backends (Phase 1) | `pytest tests/video_compression/perf/test_compiled_runtime.py tests/video_compression/perf/test_onnx_runtime.py tests/video_compression/perf/test_tensorrt_runtime.py tests/video_compression/perf/test_benchmark_dispatch.py tests/video_compression/runtime/ -v` | Guards the 4 decoder runtime backends: PyTorch eager, torch.compile, ONNX Runtime, TensorRT. Protocol compliance, lifecycle (prepare/decode/teardown), shape/hash validation, precision dispatch, metadata provenance. |
| Codec model zoo (Phase 2-B) | `pytest tests/video_compression/zoo/ --cov=src/video_compression/zoo --cov-fail-under=85 -v` | Guards the dual-GPU R-D Lagrangian sweep: Pydantic schemas (entry / manifest / optimizer / scheduler), JSON+YAML manifest round-trip, `_migrate_manifest_document` versioning, four device-assignment strategies (`VRAM_AWARE` / `ROUND_ROBIN` / `SINGLE_DEVICE` / `MANUAL`), `_resolve_explicit_device` (bare-`cuda` / `cuda:N` / `cpu`), `VideoCodecZoo` filesystem registry, schema-versioned forward compat. Per-module coverage **100%** on `__init__.py` / `config.py` / `device_planner.py` / `manifest.py` / `storage.py`. |
| Codec sweep orchestrator (Phase 2-D) | `pytest tests/video_compression/zoo/test_sweep.py tests/video_compression/zoo/test_sweep_parallel.py tests/video_compression/zoo/test_cli_helpers.py tests/scripts/test_train_compression_zoo.py tests/scripts/test_train_compression_zoo_entry.py tests/video_compression/training/test_zoo_trainer.py -v` | Guards the manifest-level sweep orchestrator and the multi-entry CLI: `ZooSweep.run()` + `run_parallel()` (per-device worker threads, manifest-order results), `default_entry_runner` + `make_subprocess_entry_runner` (`CUDA_VISIBLE_DEVICES` pinning + parent/child device-label translation, exit-code propagation, missing-checkpoint failure mode, `cuda_pinning="none"` pass-through), `_device_index` parser, `should_skip` hash gate, `train_compression_zoo` `dry-run`/`train` subcommands and `_only_entry_id` filter, `ZooTrainer` `train_wallclock_s` / `eval_wallclock_s` round-trip via `metrics.json`. CLI helpers (`load_dict` YAML/JSON/unsupported/empty/non-dict, `resolve_path` abs/cwd-relative/manifest-relative, `load_codec_config`, `resolve_entry` + KeyError, `resolve_codec_config_for_entry` ref-precedence + fallback + no-ref raise, `override_entry` short-circuit, `resolve_device` cascade). Zoo-subpackage coverage **98.44%** (cli_helpers 100%). |
| SBIR P40 hardening surface | `pytest tests/research/test_baselines.py tests/research/test_pinn_device.py tests/research/test_gpu_profiler.py tests/research/test_pde_benchmarks.py tests/research/test_ns_baseline.py tests/pde/test_taylor_green_invariants.py tests/scripts/test_run_sbir_p40.py -v` | Guards every SBIR P40 fix: (a) `NavierStokesOperator.exact_solution` numpy/torch parity (cross-branch property test); (b) `AMRConfig` raised defaults + target-aware `_solve_amr_1d` n_start escape from the 18-DOF Burgers ceiling; (c) `PINNConfig.device` resolution including indexed `cuda:N`, `vector_pde` auto-detect for NS, `_build_network(output_dim=2)`; (d) `GpuUtilizationProfiler` lifecycle (`__enter__`/`__exit__`, terminate-timeout fallback to `kill()`, dmon parser, no-op when `nvidia-smi` is missing); (e) `PDEBenchmarkRunner(heavy=True)` opt-in extends `refinement_levels` with `heavy_refinement_levels`; (f) `scripts/run_sbir_p40.py` helper functions (`load_config`, `apply_overrides`, `apply_benchmark_overrides`, `filter_baselines`, `build_pinn_config`, `register_pinn_profiles`). |
| SBIR P40 per-module coverage | `pytest tests/research/test_baselines.py tests/research/test_pinn_device.py tests/research/test_gpu_profiler.py tests/research/test_pde_benchmarks.py tests/research/test_ns_baseline.py tests/pde/test_taylor_green_invariants.py tests/scripts/test_run_sbir_p40.py --cov=src.research.gpu_profiler --cov=src.research.baselines --cov=src.research.pde_benchmarks --cov=src.poc.device --cov-fail-under=85` | Coverage gate on the SBIR P40 hardening surface; current `gpu_profiler.py` 96% / `baselines.py` 95% / `pde_benchmarks.py` 94% / `poc/device.py` 100% (overall 94.8%) |
| LLM-prior MCTS basis selection (mocked CPU) | `pytest tests/integrations tests/poc/test_llm_prior_ablation_config.py tests/poc/test_llm_prior_ablation_scenario.py -v -m "not gpu_required"` | Guards the LM Studio integration + `llm_prior_ablation` scenario: `LMStudioConfig` Pydantic validation, `LMStudioPolicyResponse` schema + typed exception hierarchy, deterministic `build_policy_prompt`/`prompt_hash`, `LMStudioClient` retry behaviour (JSON parse failure, action-size mismatch with corrective user-turn, `APIConnectionError`/`APITimeoutError` coercion, fallback-to-uniform on exhausted retries), evaluator protocol compliance + illegal-action masking + batch sequential equivalence + latency-sample collection, preflight (server unreachable / model missing / insufficient VRAM / passing), scenario gating (LLM-arm preflight failure / `lm_studio.enabled=False` / client-construction exception / missing trained checkpoint / `create_model_from_checkpoint` raises) and threshold-list mutation when arms drop, real-MCTS micro-run on Poisson with `RandomEvaluator`, and HTML report artifact emission. |
| LLM-prior MCTS basis selection (per-module coverage) | `pytest tests/integrations tests/poc/test_llm_prior_ablation_config.py tests/poc/test_llm_prior_ablation_scenario.py -m "not gpu_required" --cov=src/integrations/lm_studio --cov=src/poc/scenarios/llm_prior_ablation.py --cov=src/poc/scenarios/llm_prior_config.py --cov-branch --cov-fail-under=85` | Coverage gate on the LM Studio integration; current `lm_studio` package 91% (line+branch combined: `client.py` 91%, `evaluator.py` 95%, `preflight.py` 97%, `prompt.py` 100%, `config.py`/`schema.py`/`__init__.py` 100%), `llm_prior_ablation.py` 86%, `llm_prior_config.py` 100%. |
| LLM-prior MCTS basis selection (GPU smoke, manual) | `LM_STUDIO_URL=http://127.0.0.1:1234/v1 pytest tests/integrations/test_lm_studio_smoke.py -v -m gpu_required` | Real-server smoke against a running LM Studio + GPU: `complete_policy` round-trip + loose latency ceiling (10 s; the scenario asserts the headline 3 s threshold), seed reproducibility within 1e-2 or argmax stability, preflight returns `PreflightReport(passed=True)`. Auto-skips on CPU CI via the root `conftest.py` hook. |
| OOD operators (Helmholtz + Biharmonic) | `pytest tests/pde/test_ood_operators.py -v` | Guards the two held-out-generalisation operators: properties/order, manufactured `source_term`/`boundary_value`/`exact_solution` analytics, residual-vanishes-on-exact (≤1e-3, incl. Hypothesis wavenumber sweep), Helmholtz wavenumber resolution (arg → `reaction_coeff` → `DEFAULT_HELMHOLTZ_WAVENUMBER`, non-positive raises), biharmonic disconnected-solution zeros, registry round-trip, and `BasisSelectionGame` construction with finite initial error. |
| OpenAI-compatible LLM backends (WS1) | `pytest tests/integrations/test_backend_registry.py -v` | Guards the multi-backend layer: `BackendProfile` registry (built-ins, `get_backend` unknown→`KeyError`, duplicate `register_backend`→`ValueError`, frozen/extra-forbid), `apply_backend_defaults` fill-unset semantics (lm_studio no-op, vllm/llama_cpp fill, explicit values win, other fields preserved), `lm_studio` profile == historical `LMStudioConfig` defaults, `LMStudioConfig.backend`/`vram_check_mode` backwards compat (old dicts parse, invalid backend rejected), and preflight `vram_check_mode` conditioning (`off` skips `_check_vram`, `local` invokes it). Per-module coverage `openai_compat` 94%. |
| PoC baseline harness (WS2) | `pytest tests/poc/test_scenario_baselines.py tests/poc/test_cli_baselines.py tests/agents/test_research_persistence.py -v` | Guards record/diff: `ScenarioBaselineEntry`/`Document` schema + forward-compat `extra="ignore"`, `migrate_baseline_document` (unversioned→v1, future-schema raise, no caller mutation, Hypothesis idempotence), `regression_pct` sign convention (higher/lower-better, zero-baseline floor), `from_observed` direction recording, `compare` (self-clean, lower/higher-better regression, improvement, within-tolerance, missing-metric/scenario not-a-regression), JSON round-trip + load error paths, `observed_from_result_dicts` (`scenario_name`/`name` + non-numeric drop), poc-CLI `record-baseline`/`diff` (self-diff exit 0, regression exit 1, `_resolve_higher_better` suffix+extras, missing-run raise), and `agents.cli research --output-dir` persistence (run-scoped JSON, `from_dict` round-trip, baseline-diffable). Per-module coverage `poc/baselines` 100%. |
| Scaling-law scenario (mocked CPU) | `pytest tests/poc/test_scaling_law_config.py tests/poc/test_scaling_law_scenario.py tests/poc/test_centaur_common.py -v -m "not gpu_required"` | Guards `ScalingLawConfig` validation (name lock, arms/budget dedup + ≥2 distinct budgets, seed derivation, `max_rollouts_for_budget`, threshold derivation), `fit_log_log` (perfect power law, <2 points, NaN, zero-residual floor), the synthetic sweep (primary-arm aliases + per-arm/per-budget metrics, threshold pass/fail, arm-vs-arm comparison), arm gating (llm preflight fail → primary thresholds dropped → SKIPPED; trained without checkpoint skipped), HTML artifact, a real random-arm micro-run, and the shared `_centaur_common` primitives (operator/game build, `build_arm_evaluator` random/trained/llm + unknown-arm/missing-resource raises, `run_basis_selection_cell` early-return + real loop). |
| Scaling-law + centaur-common (per-module coverage) | `pytest tests/poc/test_scaling_law_config.py tests/poc/test_scaling_law_scenario.py tests/poc/test_centaur_common.py --cov=src.poc.scenarios.scaling_law --cov=src.poc.scenarios.scaling_law_config --cov=src.poc.scenarios._centaur_common --cov-branch --cov-fail-under=85` | Coverage gate; current `scaling_law.py` 86%, `scaling_law_config.py` 98%, `_centaur_common.py` 89%. |
| Centaur research-loop harness (mocked CPU) | `pytest tests/agents/test_research_loop.py -v -m "not gpu_required"` | Guards `ResearchLoopConfig`/`ResearchProblemSpec` validation (≥1 problem, unique names, default_arms dedup, seed derivation, `arms_for` override/fallback), `AgentType.RESEARCH`, and `ResearchLoopOrchestrator` orchestration via a synthetic `_solve_cell`: discovery-ledger best-arm selection + `solved_fraction`/`arm_wins_*`, status COMPLETED/FAILED by `min_solved_fraction`, parallel==sequential ledger equivalence, arm gating (llm preflight fail / trained no-checkpoint / no-arms→SKIPPED), per-problem arm override, and a real random-arm micro-run. |
| Centaur research-loop (per-module coverage) | `pytest tests/agents/test_research_loop.py tests/agents/test_config.py --cov=src.agents.research_loop --cov=src.agents.config --cov-branch --cov-fail-under=85` | Coverage gate; current `research_loop.py` 86%, `agents/config.py` 98%. |
| Scaling-law + research-loop (GPU smoke, manual) | `pytest tests/poc/test_scaling_law_smoke.py tests/agents/test_research_loop_smoke.py -v -m gpu_required` | Real-CUDA smoke (auto-skips on CPU CI): random-arm scaling sweep + manifest sweep build their fits/ledgers; LLM-arm variants run only when `LM_STUDIO_URL` is set. |
| Centaur test pyramid (sanity / integration / e2e / regression / AQA) | `pytest tests/integration/test_centaur_sanity.py tests/integration/test_centaur_integration.py tests/integration/test_centaur_aqa.py tests/regression/test_centaur_regression.py tests/e2e/test_centaur_e2e.py -v` | CPU-only, real-interface coverage of the three deliverables: **sanity** (imports/registry/PDE_TYPE_MAP/AgentType/demo-YAML parse/ood_pde Literal), **integration** (OOD operator→game→adapter→MCTS micro-run, `ScalingLawScenario` via `ScenarioRunner`, real multi-PDE research-loop, parallel-ledger consistency, shared gating helpers), **e2e** (shipped demo YAML → `load_config_from_dict`/`load_config_file` dispatch, `run_from_config` with persisted result JSON, poc/agents CLI journeys), **regression** (OOD residual≤1e-3 contract, `_PDE_TYPE_MAP is PDE_TYPE_MAP` + llm_prior private-API intact, `fit_log_log` sign, discovery-ledger winner, gating skip semantics, (N,1) broadcasting guard), **AQA** (each Brown theme's acceptance criterion on a real run + "every knob is a typed field" governance check). |

The trained-evaluator path additionally depends on `src/training/checkpoint.py::create_model_from_checkpoint`, `src/modeling/model.py::AlphaGalerkinModel`, and `src/mcts/evaluator.py::FNetEvaluator` — changes there should run the *Trained evaluator* surface above as a smoke test.

The LLM-prior surface depends on the MCTS `Evaluator` Protocol (`src/mcts/evaluator.py`), `BasisSelectionGame` (`src/pde/games/basis_selection.py`) and `PDEGameAdapter` (`src/pde/mcts_adapter.py`), `BaseScenario` lifecycle (`src/poc/registry.py`), `load_config_from_dict` dispatch (`src/poc/config.py`), and `resolve_device` (`src/poc/device.py`) — changes there should run the *LLM-prior MCTS basis selection (mocked CPU)* surface as a smoke test.

The Noyron HX surface depends on `src/pde/operators_picogk.py::HelicalHeatOperator`, `src/physics/voxel_fdm.py::solve_steady_heat_voxel`, and `src/experiments/physics_model.py::PhysicsOperator` (3D-aware) — changes there should run the *Noyron HX scenario* surface above as a smoke test.

The scaling-law and centaur research-loop surfaces both depend on the shared `src/poc/scenarios/_centaur_common.py` primitives (canonical `PDE_TYPE_MAP`, `build_pde_operator`/`build_basis_game`/`build_arm_evaluator`/`run_basis_selection_cell`), which `llm_prior_ablation` also delegates to — changes to `_centaur_common` should run the *LLM-prior MCTS basis selection (mocked CPU)*, *Scaling-law scenario (mocked CPU)*, and *Centaur research-loop harness (mocked CPU)* surfaces together. Adding a new PDE operator means: register it in `src/pde/registry.py`, add it to `PDEType` (`src/pde/config.py`) and the canonical `PDE_TYPE_MAP`, then extend the relevant `Literal` enums (`ood_pde`, `ScalingLawConfig.pde`, `ResearchPDEName`).

## Verification Commands
```bash
# Linting and type checking
ruff check src/
mypy src/ --strict

# Unit tests
pytest tests/math_kernel/ -v
pytest tests/training/ -v

# Integration tests
pytest tests/integration/ -v

# Full test suite
pytest tests/ -v

# Verify resolution independence
python -m src.tools.verify_invariance --train-size 9 --infer-size 19

# LLM-prior MCTS basis selection (CPU-safe, mocked)
ruff check src/integrations/lm_studio src/poc/scenarios/llm_prior_ablation.py src/poc/scenarios/llm_prior_config.py
ruff format --check src/integrations/lm_studio src/poc/scenarios/llm_prior_ablation.py src/poc/scenarios/llm_prior_config.py
mypy --strict src/integrations/lm_studio src/poc/scenarios/llm_prior_ablation.py src/poc/scenarios/llm_prior_config.py
pytest tests/integrations tests/poc/test_llm_prior_ablation_config.py tests/poc/test_llm_prior_ablation_scenario.py -v -m "not gpu_required"

# LLM-prior coverage gate (85%)
pytest tests/integrations tests/poc/test_llm_prior_ablation_config.py tests/poc/test_llm_prior_ablation_scenario.py -m "not gpu_required" --cov=src/integrations/lm_studio --cov=src/poc/scenarios/llm_prior_ablation.py --cov=src/poc/scenarios/llm_prior_config.py --cov-branch --cov-fail-under=85

# LLM-prior GPU smoke (manual; requires CUDA + a live LM Studio endpoint)
LM_STUDIO_URL=http://127.0.0.1:1234/v1 pytest tests/integrations/test_lm_studio_smoke.py -v -m gpu_required

# End-to-end demo (GPU + LM Studio required)
pip install -e '.[lm-studio]'
python -m src.poc.cli run --config config/scenarios/llm_prior_demo.yaml
```

## Headline-Run Operationalisation Commands (OpenAI-compatible backends + baseline harness)
```bash
# WS1 — multi-backend LLM (CPU-safe, mocked) + coverage gate
ruff check src/integrations/openai_compat
mypy --strict src/integrations/openai_compat src/poc/baselines
pytest tests/integrations/test_backend_registry.py -v
pytest tests/integrations/test_backend_registry.py \
       --cov=src/integrations/openai_compat --cov-branch --cov-fail-under=85

# WS2 — PoC baseline harness + research persistence (CPU-safe) + coverage gate
pytest tests/poc/test_scenario_baselines.py tests/poc/test_cli_baselines.py \
       tests/agents/test_research_persistence.py -v
pytest tests/poc/test_scenario_baselines.py tests/poc/test_cli_baselines.py \
       --cov=src/poc/baselines --cov-branch --cov-fail-under=85

# Record a headline baseline from a completed run and regression-gate a later run
python -m src.poc.cli run --config config/scenarios/scaling_law_cpu.yaml
python -m src.poc.cli record-baseline --run-id <id> --out /tmp/base.json --tolerance-pct 10
python -m src.poc.cli diff --baseline /tmp/base.json --run-id <id>   # exits 0 (self), 1 on regression

# Switch the LLM arm to vLLM / llama.cpp (headline GPU run; set backend in the YAML lm_studio block)
#   lm_studio: { enabled: true, backend: vllm, model: Qwen/Qwen2.5-14B-Instruct }
python -m src.agents.cli research --config config/agents/research_loop_cpu.yaml --output-dir /tmp/rl
```

## AI-for-Physics Scaling Commands (OOD operators + scaling law + research loop)
```bash
# OOD operators — registry + residual property tests
pytest tests/pde/test_ood_operators.py -v

# Scaling-law scenario (CPU-safe, mocked) + coverage gate
pytest tests/poc/test_scaling_law_config.py tests/poc/test_scaling_law_scenario.py \
       tests/poc/test_centaur_common.py -v -m "not gpu_required"
pytest tests/poc/test_scaling_law_config.py tests/poc/test_scaling_law_scenario.py \
       tests/poc/test_centaur_common.py \
       --cov=src.poc.scenarios.scaling_law --cov=src.poc.scenarios.scaling_law_config \
       --cov=src.poc.scenarios._centaur_common --cov-branch --cov-fail-under=85

# Centaur research-loop harness (CPU-safe, mocked) + coverage gate
pytest tests/agents/test_research_loop.py -v -m "not gpu_required"
pytest tests/agents/test_research_loop.py tests/agents/test_config.py \
       --cov=src.agents.research_loop --cov=src.agents.config --cov-branch --cov-fail-under=85

# List scenarios / agents (new entries appear)
python -m src.poc.cli info scaling_law
python -m src.agents.cli list-agents   # AgentType.RESEARCH; 'research' subcommand via --help

# GPU smoke (manual; auto-skips on CPU CI, LLM arm gated on LM_STUDIO_URL)
LM_STUDIO_URL=http://127.0.0.1:1234/v1 \
  pytest tests/poc/test_scaling_law_smoke.py tests/agents/test_research_loop_smoke.py -v -m gpu_required

# Headline GPU runs (manual; CUDA + LM Studio)
python -m src.poc.cli run --config config/scenarios/scaling_law_demo.yaml
python -m src.agents.cli research --config config/agents/research_loop_demo.yaml
```

## Training Commands
```bash
# Default training (full config)
python -m scripts.train

# Fast test training (small model, few steps)
python -m scripts.train --config-name=train_fast

# Override parameters
python -m scripts.train training.batch_size=64 training.total_steps=10000

# Resume from checkpoint
python -m scripts.train +resume=checkpoints/alphagalerkin/checkpoint_00010000.pt

# Train on GPU with custom experiment name
python -m scripts.train device=cuda experiment_name=my_experiment
```

## Physics PoC Commands
```bash
# Train physics operator on Poisson data (supervised learning)
python -m src.experiments.train_physics

# Train with W&B logging
python -m src.experiments.train_physics --wandb

# Custom training configuration
python -m src.experiments.train_physics --train-size 9 --eval-size 19 --n-epochs 100

# Verify zero-shot transfer (train 9x9 → eval 9,13,19)
python -m src.experiments.verify_transfer

# Verify with existing model
python -m src.experiments.verify_transfer --model-path outputs/physics_poc/best_model.pt

# Run FNet vs Softmax speed benchmark
python -m src.experiments.benchmark_fnet

# Benchmark with custom sizes
python -m src.experiments.benchmark_fnet --sizes 81,169,361,625 --batch-size 64

# Run Fredholm integral property tests
pytest tests/math_kernel/test_fredholm.py -v
```

## PoC Scenario Framework Commands
```bash
# List available scenarios
python -m src.poc.cli list

# Show scenario details
python -m src.poc.cli info transfer

# Run all scenarios
python -m src.poc.cli run

# Run specific scenario
python -m src.poc.cli run --scenario transfer

# Run from config file (full suite)
python -m src.poc.cli run --config config/scenarios/poc_full.yaml

# Run quick validation suite
python -m src.poc.cli run --config config/scenarios/poc_quick.yaml

# Run with parallel workers
python -m src.poc.cli run --parallel 4

# Compare two runs
python -m src.poc.cli compare run_a run_b

# PoC framework unit tests
pytest tests/poc/ -v
```

## Distributed Training Commands
```bash
# Launch distributed training with torchrun (4 GPUs)
torchrun --nproc_per_node=4 scripts/train_distributed.py

# Multi-node training (2 nodes, 4 GPUs each)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=<MASTER_IP> scripts/train_distributed.py

# Unit tests for distributed module
pytest tests/distributed/ -v
```

## Vertex AI Training Commands
```bash
# Build and push training container to Artifact Registry
./scripts/build_vertex_container.sh my-project us-central1

# Launch training job on Vertex AI
python -m scripts.train_vertex \
    --project my-project \
    --region us-central1 \
    --bucket gs://my-training-bucket \
    --machine-type a2-highgpu-1g \
    --accelerator-type NVIDIA_TESLA_A100 \
    --accelerator-count 1 \
    --container-uri us-central1-docker.pkg.dev/my-project/alphagalerkin/trainer:latest

# Launch spot (preemptible) training for cost savings
python -m scripts.train_vertex \
    --project my-project \
    --bucket gs://my-training-bucket \
    --spot

# Launch multi-node distributed training
python -m scripts.train_vertex \
    --project my-project \
    --bucket gs://my-training-bucket \
    --machine-type a2-highgpu-8g \
    --accelerator-type NVIDIA_TESLA_A100 \
    --accelerator-count 8 \
    --replica-count 4

# List running Vertex AI jobs
python -m scripts.vertex_jobs list --project my-project

# Show job status
python -m scripts.vertex_jobs show JOB_ID --project my-project

# Wait for job completion
python -m scripts.vertex_jobs wait JOB_ID --project my-project

# Cancel a running job
python -m scripts.vertex_jobs cancel JOB_ID --project my-project

# View job logs
python -m scripts.vertex_jobs logs JOB_ID --project my-project

# Unit tests for Vertex AI module
pytest tests/vertex/ -v
```

## ONNX Export Commands
```bash
# Export model to ONNX
python -m src.deployment.export_onnx \
    --checkpoint path/to/model.pt \
    --output model.onnx

# Export with quantization
python -m src.deployment.export_onnx \
    --checkpoint path/to/model.pt \
    --output model_int8.onnx \
    --quantize dynamic

# Validate exported model
python -m src.deployment.validate \
    --pytorch path/to/model.pt \
    --onnx model.onnx

# Unit tests for deployment module
pytest tests/deployment/ -v
```

## Multi-Game Commands
```bash
# Train on Go (default)
python -m scripts.train game=go

# List registered games
python -c "from src.games import GameRegistry; print(GameRegistry().list_games())"

# Unit tests for games module
pytest tests/games/ -v
```

## Module Development Template Commands
```bash
# Run template tests
pytest tests/templates/ -v

# Example: Create a new module configuration
python -c "
from src.templates.config import BaseModuleConfig, create_config_class
from pydantic import Field

# Method 1: Subclass directly
class MyModuleConfig(BaseModuleConfig):
    my_param: int = Field(default=100, ge=1, description='My parameter')

config = MyModuleConfig(name='test')
print(f'Config hash: {config.compute_hash()}')

# Method 2: Use factory function
QuickConfig = create_config_class(
    'QuickConfig',
    my_float=(float, Field(default=0.5, gt=0, lt=1)),
)
quick = QuickConfig(name='quick')
print(f'Quick config: {quick.my_float}')
"

# Example: Create and use a registry
python -c "
from src.templates.registry import create_registry

class BaseProcessor:
    def process(self, data): raise NotImplementedError

ProcessorRegistry, register_processor = create_registry('Processor', BaseProcessor)

@register_processor('upper')
class UpperProcessor(BaseProcessor):
    def process(self, data): return data.upper()

# Use the registry
proc_cls = ProcessorRegistry().get('upper')
processor = proc_cls()
print(processor.process('hello'))  # HELLO
"

# Example: Use structured logging
python -c "
from src.templates.logging import create_logger_class, configure_module_logging

configure_module_logging(level='DEBUG')
MyLogger = create_logger_class('MyModule')
logger = MyLogger('component', run_id='test123')

with logger.timed('operation'):
    logger.metric('accuracy', 0.95, epoch=1)
"
```

## Hyperparameter Tuning Commands
```bash
# Run hyperparameter tuning for transfer scenario
python -c "
from src.poc.tuning import HyperparameterTuner, TuningConfig
from src.poc.scenarios.transfer import TransferScenario

config = TuningConfig(
    n_trials=50,
    sampler='tpe',
    search_space={
        'd_model': {'type': 'int', 'low': 64, 'high': 256, 'log_scale': True},
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log_scale': True},
    }
)
tuner = HyperparameterTuner(config, TransferScenario)
result = tuner.tune()
print(f'Best params: {result.best_params}')
"

# Statistical comparison of two runs
python -c "
from src.poc.statistics import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
result = analyzer.compare_runs([0.05, 0.04, 0.06], [0.03, 0.02, 0.04])
print(f'p-value: {result.p_value}, significant: {result.is_significant}')
"
```

## Video Compression Commands
```bash
# Train compression model
python scripts/train_compression.py --data-dir data/images --epochs 100

# Train with specific lambda
python scripts/train_compression.py --data-dir data/images --lambda-rd 0.01

# Encode video
python scripts/encode_video.py input.mp4 output.agk --qp 32

# Encode with custom model
python scripts/encode_video.py input.mp4 output.agk --model checkpoints/codec.pt

# Run video compression tests
pytest tests/video_compression/ -v

# Test configuration validation
pytest tests/video_compression/unit/test_config.py -v

# Test encoder/decoder
pytest tests/video_compression/unit/test_encoder.py tests/video_compression/unit/test_decoder.py -v
```

## PDE Game Commands
```bash
# Run PDE game tests
pytest tests/pde/ -v

# Example: Create and use PDE operators
python -c "
from src.pde.operators import PoissonOperator
from src.pde.config import PDEConfig, PDEType
import numpy as np

config = PDEConfig(name='test', pde_type=PDEType.POISSON)
operator = PoissonOperator(config)
points = operator.generate_collocation_points(100)
source = operator.source_term(points)
print(f'Generated {len(points)} collocation points')
"

# Example: Create basis selection game
python -c "
from src.pde.games import BasisSelectionGame
from src.pde.operators import PoissonOperator
from src.pde.config import PDEConfig, PDEGameConfig, PDEType

pde_config = PDEConfig(name='poisson', pde_type=PDEType.POISSON)
game_config = PDEGameConfig(name='basis_game', pde_config=pde_config, game_mode='basis_selection')
operator = PoissonOperator(pde_config)
game = BasisSelectionGame(operator, game_config)
state = game.get_initial_state()
print(f'Initial error: {state.error_estimate:.6f}')
"

# Example: Use adaptive loss balancing
python -c "
from src.training.loss_balancing import create_loss_balancer, LossBalancingConfig, BalancingStrategy
import torch

config = LossBalancingConfig(name='test', strategy=BalancingStrategy.RELOBRALO)
balancer = create_loss_balancer(config, ['policy', 'value', 'physics'])
losses = {'policy': torch.tensor(1.0), 'value': torch.tensor(0.5), 'physics': torch.tensor(2.0)}
result = balancer.compute_weighted_loss(losses)
print(f'Weights: {result.weights}')
"
```

## Directory Structure
```
src/
  integrations/ - Third-party-service integrations (gated behind optional extras)
    lm_studio/      - OpenAI-compatible local-LLM client + MCTS evaluator
      config.py     - LMStudioConfig Pydantic schema (no hardcoded values)
      schema.py     - LMStudioPolicyResponse + typed exception hierarchy
      prompt.py     - Deterministic prompt builder + sha256-truncated hash
      client.py     - Synchronous openai-SDK wrapper with bounded retries
      preflight.py  - Server reachable + model present + free-VRAM check
      evaluator.py  - LMStudioEvaluator implementing src/mcts/evaluator.py::Evaluator
  modeling/     - Neural architectures and layers
    multiscale_fourier.py - Multi-scale Fourier features for spectral bias mitigation
  math_kernel/  - Basis functions, integral approximations
  mcts/         - Monte Carlo Tree Search logic
    gumbel.py         - Gumbel AlphaZero MCTS implementation
  tools/        - Verification and utility scripts
  training/     - Training infrastructure
    loss.py           - AlphaGalerkinLoss (policy + value + LBB)
    loss_balancing.py - ReLoBRaLo and other adaptive loss balancing
    physics_loss.py   - Physics-informed loss components
    replay_buffer.py  - Uniform and prioritized replay buffers
    self_play.py      - MCTS-based self-play game generation
    trainer.py        - Main Trainer class
    checkpoint.py     - Checkpoint save/load management
    evaluation.py     - Win rate and policy agreement metrics
  data/         - Data loading and preprocessing
    dataset.py        - PyTorch Dataset classes
    collate.py        - Variable board size collation
  physics/      - Synthetic physics data generation
    poisson.py        - Poisson equation solver (DST-based)
  experiments/  - Physics PoC experiments
    physics_model.py  - PhysicsOperator neural network
    train_physics.py  - Supervised learning on Poisson data
    verify_transfer.py - Zero-shot transfer verification
    benchmark_fnet.py - FNet O(N log N) speed benchmark
  distributed/  - Distributed training infrastructure
    config.py         - Pydantic distributed config schemas
    trainer.py        - DistributedTrainer with DDP
    gradient_sync.py  - NCCL gradient synchronization
    launcher.py       - torchrun/SLURM launcher utilities
    worker.py         - Distributed self-play workers
    model_zoo.py      - Model checkpoint management
  deployment/   - Model export and deployment
    config.py         - Export/quantization config schemas
    export_onnx.py    - PyTorch to ONNX conversion
    quantize.py       - Model quantization utilities
    runtime.py        - ONNX Runtime inference wrapper
    validate.py       - Export validation tools
  games/        - Multi-game support
    interface.py      - Abstract GameInterface base class
    registry.py       - Game registration and discovery
    state.py          - Generic game state representation
    go.py             - Go game implementation
  pde/          - PDE Game Framework
    config.py         - Pydantic PDE configuration schemas
    game.py           - Abstract PDEGame base class
    operators.py      - PDE operator definitions (Poisson, Burgers, etc.)
    registry.py       - PDE operator registration
    mcts_adapter.py   - Adapter bridging PDE games to MCTS GameInterface
    games/            - Concrete PDE game implementations
      basis_selection.py  - Galerkin basis selection game
      mesh_refinement.py  - Adaptive mesh refinement game
  poc/          - PoC scenario framework
    config.py         - Pydantic configuration schemas
    registry.py       - Scenario registration and discovery
    runner.py         - Scenario execution engine
    results.py        - Result collection and persistence
    logging.py        - Structured logging utilities
    cli.py            - CLI entry point
    scenarios/        - Built-in scenario implementations
      transfer.py     - Zero-shot transfer scenario
      complexity.py   - O(N) complexity benchmark
      stability.py    - LBB stability monitoring
      llm_prior_config.py   - LLMPriorAblationConfig (Pydantic)
      llm_prior_ablation.py - LLMPriorAblationScenario (LM Studio MCTS prior)
    tuning/           - Hyperparameter tuning
      config.py       - Tuning configuration schemas
      sampler.py      - Parameter samplers (TPE, grid, random)
      tuner.py        - HyperparameterTuner orchestrator
    statistics/       - Statistical analysis
      significance.py - Significance testing & effect sizes
  templates/    - Reusable module development infrastructure
    config.py         - Base Pydantic configuration classes
    registry.py       - Thread-safe singleton registry pattern
    logging.py        - Structured logging with context binding
    base.py           - Base executable classes with result tracking
    cli.py            - CLI utilities with common options
  video_compression/  - Neural video compression system
    config.py         - Pydantic configuration schemas
    models/           - Neural network models
      encoder.py      - Analysis transform with FNet + Galerkin
      decoder.py      - Synthesis transform
      hyperprior.py   - Scale hyperprior entropy model
      quantizer.py    - Differentiable quantization
    codec/            - Codec implementation
      codec.py        - Complete encode/decode pipeline
      entropy_coder.py - Range encoder/decoder
      gop_manager.py  - GOP and reference management
    runtime/          - Decoder runtime backends (Phase 1)
      protocol.py     - DecoderRuntime Protocol + DecoderRuntimeContext
      registry.py     - @register_runtime + RuntimeRegistry
      metadata.py     - CompiledArtifactMetadata provenance
      pytorch_eager.py    - Baseline eager runtime
      pytorch_compiled.py - torch.compile + inductor
      onnx_runtime.py     - ONNX Runtime + CUDA EP
      tensorrt_runtime.py - torch_tensorrt Dynamo IR
    perf/             - Performance benchmark harness (Phase 0)
    mcts/             - MCTS rate control
      networks.py     - Policy, value, dynamics networks
      rate_control.py - MCTS-based rate controller
    metrics/          - Quality metrics
      quality.py      - PSNR, SSIM, MS-SSIM
      rd_curves.py    - BD-rate computation
    training/         - Training utilities
      loss.py         - R-D loss functions
      trainer.py      - Compression trainer
tests/
  math_kernel/  - Property-based tests for mathematical operators
    test_fredholm.py  - Fredholm integral equation tests
  training/     - Tests for training infrastructure
  integration/  - End-to-end integration tests
  poc/          - PoC framework tests
    test_config.py    - Configuration validation tests
    test_registry.py  - Scenario registration tests
    test_runner.py    - Runner execution tests
    test_results.py   - Result collection tests
  distributed/  - Distributed training tests
    test_config.py    - Config validation tests
  deployment/   - Deployment tests
    test_config.py    - Export/quantization config tests
  games/        - Multi-game tests
    test_go.py        - Go implementation tests
  pde/          - PDE framework tests
    test_config.py    - PDE configuration validation tests
    test_operators.py - PDE operator tests
  modeling/     - Neural network modeling tests
    test_multiscale_fourier.py - Multi-scale Fourier feature tests
  templates/    - Module development template tests
    test_config.py    - Configuration validation tests
    test_registry.py  - Registry pattern tests
    test_logging.py   - Logging utilities tests
    test_base.py      - Base executable tests
  video_compression/  - Video compression tests
    unit/             - Unit tests
      test_config.py    - Configuration validation
      test_encoder.py   - Encoder tests
      test_decoder.py   - Decoder tests
      test_quantizer.py - Quantizer tests
      test_metrics.py   - Quality metrics tests
config/         - Hydra/Pydantic configuration schemas
  train.yaml          - Default training config
  train_fast.yaml     - Fast test config
  scenarios/          - PoC scenario configurations
    poc_full.yaml     - Full PoC suite
    poc_quick.yaml    - Quick validation suite
    transfer_ablation.yaml - Transfer ablation study
docs/           - Documentation
  architecture/       - C4 architecture diagrams
    c4_model.md       - C4 model documentation
  templates/          - Module development templates
    IMPLEMENTATION_TEMPLATE.md - Agentic coding system prompt template
    C4_TEMPLATE.md    - C4 architecture template in Mermaid format
  IMPLEMENTATION_PLAN.md - Next-phase implementation plan
  PROMPT_TEMPLATE.md  - Agentic coding prompt template
scripts/        - CLI entry points
  train.py            - Training CLI with Hydra
  train_vertex.py     - Vertex AI training job launcher
  vertex_jobs.py      - Vertex AI job management CLI
  build_vertex_container.sh - Build and push training container
docker/         - Container definitions
  Dockerfile.vertex   - Vertex AI training container
```
