# AlphaGalerkin Next Steps Plan

> **Investigation Date:** 2026-02-01
> **Status:** Active — Milestones 1, 2, 3, 4, 6, 8 ✅ Complete; 5, 7, 9 Partial; **Learned PDE Evaluator ✅ Complete (2026-04-25, PR #54)**
> **Methodology:** Universal Dev Agent with Agentic Sub-Tasks
> **Last Updated:** 2026-04-25

---

## Executive Summary

Based on comprehensive codebase exploration, AlphaGalerkin is a **mature v2.0 implementation** with excellent mathematical foundations and extensive module coverage. However, several critical gaps exist that prevent true production readiness:

| Gap Category | Severity | Modules Affected |
|--------------|----------|------------------|
| ~~No CI/CD Pipeline~~ | ✅ Fixed | All |
| ~~Video Compression Hyperprior~~ | ✅ Fixed | Video Compression |
| ~~Integration Gaps~~ | ✅ Fixed | PDE-Training, Curriculum Learning |
| ~~Trained-Evaluator Stub~~ | ✅ Fixed (2026-04-25, PR #54) | `src/alphagalerkin/solver.py`, `src/mcts/evaluator.py` |
| **Documentation Gaps** | Low | Distributed, Edge Deployment |

---

## Project Context

```yaml
PROJECT_NAME: AlphaGalerkin
DOMAIN: Neural Operator Learning for Game AI
GOAL: Production-grade resolution-independent Go AI with multi-game support
CONSTRAINTS:
  - Python 3.11+, PyTorch 2.0+
  - No hardcoded board sizes
  - LBB stability (dim(Key) >= dim(Query))
  - >80% test coverage on core logic
```

---

## Milestone 1: CI/CD & Production Foundation ✅

**Goal:** Establish automated quality gates and reproducible builds.
**Duration:** 1-2 days
**Subagents:** Planner, SQE, Orchestrator

**Status:** ✅ **COMPLETED** (January 2026 → March 2026)

### What Was Delivered
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) with 8 stages including chess pipeline
- Coverage gates: 85% overall, 85% per-module (pde, modeling, training, research)
- MyPy strict enforcement (`continue-on-error: false`)
- Nightly schedule (`cron: '0 4 * * *'`) and performance benchmark job
- Pre-commit hooks (`.pre-commit-config.yaml`) with ruff, mypy, trailing whitespace

### Epic 1.1: GitHub Actions CI Pipeline

**Story 1.1.1: Create Core CI Workflow**

- **Task:** Create `.github/workflows/ci.yml` with:
  - `ruff check src/ tests/`
  - `mypy src/ --strict`
  - `pytest tests/ -v --tb=short`
- **Acceptance Criteria:**
  - [ ] PR triggers CI automatically
  - [ ] All existing tests pass
  - [ ] Badge displays on README
- **Subagent:** Orchestrator

**Story 1.1.2: Add Test Coverage Reporting**

- **Task:** Integrate pytest-cov with codecov/coveralls
- **Acceptance Criteria:**
  - [ ] Coverage report generated on each PR
  - [ ] Coverage badge in README
  - [ ] Minimum 80% threshold enforced
- **Subagent:** SQE

**Story 1.1.3: Create Pre-commit Hooks**

- **Task:** Add `.pre-commit-config.yaml` with ruff, mypy, trailing whitespace
- **Acceptance Criteria:**
  - [ ] Developers can run `pre-commit install`
  - [ ] Hooks auto-fix lint issues
  - [ ] CI validates pre-commit ran
- **Subagent:** Coder

### Epic 1.2: Reproducible Builds

**Story 1.2.1: Create Dependency Lock File**

- **Task:** Generate `requirements-lock.txt` with pinned versions
- **Acceptance Criteria:**
  - [ ] All transitive dependencies pinned
  - [ ] Lock file tested in clean environment
  - [ ] CI uses lock file for reproducibility
- **Subagent:** Coder

**Story 1.2.2: Add Development Dockerfile**

- **Task:** Create `docker/Dockerfile.dev` for local development
- **Acceptance Criteria:**
  - [ ] Builds successfully with all dependencies
  - [ ] Tests pass inside container
  - [ ] GPU passthrough documented
- **Subagent:** Coder

---

## Milestone 2: Critical Bug Fixes ✅

**Goal:** Fix incomplete implementations blocking production use.
**Duration:** 2-3 days
**Subagents:** Coder, SQE, Reviewer

**Status:** ✅ **COMPLETED** (February 2026)

### What Was Delivered
- Video Compression Hyperprior: proper z_bitstream encoding/decoding for entropy model ✅
- Collator action mask size: fixed tensor size mismatch for chess 4672-action policy ✅
- Chess underpromotion encode/decode mismatch fixed ✅
- MCTS tree advance bug fixed ✅
- Race condition in `forward()` DDP mutation fixed ✅

### Epic 2.1: Video Compression Hyperprior

**Story 2.1.1: Implement Proper Hyperprior Encoding**

- **Task:** Fix TODO at `scripts/encode_video.py:261`
- **Context:** Currently uses approximation instead of proper z encoding
- **Acceptance Criteria:**
  - [ ] Hyperprior z data properly encoded
  - [ ] Compression ratio improves 5-10%
  - [ ] Unit tests for encoding/decoding roundtrip
- **Subagent:** Coder

**Story 2.1.2: Implement Proper Hyperprior Decoding**

- **Task:** Fix TODO at `scripts/decode_video.py:420`
- **Context:** Currently uses placeholder scales
- **Acceptance Criteria:**
  - [ ] Accurate scales decoded from hyperprior
  - [ ] No quality degradation vs. encoding
  - [ ] Integration test: encode → decode → verify
- **Subagent:** Coder

### Epic 2.2: SGF Parsing Completion

**Story 2.2.1: Implement Variation/Tree Node Parsing**

- **Task:** Fix skipped test for SGF variation parsing
- **Location:** `tests/games/test_sgf.py`
- **Acceptance Criteria:**
  - [ ] Parse SGF files with variations
  - [ ] Handle nested variations correctly
  - [ ] Test with real KGS/IGS game files
- **Subagent:** Coder

---

## Milestone 3: Multi-Game Support Completion ✅

**Goal:** Validate multi-game abstraction with Chess implementation.
**Status:** ✅ **COMPLETED** (Sprint 1-2, March 2026)

### What Was Delivered

- **Chess Rules Engine** (`src/games/chess.py`): Full legal move generation, check/checkmate, castling, en passant, promotion, 50-move rule, threefold repetition, insufficient material
- **119-Channel State Encoding**: AlphaZero-compatible tensor representation with 8-move history
- **4672-Action Policy Head**: Dense action encoding covering all queen/knight/underpromotion moves
- **Game-Agnostic Training**: `SelfPlayWorker`, `Trainer`, collators all work with both Go and Chess
- **Stockfish Benchmark Evaluation**: `Trainer._run_engine_evaluation()` with W&B Elo logging
- **UCI Engine Integration**: Full `EngineMatch`, `UCIAdapter`, `EloCalculator` subsystem
- **78 Chess Tests**: Exhaustive encode/decode, E2E training, checkpoint resume, security tests
- **CI Coverage Gate**: `--cov-fail-under=80` on `chess.py` (97%) and `wrapper.py` (100%)
- **Training Validated**: 10-step run completed, loss decreased 9.06 → 7.50

### Remaining (Sprint 3-4)

- [ ] Full-scale training (5K-50K steps)
- [ ] Transfer learning experiment (Go → Chess)

---

## Milestone 4: PDE Game Integration ✅

**Goal:** Connect PDE module to main training pipeline.
**Duration:** 3-4 days
**Subagents:** Planner, Coder, Reviewer

**Status:** ✅ **COMPLETED** (February-April 2026)

### What Was Delivered
- `CombinedAlphaGalerkinPhysicsLoss` wired into trainer with `lbb_constant`, `action_mask`, `model` params
- `PDEGameInterface` bridges PDE games to `GameInterface` for `GameRegistry` registration
- `PDEGameAdapter` bridges PDE games to MCTS search engine
- PDE games (`pde_basis`, `pde_mesh`) registered in `GameRegistry` via `src/pde/register_games.py`
- `config/train_pde.yaml` for MCTS-guided basis selection training
- 52 comprehensive physics loss tests (config toggle, gradient flow, property-based)
- 40 PDE-MCTS self-play tests
- `PhysicsLoss` with Laplacian regularization via autodiff

### Epic 4.1: PDE-Informed Training Mode

**Story 4.1.1: Enable Physics Loss in Standard Training**

- **Task:** Wire `CombinedAlphaGalerkinPhysicsLoss` into trainer
- **Config Addition:** `training.physics_informed: bool = False`
- **Acceptance Criteria:**
  - [x] Physics loss computed when enabled
  - [x] Gradient flow verified (no NaN/inf)
  - [x] Config validated via Pydantic
- **Subagent:** Coder

**Story 4.1.2: PDE Basis Selection Self-Play**

- **Task:** Use MCTS for Galerkin basis selection
- **Integration Point:** `src/pde/games/basis_selection.py` + `src/mcts/`
- **Acceptance Criteria:**
  - [x] MCTS can play BasisSelectionGame
  - [x] Policy/value heads work with PDE state
  - [x] Error reduction logged per episode
- **Subagent:** Coder

### Epic 4.2: PDE Documentation

**Story 4.2.1: Create PDE User Guide**

- **Task:** Document PDE module usage in `docs/pde_guide.md`
- **Sections:**
  - Supported PDE operators
  - Game mode configuration
  - Example usage for Poisson, Burgers
- **Acceptance Criteria:**
  - [x] All config options documented
  - [x] Working code examples
  - [ ] Linked from README
- **Subagent:** Coder

---

## Milestone 5: Distributed Training Validation ⚠️ Partial

**Goal:** Verify distributed training at scale.
**Duration:** 3-4 days
**Subagents:** SQE, Orchestrator

**Status:** ⚠️ **PARTIALLY COMPLETED** — 35 new DistributedTrainer tests added (April 2026), multi-node validation pending

### What Was Delivered
- 35 new DistributedTrainer tests covering metrics, checkpoints, multi-process patterns
- BaseTrainer extracted with shared AMP, gradient clipping, LR scheduling
- Mocked MCTS self-play in all trainer tests to prevent hanging

### Epic 5.1: Multi-Process Integration Tests

**Story 5.1.1: Mock NCCL Multi-Process Test**

- **Task:** Create integration test with 4 processes
- **File:** `tests/integration/test_distributed_multiprocess.py`
- **Acceptance Criteria:**
  - [x] Test spawns 4 processes
  - [x] Gradient synchronization verified
  - [x] No deadlocks or race conditions
- **Subagent:** SQE

**Story 5.1.2: Fix Skipped Vertex Launcher Tests**

- **Task:** Improve SDK mocking for launcher tests
- **Location:** `tests/vertex/test_launcher.py` (5 skipped)
- **Acceptance Criteria:**
  - [ ] All 5 tests enabled and passing
  - [ ] SDK interactions properly mocked
  - [ ] No external API calls in tests
- **Subagent:** SQE

### Epic 5.2: Distributed Training Guide

**Story 5.2.1: Create End-to-End Tutorial**

- **Task:** Document distributed training in `docs/distributed_guide.md`
- **Sections:**
  - Environment setup (NCCL, network)
  - Single-node multi-GPU
  - Multi-node with torchrun
  - SLURM integration
  - Troubleshooting
- **Acceptance Criteria:**
  - [ ] Step-by-step instructions
  - [ ] Copy-paste commands
  - [ ] Tested on real cluster (if available)
- **Subagent:** Coder

---

## Milestone 6: Enhanced PoC Framework ✅

**Goal:** Complete visualization and reporting capabilities.
**Duration:** 1 week
**Subagents:** Coder, SQE

**Status:** ✅ **COMPLETED** (March-April 2026)

### What Was Delivered
- `PlotRegistry` with 5 plot types and `HTMLReportGenerator` with themed templates
- SBIR benchmark demo (`sbir_demo.py`) with HTML/JSON/Markdown report generation
- Curriculum config schema (`curriculum_schedule` field on `TrainingConfig`) with transition logging
- 390+ new tests across training, PDE, games, curriculum, modeling modules

### Epic 6.1: Visualization Module

**Story 6.1.1: Implement Interactive Plots**

- **Task:** Create `src/poc/visualization/plots.py` with Plotly
- **Plot Types:**
  - Training curves (loss, accuracy over time)
  - Hyperparameter importance (parallel coordinates)
  - Statistical comparison (box plots, violin plots)
- **Acceptance Criteria:**
  - [x] At least 5 plot types implemented
  - [x] Interactive HTML output
  - [x] Static PNG fallback
- **Subagent:** Coder

**Story 6.1.2: Implement HTML Report Generator**

- **Task:** Create `src/poc/visualization/reports.py`
- **Features:**
  - Jinja2 templated reports
  - Embedded plots
  - Metric tables
  - Comparison summaries
- **Acceptance Criteria:**
  - [x] Single HTML file output
  - [x] Offline viewable (embedded assets)
  - [x] Professional styling
- **Subagent:** Coder

### Epic 6.2: Activation of Curriculum Learning

**Story 6.2.1: Wire Curriculum Learning in Trainer**

- **Task:** Activate existing curriculum infrastructure
- **Location:** Model zoo + trainer integration
- **Acceptance Criteria:**
  - [x] Config enables curriculum mode
  - [x] Training progresses through curriculum stages
  - [x] Stage transitions logged
- **Subagent:** Coder

---

## Milestone 7: Real-World Validation

**Goal:** Validate against external benchmarks.
**Duration:** 2 weeks
**Subagents:** SQE, Reviewer

### Epic 7.1: Video Codec Benchmarking

**Story 7.1.1: Compare Against H.265/VP9**

- **Task:** Benchmark compression ratio vs. traditional codecs
- **Metrics:** BD-rate, PSNR, encoding time
- **Acceptance Criteria:**
  - [ ] Test on standard video datasets (Xiph.org)
  - [ ] Results documented with graphs
  - [ ] Identify competitive/non-competitive scenarios
- **Subagent:** SQE

### Epic 7.2: Go Engine Validation

**Story 7.2.1: Play Against GnuGo/KataGo**

- **Task:** Tournament against established engines
- **Acceptance Criteria:**
  - [ ] 100+ games played
  - [ ] Win rate computed with confidence intervals
  - [ ] Elo estimate calculated
- **Subagent:** SQE

---

## Milestone 8: Agent Orchestration Framework ✅

**Goal:** Multi-physics PDE solving with specialized sub-agents.
**Status:** ✅ **COMPLETED** (March-April 2026)

### What Was Delivered
- `src/agents/` — Multi-physics PDE agent orchestration framework
  - `OrchestratorAgent`, `CollocationAgent`, `DecompositionAgent`, `CouplingAgent`
  - `MetaAgent` with message passing between agents
  - Pydantic-validated `AgentConfig` schemas, `AgentRegistry`
- `src/research/` — SBIR benchmarking infrastructure
  - Classical solver baselines: FDM, Dorfler AMR, PINN
  - `PDEBenchmarkRunner` with JSON/Markdown reports + convergence rates
  - Comparison, reporter, and validator modules
- `src/engines/` — UCI chess engine integration (Stockfish evaluation)
- `src/tournament/` — Tournament management and Elo calculation
- `src/curriculum/` — Curriculum learning infrastructure
- SBIR proposal configs (Navy N252-088, DOE ASCR C59, NSF SBIR, AFWERX Open)

### Remaining (Sprint 3-4)
- [ ] Multi-field PDE coupling (fluid-structure interaction)
- [ ] Uncertainty quantification for PDE solutions

---

## Milestone 9: Production Hardening (v0.4.0) ⚠️ Partial

**Goal:** Production-ready deployment and extended benchmarking.
**Duration:** 3-4 weeks
**Priority:** P1

**Status:** ⚠️ **PARTIALLY COMPLETED** — SBIR demos and BaseTrainer done; ONNX production export and full PDE training loop pending

### What Was Delivered
- End-to-end `sbir_demo.py` with HTML/JSON/Markdown report generation (April 2026)
- `BaseTrainer[ConfigT]` with shared AMP, gradient clipping, LR scheduling, checkpoint save/load
- 39 BaseTrainer tests
- Loss balancing audit: fixed NaN/Inf propagation bugs in ReLoBRaLo/SoftAdapt, 96 property-based tests

### Epic 9.1: SBIR Benchmark Demos

**Story 9.1.1: End-to-End Demo Script**
- **Task:** Create `src/demos/sbir_demo.py` with benchmark visualization
- **Acceptance Criteria:**
  - [x] End-to-end demo runs in <5 minutes
  - [x] Compares AlphaGalerkin vs FDM, AMR, PINN baselines
  - [x] Generates HTML report with convergence plots

**Story 9.1.2: Multi-Field PDE Support**
- **Task:** Extend `ModelOutput` for vector field predictions
- **Acceptance Criteria:**
  - [ ] NavierStokes velocity+pressure output
  - [ ] Fluid-structure interaction coupling

### Epic 9.2: Deployment & Export

**Story 9.2.1: ONNX Production Export**
- **Task:** Complete ONNX export pipeline with dynamic shapes
- **Acceptance Criteria:**
  - [ ] Export works for Go (9x9, 13x13, 19x19) and Chess
  - [ ] Quantized INT8 model passes accuracy threshold

**Story 9.2.2: BaseTrainer Migration** ✅
- **Task:** Migrate `Trainer` and `OperatorTrainer` to `BaseTrainer` inheritance
- **Acceptance Criteria:**
  - [x] DRY — shared AMP, grad clip, LR scheduling in base class
  - [x] All existing tests pass

### Epic 9.3: PDE Training Loop

**Story 9.3.1: PDE Self-Play via MCTS**
- **Task:** Wire `BasisSelectionGame` + `MeshRefinementGame` to standard `Trainer`
- **Acceptance Criteria:**
  - [x] Config flag `training.game=pde_basis` works end-to-end
  - [x] Error reduction logged per episode
  - [ ] PettingZoo training loop for swarm games

---

## Milestone 10: Self-Hosted Neural Transcoder ⚠️ Phase 0+1 ✅

**Goal:** Realtime-decode-on-consumer-hardware target for `src/video_compression/`. Every later phase is conditional on Phase 0's headline measurement.
**Duration:** Phase 0+1 complete; Phases 2-4 estimated 4-8 weeks total
**Priority:** P1 (gates self-hosted streaming-compression product line)

**Status:** ⚠️ **Phase 0+1 COMPLETE** (2026-04-27 → 2026-04-30); Phases 2-4 NOT STARTED.

### Phase 0 — Codec Performance Benchmark Harness ✅ (PR #75 + follow-up)

GPU-primary perf harness in `src/video_compression/perf/`. **The gating measurement** for whether the project can target realtime decode on consumer hardware.

**What was delivered:**

- `PerfBenchmark(BaseExecutable)` with per-profile `cuda:N` device pinning so a single sweep covers both cards of the reference dual-GPU rig (RTX 5060 Ti 16 GB at `cuda:0` + RTX 5060 8 GB at `cuda:1`).
- `PerfBenchmarkConfig` Pydantic schema with **zero hardcoded values** — every measurement-affecting knob (resolution / batch / phase / warmup / repeats / tolerance / track-VRAM / pattern / data-seed) is a validated field with bounds.
- `BaselineRegistry` with explicit JSON schema versioning, `extra="ignore"` forward-compat, and `_migrate_baseline_document` hook (unversioned-to-v1 migration shipped).
- `BenchmarkSubject` Protocol — runtime-agnostic. Phase-1+ runtime backends (ONNX Runtime, TensorRT, FP16, `torch.compile`) drop in without touching the benchmark loop. Extended docstring includes a runnable Phase-1 example.
- Three YAML configs: `config/perf/smoke.yaml` (CPU CI), `cuda0_headline.yaml` (single-card), `default.yaml` (dual-card sweep).
- CLI `scripts/benchmark_codec.py` with `run` / `record-baseline` / `diff` subcommands; structured `structlog` events bound to `benchmark_id` + `cell_key`.
- **126 passing tests + 4 skipped (CPU-only paths skipping correctly on CUDA host); per-module coverage 98.42%** (gate at 85% via `.github/workflows/codec-perf-coverage.yml`).

### Phase 1 — Decoder Runtime Backends ✅ (2026-04-30)

Four decoder runtime backends implemented in `src/video_compression/runtime/`:

- **Story 10.1.1:** `PyTorchCompiledRuntime` — `torch.compile` with inductor, CUDA graphs via `reduce-overhead` ✅
- **Story 10.1.2:** `ONNXDecoderRuntime` — In-memory ONNX export + `CUDAExecutionProvider` ✅
- **Story 10.1.3:** `TensorRTRuntime` — `torch_tensorrt.compile` with Dynamo IR, FP16 via `enabled_precisions` ✅
- **Story 10.1.4:** FP16/BF16 activation — `NotImplementedError` gates removed, `_runtime_name_for_profile()` dispatch ✅

**CUDA environment:** PyTorch 2.11.0+cu126, torch_tensorrt 2.11.0+cu126, GTX 1660 Ti.
**Full regression:** 244 passed, 17 skipped, 0 failed.

### Phase 2 — Pretrained Model Zoo (~2 weeks)

Train + ship 8 model checkpoints across the declared λ rate-distortion points. Each checkpoint becomes a `RuntimeProfile` in the Phase 0 sweep.

### Phase 3 — MCTS Rate Control (~3 weeks)

Resolves the documented "Known Issue" in `CLAUDE.md` (rate-control tests skipped pending trained MCTS model). Wires `src/video_compression/mcts/` into the codec, enabling GOP-level bit allocation.

### Phase 4 — FFmpeg Bridge → Library Daemon → Plugins (~2-3 weeks)

External integration: FFmpeg encoder/decoder shim → background daemon for media-server consumption → Jellyfin / Plex plugins. Phase 4 only ships if Phases 1-3 produce a competitive codec.

---

## Implementation Priority Matrix

| Priority | Milestone | Estimated Effort | Dependencies |
|----------|-----------|------------------|--------------|
| **P0** | M1: CI/CD | 1-2 days | None |
| **P0** | M2: Critical Bugs | 2-3 days | None |
| **P1** | M3: Multi-Game | 1 week | M2 |
| **P1** | M5: Distributed Validation | 3-4 days | M1 |
| **P2** | M4: PDE Integration | 3-4 days | M1 |
| **P2** | M6: Enhanced PoC | 1 week | M1 |
| **P3** | M7: Real-World Validation | 2 weeks | M3, M5 |

---

## Quick Wins (Immediately Actionable)

These can be completed without blocking dependencies:

1. **Add GitHub Actions CI** (~2 hours)
   - Create `.github/workflows/ci.yml`
   - Run lint + type check + tests on PR

2. **Add pre-commit hooks** (~30 min)
   - Create `.pre-commit-config.yaml`
   - Document in README

3. **Fix hyperprior TODOs** (~4 hours)
   - `scripts/encode_video.py:261`
   - `scripts/decode_video.py:420`

4. **Enable physics loss config** (~1 hour)
   - Add `training.physics_informed: bool` to config
   - Wire into trainer

5. **Create distributed training guide** (~2 hours)
   - Document existing CLI commands
   - Add troubleshooting section

---

## Success Metrics

| Module | Metric | Current | Target | Measurement |
|--------|--------|---------|--------|-------------|
| CI/CD | Pipeline exists | ✅ Yes (8-stage) | Yes | `.github/workflows/ci.yml` |
| Test Coverage | Overall / Chess | ✅ 85% / 97% | >80% | `--cov-fail-under=85` |
| Multi-Game | Games implemented | ✅ 2 (Go, Chess) | 2+ | Registry count |
| Engine Eval | Stockfish integration | ✅ Yes | Yes | `_run_engine_evaluation()` |
| Agent Framework | Agents implemented | ✅ 7 agents | 1+ | Agent count |
| SBIR Benchmarks | Benchmark runner | ✅ Yes (HTML reports) | Yes | PDEBenchmarkRunner + sbir_demo.py |
| Distributed | Multi-node validation | ⚠️ Partial (35 tests) | Yes | Integration test passes |
| PDE | Training integration | ✅ Yes | Yes | `config/train_pde.yaml` works |
| Video Compression | Hyperprior complete | No | Yes | No TODO comments |

---

## Appendix A: Critical TODOs in Codebase

| File | Line | Description | Priority | Status |
|------|------|-------------|----------|--------|
| `scripts/encode_video.py` | 261 | Add hyperprior z encoding | High | Open |
| `scripts/decode_video.py` | 420 | Properly decode hyperprior z_data | High | Open |
| ~~`src/training/self_play.py`~~ | ~~419~~ | ~~True parallel generation~~ | ~~Medium~~ | ✅ Fixed (2026-02-04) |
| `tests/games/test_sgf.py` | - | Variation parsing (skipped) | Medium | Open |

---

## Appendix B: Skipped Tests Summary

| Module | Tests Skipped | Reason | Resolution Path |
|--------|---------------|--------|-----------------|
| Vertex Launcher | 5 | Complex SDK mocking | Improve mock strategy |
| SGF Variation | 1 | Not implemented | Complete parser |
| CLI Module | 2 | Discovery issues | Fix import paths |

---

## Appendix C: Subagent Delegation Map

```
Planner          → Architecture decisions, module design
├── M3.2.2: Transfer learning experiment design
├── M4: PDE integration strategy
└── Overall milestone sequencing

SQE              → Test creation, validation, benchmarking
├── M1.1.2: Coverage reporting
├── M3.1.3: Chess symmetry tests
├── M5.1.1: Multi-process integration tests
├── M7: All real-world validation
└── Quality gates for each milestone

Coder            → Implementation of features
├── M1: CI/CD setup
├── M2: Bug fixes
├── M3.1: Chess implementation
├── M4.1: PDE integration code
├── M6.1: Visualization module
└── All documentation

Reviewer         → PR reviews, code quality
├── M2: Critical bug fix reviews
├── M4: PDE integration review
└── Pre-merge validation

Orchestrator     → Integration, deployment, automation
├── M1.1.1: GitHub Actions setup
├── M5: Distributed validation coordination
└── Release management
```

---

*Last Updated: 2026-04-10*
*Version: 3.0.0*
*Author: Claude Code Agent Investigation*


---

## Milestone 11: Codec Model Zoo (R-D Lagrangian Sweep) ?? Phase 2-B ?

**Goal:** Schedule the 8-point R-D Lagrangian sweep across the dual-GPU rig
(`cuda:0=RTX 5060 Ti 16 GiB` + `cuda:1=RTX 5060 8 GiB`) and produce a
reproducible BD-rate curve.
**Duration:** Phase 2-B complete; Phases 2-C through 2-G estimated 3-5 weeks total
**Priority:** P1 (gates the headline R-D number for the self-hosted transcoder)

### Phase 2-B: Schemas, Manifest, Storage, Planner ? COMPLETE (2026-05-01)

Shipped in src/video_compression/zoo/:

- `ModelZooEntryConfig` / `ModelZooManifestConfig` / `OptimizerConfig` /
  `SchedulerConfig` � Pydantic v2, zero hardcoded values, schema-versioned.
- `load_manifest` / `save_manifest` � JSON+YAML by suffix, forward-compat
  migration via `_migrate_manifest_document`.
- `scan_devices` / `assign_devices` � four strategies
  (`VRAM_AWARE` / `ROUND_ROBIN` / `SINGLE_DEVICE` / `MANUAL`).
- `VideoCodecZoo` filesystem registry; GCS backend gated for Phase 2-D.
- `config/video_compression/zoo/lambda_grid.yaml` ships the 8-point grid.
- **100% line + branch coverage** across all five modules; 68 tests; mypy
  `--strict` and `ruff` clean.

### Phase 2-C: ZooTrainer (Per-Entry Composition) � NEXT

- `src/video_compression/training/zoo_trainer.py::ZooTrainer` composes
  `VideoCompressionTrainer` per-entry with fixed-? + AMP + grad-clip +
  warmup wired from `ModelZooEntryConfig`.
- Warm-start via `parent_entry_id`: load checkpoint from parent entry,
  re-init optimizer.
- Structured logging bound to `entry_id` / `lambda_rd` / `device`.
- Acceptance: an 8 GiB-friendly path on `cuda:1` (FP16/BF16 + grad
  accumulation) trains a single entry to its target_psnr_db.

### Phase 2-D: Subprocess Sweep Driver

- `src/video_compression/zoo/sweep.py::ZooSweep(BaseExecutable)`.
- Subprocess-per-device with `CUDA_VISIBLE_DEVICES` pinning so each entry
  sees exactly one GPU and PyTorch never has to multiplex.
- Manifest-hash resumability � re-running with the same manifest skips
  entries whose `checkpoint.pt` + `entry.json` already match the
  expected `hash_key`.

### Phase 2-E: Perf Harness Integration

- `RuntimeProfile.zoo_entry_id: str | None` � links a perf cell back to
  the entry that produced its weights.
- `CodecForwardSubject.from_zoo_entry(...)` � perf benchmark loads
  weights through the zoo registry.
- `config/perf/zoo_grid.yaml` � runs all eight zoo entries through the
  Phase 0 perf harness.

### Phase 2-F: BD-rate Validation

- `src/video_compression/metrics/rd_curves.py` ingests the per-entry
  `metrics.json` files and produces a BD-rate report.
- Nightly CI workflow `phase2-zoo-validation.yml` rebuilds the curve and
  diffs against the committed baseline.

### Phase 2-G: Planner Quality (Opportunistic)

- The current `VRAM_AWARE` planner is correct but conservative on
  asymmetric rigs: when one large entry takes the bigger card, subsequent
  entries that exceed the smaller card's headroom over-commit on the
  bigger card rather than using the smaller card's free VRAM. A
  `two-phase` strategy (greedy by descending VRAM, then load-balance by
  total headroom) would push more work onto `cuda:1`. Defer until
  Phase 2-D shows the over-commit cost in wall-clock.
