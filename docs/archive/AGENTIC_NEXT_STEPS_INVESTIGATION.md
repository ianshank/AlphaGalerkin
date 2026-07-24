# AlphaGalerkin Agentic Next Steps Investigation

> **Investigation Date:** 2026-02-04
> **Branch:** `claude/investigate-agentic-next-steps-Wsbgj`
> **Methodology:** Universal Dev Agent framework with Planner/SQE/Coder/Reviewer/Orchestrator subagent delegation
> **Previous Plan:** `docs/NEXT_STEPS_PLAN.md` (2026-02-01)
> **Last Updated:** 2026-04-25 — Learned PDE Evaluator wired (PR #54). All P0 and most P1/P2 items now resolved

---

## Executive Summary

AlphaGalerkin has matured significantly since the last investigation (2026-02-01). The project now contains **189 source files (~55K LOC)** and **131 test modules (~40K LOC)** with a comprehensive CI/CD pipeline. Several P0 items from the previous plan have been completed:

| Previous Plan Item | Status | Evidence |
|--------------------|--------|----------|
| CI/CD Pipeline | **DONE** | `.github/workflows/ci.yml` (6 stages, Python 3.10/3.11/3.12 matrix) |
| Pre-commit Hooks | **DONE** | `.pre-commit-config.yaml` (ruff, mypy, bandit, commitizen) |
| Chess Implementation | **DONE** | `src/games/chess.py` (1,173 LOC, full rules, 119-plane encoding) |
| Test Coverage Reporting | **DONE** | Codecov integration, 60% threshold enforced |
| Video Compression Hyperprior | **DONE** | Proper z_bitstream encoding/decoding |
| Learned PDE Evaluator | **DONE (2026-04-25)** | `AlphaGalerkinConfig.evaluator="trained"` re-enabled, `FNetEvaluator` wired through `_build_trained_evaluator` with on-instance caching, GPU-primary default with CPU fallback. PR #54. |

This investigation identifies the **remaining gaps** and provides an updated, prioritized roadmap.

---

## Current Architecture Health

```
Overall Assessment: BETA - Feature-rich with strong integration (updated 2026-04-10)

Strengths:
  - Modular architecture with clear separation of concerns
  - Pydantic validation on all configuration boundaries
  - Structured logging (structlog) throughout
  - Comprehensive test suite with markers (slow/e2e/video/gpu) — 5,100+ tests
  - Type-checked (mypy strict) with CI enforcement
  - Multiple deployment targets (local, Vertex AI, ONNX, HuggingFace)
  - 8-stage CI/CD pipeline with 85% coverage gates
  - PDE-training and curriculum-trainer integration wired and tested
  - E2E dashboard (Gradio) with 203 tests at 89% coverage

Weaknesses (remaining):
  - No end-to-end training validation on real hardware at scale
  - No release/packaging workflow
  - Some Vertex launcher tests still skipped
```

---

## Gap Analysis: Production Code Issues

### P0: Critical Stubs Blocking Production — ✅ ALL RESOLVED

#### 1. Emergency Checkpoint on Vertex AI Preemption ✅
- **File:** `src/vertex/entrypoint.py`
- **Status:** ✅ **RESOLVED** (2026-02-04)
- **Resolution:** Emergency checkpoint wired with signal-based preemption detection
- **AC:**
  - [x] Saves model state_dict, optimizer state, step count on SIGTERM
  - [x] Tested with mock signal delivery
  - [x] Resume from emergency checkpoint verified

#### 2. GTP Protocol Player Assignment ✅
- **File:** `src/tools/gtp.py`
- **Status:** ✅ **RESOLVED** (2026-02-04)
- **Resolution:** Player assignment implemented for both black and white
- **AC:**
  - [x] `genmove black` sets correct player
  - [x] `genmove white` sets correct player
  - [x] GTP protocol smoke test passes

#### 3. Parallel Self-Play Generation ✅
- **File:** `src/training/self_play.py`
- **Status:** ✅ **RESOLVED** (2026-02-04)
- **Resolution:** Parallel self-play generation implemented
- **AC:**
  - [x] `n_workers` config parameter controls parallelism
  - [x] Linear speedup up to CPU core count
  - [x] No shared state corruption (each worker gets model copy)
  - [x] Graceful fallback to sequential when `n_workers=1`

### P1: High-Priority Feature Gaps — ✅ ALL RESOLVED

#### 4. Physics Loss Integration in Trainer ✅
- **Status:** ✅ **RESOLVED** (2026-04-02)
- **Resolution:** `CombinedAlphaGalerkinPhysicsLoss` passes `lbb_constant`, `action_mask`, `model` to trainer. `PhysicsLoss` with Laplacian regularization via autodiff. 52 comprehensive tests.
- **AC:**
  - [x] `training.physics_informed: bool` config option activates physics loss
  - [x] Gradient flow verified (no NaN/inf)
  - [x] Physics loss weight appears in training logs

#### 5. PDE-MCTS Integration ✅
- **Status:** ✅ **RESOLVED** (2026-04-02)
- **Resolution:** `PDEGameInterface` bridges PDE games to `GameInterface` for `GameRegistry` registration. `pde_basis` and `pde_mesh` registered via `src/pde/register_games.py`. `config/train_pde.yaml` created. 40 integration tests.
- **AC:**
  - [x] MCTS can play BasisSelectionGame
  - [x] Policy/value heads produce valid outputs for PDE state tensors
  - [x] Error reduction tracked per episode
  - [x] Integration test in `tests/integration/test_pde_mcts.py`

#### 6. Curriculum Learning Activation ✅
- **Status:** ✅ **RESOLVED** (2026-04-02)
- **Resolution:** `curriculum_schedule` field on `TrainingConfig` with transition logging. `src/curriculum/` module wired into trainer loop.
- **AC:**
  - [x] Config option `training.curriculum.enabled: true` activates curriculum
  - [x] Training progresses through board sizes (e.g., 9x9 -> 13x13 -> 19x19)
  - [x] Stage transitions logged with structlog
  - [x] Checkpoint preserves curriculum state

### P2: Quality & Robustness — ✅ MOSTLY RESOLVED

#### 7. Test Coverage Improvement (60% -> 85%) ✅
- **Previous:** 60% threshold in CI
- **Current:** ✅ **85% overall** coverage gate enforced (as of March 2026)
- **Resolution:** 390+ new tests across training, PDE, games, curriculum, modeling. Per-module gates: modeling 85%, training 85%, research 85%, pde 75%, games 80%, distributed 60%, physics 75%. Coverage sprint added 115 new tests (statistics, tuning, ONNX).
- **AC:**
  - [x] Coverage report shows 85%+ overall
  - [x] CI threshold updated to 85%
  - [x] No decrease in existing coverage

#### 8. Skipped Test Resolution ⚠️ Partial
- **Vertex Launcher:** 5 tests still skipped (SDK mocking)
- **SGF Variation:** 1 test still skipped (parser incomplete)
- **CLI Module:** Resolved
- **AC:**
  - [ ] All skipped tests enabled and passing (6 remain)
  - [x] No external API calls in test suite

---

## Gap Analysis: Infrastructure & DevOps

### P1: Release & Distribution

#### 9. PyPI Publishing Workflow
- **Issue:** No automated release pipeline despite `pyproject.toml` being properly configured
- **Subagent:** Orchestrator
- **AC:**
  - [ ] `.github/workflows/release.yml` triggers on Git tags
  - [ ] Builds wheel and sdist
  - [ ] Publishes to PyPI (or TestPyPI initially)
  - [ ] Version managed via commitizen/bumpversion

#### 10. Dependency Management Automation
- **Issue:** No dependabot or Renovate configuration
- **Subagent:** Orchestrator
- **AC:**
  - [ ] `.github/dependabot.yml` configured for pip ecosystem
  - [ ] Weekly update schedule
  - [ ] Auto-merge for patch updates after CI passes

### P2: Security & Compliance

#### 11. Container Security Scanning
- **Issue:** `docker/Dockerfile.vertex` builds from NVIDIA NGC base but no vulnerability scanning
- **Subagent:** Orchestrator
- **AC:**
  - [ ] Trivy scan in CI for Docker images
  - [ ] No critical/high vulnerabilities in base image
  - [ ] Scan results as PR comment

#### 12. GitHub Security Configuration
- **Issue:** No CODEOWNERS, no security policy, no issue/PR templates
- **Subagent:** Orchestrator
- **AC:**
  - [ ] `CODEOWNERS` file mapping modules to owners
  - [ ] `SECURITY.md` with vulnerability reporting process
  - [ ] `.github/ISSUE_TEMPLATE/` with bug report and feature request templates
  - [ ] `.github/PULL_REQUEST_TEMPLATE.md`

---

## Gap Analysis: Integration & Validation

### P1: Cross-Module Integration Testing

#### 13. Training Pipeline End-to-End Validation
- **Issue:** Individual modules tested but full pipeline integration is thin
- **Required tests:**
  - Self-play -> Replay Buffer -> Trainer -> Checkpoint -> Resume
  - Multi-game training (Go config -> Chess config switch)
  - Curriculum progression through board sizes
  - Distributed training with mocked NCCL (4 processes)
- **Subagent:** SQE
- **AC:**
  - [ ] `tests/integration/test_full_pipeline.py` covers self-play to checkpoint
  - [ ] `tests/integration/test_multigame_training.py` validates game switching
  - [ ] `tests/integration/test_curriculum_progression.py` validates stage transitions
  - [ ] All integration tests pass in CI

#### 14. ONNX Export End-to-End Validation
- **Issue:** Export code exists but no integration test verifying PyTorch -> ONNX -> Inference roundtrip
- **Subagent:** SQE
- **AC:**
  - [ ] `tests/integration/test_onnx_e2e.py` creates model, exports, loads in ONNX Runtime
  - [ ] Output divergence < 1e-5 for policy and value heads
  - [ ] Dynamic shapes work (batch=1 and batch=32)

### P2: Real-World Benchmarking

#### 15. Go Engine Tournament
- **Goal:** Validate AlphaGalerkin's game-playing strength
- **Subagent:** SQE
- **AC:**
  - [ ] 100+ games against GnuGo via GTP protocol
  - [ ] Win rate with 95% confidence intervals
  - [ ] Elo estimate documented
  - [ ] Results in `docs/benchmarks/go_tournament.md`

#### 16. Video Codec Benchmarking
- **Goal:** Compare against H.265/VP9 on standard test sequences
- **Subagent:** SQE
- **AC:**
  - [ ] BD-rate curves computed on Xiph.org test sequences
  - [ ] PSNR, SSIM, MS-SSIM per quality point
  - [ ] Encoding/decoding speed comparison
  - [ ] Results in `docs/benchmarks/video_compression.md`

---

## Updated Implementation Roadmap

### Milestone 1: Critical Fixes (P0)
**Goal:** Fix production-blocking stubs
**Delegation:** Coder (implementation) + SQE (tests) + Reviewer (PR review)

| Story | File(s) | Subagent | Effort |
|-------|---------|----------|--------|
| Fix emergency checkpoint | `src/vertex/entrypoint.py` | Coder | S |
| Fix GTP player assignment | `src/tools/gtp.py` | Coder | XS |
| Implement parallel self-play | `src/training/self_play.py` | Coder | M |
| Tests for all fixes | `tests/` | SQE | M |

### Milestone 2: Integration Wiring (P1)
**Goal:** Connect existing modules that are built but disconnected
**Delegation:** Coder + Planner (architecture review)

| Story | Integration Point | Subagent | Effort |
|-------|-------------------|----------|--------|
| Physics loss in trainer | `physics_loss.py` -> `trainer.py` | Coder | M |
| PDE-MCTS connection | `pde/games/` -> `mcts/` | Coder | L |
| Curriculum in trainer | `curriculum/` -> `trainer.py` | Coder | M |
| Integration test suite | `tests/integration/` | SQE | L |

### Milestone 3: Quality & Coverage (P2)
**Goal:** Harden test suite and improve coverage
**Delegation:** SQE (primary) + Coder (fixes)

| Story | Target | Subagent | Effort |
|-------|--------|----------|--------|
| Raise coverage to 80% | Core modules | SQE | L |
| Fix 8 skipped tests | vertex, sgf, cli | SQE | M |
| ONNX roundtrip test | `tests/integration/` | SQE | M |
| Full pipeline test | `tests/integration/` | SQE | L |

### Milestone 4: DevOps Maturity (P2)
**Goal:** Production-ready release and security infrastructure
**Delegation:** Orchestrator

| Story | Deliverable | Subagent | Effort |
|-------|-------------|----------|--------|
| PyPI release workflow | `.github/workflows/release.yml` | Orchestrator | M |
| Dependabot config | `.github/dependabot.yml` | Orchestrator | XS |
| Container scanning | Trivy in CI | Orchestrator | S |
| GitHub templates | CODEOWNERS, issue/PR templates | Orchestrator | S |

### Milestone 5: Real-World Validation (P3)
**Goal:** External benchmarks and tournament results
**Delegation:** SQE + Reviewer

| Story | Deliverable | Subagent | Effort |
|-------|-------------|----------|--------|
| Go tournament vs GnuGo | Elo estimate, win rate | SQE | L |
| Video codec benchmarks | BD-rate curves | SQE | L |
| Zero-shot transfer ablation | Systematic board size study | SQE | M |
| Results documentation | `docs/benchmarks/` | Coder | M |

---

## Effort Key

| Symbol | Meaning |
|--------|---------|
| XS | < 1 hour |
| S | 1-4 hours |
| M | 4-8 hours |
| L | 1-3 days |
| XL | 1+ week |

---

## Dependency Graph

```
Milestone 1 (Critical Fixes)
    |
    v
Milestone 2 (Integration Wiring)
    |        \
    v         v
Milestone 3   Milestone 4
(Quality)     (DevOps)
    |        /
    v       v
Milestone 5 (Real-World Validation)
```

Milestones 3 and 4 can run in parallel. Milestone 5 depends on both.

---

## Subagent Delegation Summary

```
Planner
├── M2: Architecture review for PDE-MCTS integration
└── Overall milestone sequencing and dependency management

SQE (Software Quality Engineer)
├── M1: Test coverage for P0 fixes
├── M2: Integration test suite creation
├── M3: Coverage improvement, skipped test resolution
├── M5: Tournament execution, benchmarking
└── Quality gates for each milestone

Coder
├── M1: Fix emergency checkpoint, GTP, parallel self-play
├── M2: Physics loss wiring, PDE-MCTS connection, curriculum activation
├── M4: DevOps templates (if needed)
└── M5: Documentation of results

Reviewer
├── M1: PR review for critical fixes
├── M2: Integration architecture review
└── Pre-merge validation for all milestones

Orchestrator
├── M4: CI/CD workflows (release, dependabot, scanning)
└── Release management and coordination
```

---

## Success Metrics

| Module | Metric | Current | Target | Status |
|--------|--------|---------|--------|--------|
| Vertex AI | Emergency checkpoint | ✅ Functional | Functional | ✅ Done |
| GTP | Player assignment | ✅ Working | Working | ✅ Done |
| Self-play | Parallelism | ✅ N workers | N workers | ✅ Done |
| Physics | Loss integration | ✅ Wired | Wired | ✅ Done |
| PDE-MCTS | Integration | ✅ Working | Working | ✅ Done |
| Curriculum | Activation | ✅ Wired | Wired | ✅ Done |
| Coverage | Test coverage | ✅ 85% | 80% | ✅ Exceeded |
| Skipped tests | Count | 6 | 0 | ⚠️ Partial |
| Release | PyPI workflow | None | Automated | ❌ Pending |
| Security | Container scan | None | Automated | ❌ Pending |

---

## Appendix A: Files Requiring Immediate Attention

| File | Line | Issue | Priority | Status |
|------|------|-------|----------|--------|
| ~~`src/vertex/entrypoint.py`~~ | ~~451~~ | ~~Emergency checkpoint is `pass`~~ | ~~P0~~ | ✅ Fixed (2026-02-04) |
| ~~`src/tools/gtp.py`~~ | ~~574-576~~ | ~~Player assignment is `pass`~~ | ~~P0~~ | ✅ Fixed (2026-02-04) |
| ~~`src/training/self_play.py`~~ | ~~417~~ | ~~Parallel generation TODO~~ | ~~P0~~ | ✅ Fixed (2026-02-04) |
| ~~`src/experiments/physics_model.py`~~ | ~~254~~ | ~~Physics weight branch is `pass`~~ | ~~P1~~ | ✅ Fixed (2026-04-02) |
| `src/modeling/operator.py` | 87 | Only 2 backends supported | P2 | Open |
| `src/games/chess.py` | 898 | Simplified bishop color check | P3 | Open |

## Appendix B: Completed Items from Previous Plan

| Item | Completed Date | PR/Commit |
|------|---------------|-----------|
| GitHub Actions CI Pipeline | 2026-02-01 | ci.yml with 6 stages |
| Pre-commit Hooks | 2026-02-01 | ruff, mypy, bandit, commitizen |
| Chess Implementation | 2026-02-01 | Full rules, 119-plane encoding |
| Test Coverage Reporting | 2026-02-01 | Codecov, 60% threshold |
| Video Hyperprior Fix | 2026-02-01 | Proper z_bitstream encoding |
| Ruff/MyPy CI | 2026-02-01 | Lint + type check in CI |

## Appendix C: Module Line Counts (Top 10)

> **Note:** LOC counts from Feb 2026 investigation. Project has grown significantly since (25 modules, 5,100+ tests).

| Module | LOC (Feb 2026) | Files | Status (Apr 2026) |
|--------|-----|-------|--------|
| video_compression/ | 15,927+ | 30+ | Complete |
| training/ | 8,800+ | 14 | Complete, integration wired ✅ |
| vertex/ | 5,800 | 10 | Complete, checkpoint fixed ✅ |
| poc/ | 5,081 | 12 | Complete, visualization added ✅ |
| modeling/ | 3,654 | 10 | Complete |
| pde/ | 3,500 | 7 | Complete, MCTS integrated ✅ |
| distributed/ | 2,986 | 6 | Complete, 35 new tests ✅ |
| games/ | 2,856 | 6 | Complete (Go + Chess) |
| deployment/ | 2,147 | 5 | Complete |
| templates/ | 2,063 | 5 | Complete |

---

*Last Updated: 2026-04-10*
*Version: 3.0.0*
*Author: Claude Code Agent Investigation*
*Previous Version: docs/NEXT_STEPS_PLAN.md (v1.0.0, 2026-02-01)*
