# AlphaGalerkin Agentic Next Steps Investigation

> **Investigation Date:** 2026-02-04
> **Branch:** `claude/investigate-agentic-next-steps-Wsbgj`
> **Methodology:** Universal Dev Agent framework with Planner/SQE/Coder/Reviewer/Orchestrator subagent delegation
> **Previous Plan:** `docs/NEXT_STEPS_PLAN.md` (2026-02-01)

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

This investigation identifies the **remaining gaps** and provides an updated, prioritized roadmap.

---

## Current Architecture Health

```
Overall Assessment: ALPHA - Feature-rich, needs integration hardening

Strengths:
  - Modular architecture with clear separation of concerns
  - Pydantic validation on all configuration boundaries
  - Structured logging (structlog) throughout
  - Comprehensive test suite with markers (slow/e2e/video/gpu)
  - Type-checked (mypy strict) with CI enforcement
  - Multiple deployment targets (local, Vertex AI, ONNX, HuggingFace)

Weaknesses:
  - Several production code stubs (emergency checkpoint, GTP player, parallel self-play)
  - No end-to-end training validation on real hardware at scale
  - Cross-module integration untested (PDE + training, curriculum + trainer)
  - Coverage threshold at 60% (below 80% target)
  - No release/packaging workflow
```

---

## Gap Analysis: Production Code Issues

### P0: Critical Stubs Blocking Production

#### 1. Emergency Checkpoint on Vertex AI Preemption
- **File:** `src/vertex/entrypoint.py:451-453`
- **Issue:** `emergency_checkpoint()` is a no-op `pass` statement
- **Impact:** All training progress lost when spot instances are preempted on Vertex AI
- **Fix:** Wire `checkpoint_manager.save()` with current trainer state
- **Subagent:** Coder
- **AC:**
  - [ ] Saves model state_dict, optimizer state, step count on SIGTERM
  - [ ] Tested with mock signal delivery
  - [ ] Resume from emergency checkpoint verified

#### 2. GTP Protocol Player Assignment
- **File:** `src/tools/gtp.py:574,576`
- **Issue:** `expected_player` is never assigned; both branches are `pass`
- **Impact:** `genmove` command uses uninitialized variable, crashes at line 579
- **Fix:** Set `expected_player = 1` for black, `expected_player = 2` for white
- **Subagent:** Coder
- **AC:**
  - [ ] `genmove black` sets correct player
  - [ ] `genmove white` sets correct player
  - [ ] GTP protocol smoke test passes

#### 3. Parallel Self-Play Generation
- **File:** `src/training/self_play.py:417`
- **Issue:** TODO comment; no multiprocessing despite parallel API surface
- **Impact:** Training throughput bottleneck; single-threaded game generation
- **Fix:** Implement `multiprocessing.Pool` or `torch.multiprocessing` workers
- **Subagent:** Coder
- **AC:**
  - [ ] `n_workers` config parameter controls parallelism
  - [ ] Linear speedup up to CPU core count
  - [ ] No shared state corruption (each worker gets model copy)
  - [ ] Graceful fallback to sequential when `n_workers=1`

### P1: High-Priority Feature Gaps

#### 4. Physics Loss Integration in Trainer
- **File:** `src/experiments/physics_model.py:254-256`
- **Issue:** `physics_weight > 0` branch is a `pass` — physics regularization never computed
- **Context:** `CombinedAlphaGalerkinPhysicsLoss` exists in `src/training/physics_loss.py` but isn't wired into the main trainer
- **Subagent:** Coder
- **AC:**
  - [ ] `training.physics_informed: bool` config option activates physics loss
  - [ ] Gradient flow verified (no NaN/inf)
  - [ ] Physics loss weight appears in training logs

#### 5. PDE-MCTS Integration
- **Context:** `src/pde/games/basis_selection.py` implements `BasisSelectionGame` compatible with `GameInterface`, and `src/mcts/` has full MCTS. But they've never been connected.
- **Subagent:** Coder + SQE
- **AC:**
  - [ ] MCTS can play BasisSelectionGame
  - [ ] Policy/value heads produce valid outputs for PDE state tensors
  - [ ] Error reduction tracked per episode
  - [ ] Integration test in `tests/integration/test_pde_mcts.py`

#### 6. Curriculum Learning Activation
- **Context:** `src/curriculum/` has scheduler, manager, stage definitions. `src/training/curriculum.py` has `BoardSizeCurriculum`. But these aren't wired into the trainer loop.
- **Subagent:** Coder
- **AC:**
  - [ ] Config option `training.curriculum.enabled: true` activates curriculum
  - [ ] Training progresses through board sizes (e.g., 9x9 -> 13x13 -> 19x19)
  - [ ] Stage transitions logged with structlog
  - [ ] Checkpoint preserves curriculum state

### P2: Quality & Robustness

#### 7. Test Coverage Improvement (60% -> 80%)
- **Current:** 60% threshold in CI
- **Target:** 80% on core logic (modeling, training, mcts, games)
- **Key uncovered areas to investigate:**
  - `src/training/trainer.py` (main training loop)
  - `src/mcts/search.py` (tree search core)
  - `src/distributed/trainer.py` (DDP training)
- **Subagent:** SQE
- **AC:**
  - [ ] Coverage report shows 80%+ on `src/modeling/`, `src/training/`, `src/mcts/`, `src/games/`
  - [ ] CI threshold updated to 80%
  - [ ] No decrease in existing coverage

#### 8. Skipped Test Resolution
- **Vertex Launcher:** 5 tests skipped (SDK mocking)
- **SGF Variation:** 1 test skipped (parser incomplete)
- **CLI Module:** 2 tests skipped (import path issues)
- **Subagent:** SQE
- **AC:**
  - [ ] All 8 skipped tests enabled and passing
  - [ ] No external API calls in test suite

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

| Module | Metric | Current | Target | Measurement |
|--------|--------|---------|--------|-------------|
| Vertex AI | Emergency checkpoint | No-op | Functional | Signal handler saves state |
| GTP | Player assignment | Broken | Working | `genmove` succeeds |
| Self-play | Parallelism | Sequential | N workers | Throughput benchmark |
| Physics | Loss integration | Disconnected | Wired | Config activates physics loss |
| PDE-MCTS | Integration | None | Working | MCTS plays BasisSelectionGame |
| Curriculum | Activation | Disconnected | Wired | Stage transitions logged |
| Coverage | Test coverage | 60% | 80% | pytest-cov report |
| Skipped tests | Count | 8 | 0 | pytest report |
| Release | PyPI workflow | None | Automated | Tag triggers publish |
| Security | Container scan | None | Automated | Trivy in CI |

---

## Appendix A: Files Requiring Immediate Attention

| File | Line | Issue | Priority |
|------|------|-------|----------|
| `src/vertex/entrypoint.py` | 451 | Emergency checkpoint is `pass` | P0 |
| `src/tools/gtp.py` | 574-576 | Player assignment is `pass` | P0 |
| `src/training/self_play.py` | 417 | Parallel generation TODO | P0 |
| `src/experiments/physics_model.py` | 254 | Physics weight branch is `pass` | P1 |
| `src/modeling/operator.py` | 87 | Only 2 backends supported | P2 |
| `src/games/chess.py` | 898 | Simplified bishop color check | P3 |

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

| Module | LOC | Files | Status |
|--------|-----|-------|--------|
| video_compression/ | 15,927+ | 30+ | Complete |
| training/ | 8,800+ | 14 | Complete, needs integration |
| vertex/ | 5,800 | 10 | Complete, stub in checkpoint |
| poc/ | 5,081 | 12 | Complete |
| modeling/ | 3,654 | 10 | Complete |
| pde/ | 3,500 | 7 | Complete, needs MCTS integration |
| distributed/ | 2,986 | 6 | Complete, needs scale test |
| games/ | 2,856 | 6 | Complete (Go + Chess) |
| deployment/ | 2,147 | 5 | Complete, needs e2e test |
| templates/ | 2,063 | 5 | Complete |

---

*Last Updated: 2026-02-04*
*Version: 2.0.0*
*Author: Claude Code Agent Investigation*
*Previous Version: docs/NEXT_STEPS_PLAN.md (v1.0.0, 2026-02-01)*
