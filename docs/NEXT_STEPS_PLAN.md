# AlphaGalerkin Next Steps Plan

> **Investigation Date:** 2026-02-01
> **Status:** Active Planning
> **Methodology:** Universal Dev Agent with Agentic Sub-Tasks

---

## Executive Summary

Based on comprehensive codebase exploration, AlphaGalerkin is a **mature v2.0 implementation** with excellent mathematical foundations and extensive module coverage. However, several critical gaps exist that prevent true production readiness:

| Gap Category | Severity | Modules Affected |
|--------------|----------|------------------|
| **No CI/CD Pipeline** | Critical | All |
| **Incomplete Implementations** | High | Video Compression, Multi-Game, PDE |
| **Integration Gaps** | Medium | PDE-Training, Curriculum Learning |
| **Testing Gaps** | Medium | SGF, Experiments, CLI Tools |
| **Documentation Gaps** | Low | Distributed, PDE, Edge Deployment |

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

## Milestone 1: CI/CD & Production Foundation
**Goal:** Establish automated quality gates and reproducible builds.
**Duration:** 1-2 days
**Subagents:** Planner, SQE, Orchestrator

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

## Milestone 2: Critical Bug Fixes
**Goal:** Fix incomplete implementations blocking production use.
**Duration:** 2-3 days
**Subagents:** Coder, SQE, Reviewer

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

## Milestone 3: Multi-Game Support Completion
**Goal:** Validate multi-game abstraction with Chess implementation.
**Duration:** 1 week
**Subagents:** Planner, Coder, SQE

### Epic 3.1: Chess Implementation
**Story 3.1.1: Implement Chess Rules Engine**
- **Task:** Complete `src/games/chess.py` with full rules
- **Components:**
  - Legal move generation
  - Check/checkmate detection
  - Castling, en passant, promotion
  - Draw conditions (50-move, repetition, insufficient material)
- **Acceptance Criteria:**
  - [ ] All chess rules correctly implemented
  - [ ] Perft test passes (move count validation)
  - [ ] Compatible with GameInterface
- **Subagent:** Coder

**Story 3.1.2: Chess State Encoding**
- **Task:** Implement AlphaZero-style state encoding (119 channels)
- **Acceptance Criteria:**
  - [ ] Piece positions encoded
  - [ ] Castling rights, en passant square, repetition count
  - [ ] 8 history planes
  - [ ] to_tensor() matches interface spec
- **Subagent:** Coder

**Story 3.1.3: Chess Symmetry Support**
- **Task:** Implement get_symmetries() for data augmentation
- **Note:** Chess has only 1 symmetry (horizontal flip, different from Go's 8)
- **Acceptance Criteria:**
  - [ ] Horizontal flip produces valid positions
  - [ ] Policy tensor correctly mirrored
  - [ ] Tests verify symmetry correctness
- **Subagent:** SQE

### Epic 3.2: Multi-Game Training Validation
**Story 3.2.1: Cross-Game Training Test**
- **Task:** Validate training loop works with Chess
- **Command:** `python -m scripts.train game=chess`
- **Acceptance Criteria:**
  - [ ] Training loop completes 100 steps
  - [ ] Loss decreases (policy + value)
  - [ ] Checkpoints save/load correctly
- **Subagent:** SQE

**Story 3.2.2: Transfer Learning Experiment**
- **Task:** Design experiment: pretrain on Go → fine-tune on Chess
- **Hypothesis:** Continuous operator learns game-agnostic patterns
- **Acceptance Criteria:**
  - [ ] Experiment script in `src/experiments/`
  - [ ] Metrics tracked in W&B
  - [ ] Results documented in `docs/`
- **Subagent:** Planner

---

## Milestone 4: PDE Game Integration
**Goal:** Connect PDE module to main training pipeline.
**Duration:** 3-4 days
**Subagents:** Planner, Coder, Reviewer

### Epic 4.1: PDE-Informed Training Mode
**Story 4.1.1: Enable Physics Loss in Standard Training**
- **Task:** Wire `CombinedAlphaGalerkinPhysicsLoss` into trainer
- **Config Addition:** `training.physics_informed: bool = False`
- **Acceptance Criteria:**
  - [ ] Physics loss computed when enabled
  - [ ] Gradient flow verified (no NaN/inf)
  - [ ] Config validated via Pydantic
- **Subagent:** Coder

**Story 4.1.2: PDE Basis Selection Self-Play**
- **Task:** Use MCTS for Galerkin basis selection
- **Integration Point:** `src/pde/games/basis_selection.py` + `src/mcts/`
- **Acceptance Criteria:**
  - [ ] MCTS can play BasisSelectionGame
  - [ ] Policy/value heads work with PDE state
  - [ ] Error reduction logged per episode
- **Subagent:** Coder

### Epic 4.2: PDE Documentation
**Story 4.2.1: Create PDE User Guide**
- **Task:** Document PDE module usage in `docs/pde_guide.md`
- **Sections:**
  - Supported PDE operators
  - Game mode configuration
  - Example usage for Poisson, Burgers
- **Acceptance Criteria:**
  - [ ] All config options documented
  - [ ] Working code examples
  - [ ] Linked from README
- **Subagent:** Coder

---

## Milestone 5: Distributed Training Validation
**Goal:** Verify distributed training at scale.
**Duration:** 3-4 days
**Subagents:** SQE, Orchestrator

### Epic 5.1: Multi-Process Integration Tests
**Story 5.1.1: Mock NCCL Multi-Process Test**
- **Task:** Create integration test with 4 processes
- **File:** `tests/integration/test_distributed_multiprocess.py`
- **Acceptance Criteria:**
  - [ ] Test spawns 4 processes
  - [ ] Gradient synchronization verified
  - [ ] No deadlocks or race conditions
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

## Milestone 6: Enhanced PoC Framework
**Goal:** Complete visualization and reporting capabilities.
**Duration:** 1 week
**Subagents:** Coder, SQE

### Epic 6.1: Visualization Module
**Story 6.1.1: Implement Interactive Plots**
- **Task:** Create `src/poc/visualization/plots.py` with Plotly
- **Plot Types:**
  - Training curves (loss, accuracy over time)
  - Hyperparameter importance (parallel coordinates)
  - Statistical comparison (box plots, violin plots)
- **Acceptance Criteria:**
  - [ ] At least 5 plot types implemented
  - [ ] Interactive HTML output
  - [ ] Static PNG fallback
- **Subagent:** Coder

**Story 6.1.2: Implement HTML Report Generator**
- **Task:** Create `src/poc/visualization/reports.py`
- **Features:**
  - Jinja2 templated reports
  - Embedded plots
  - Metric tables
  - Comparison summaries
- **Acceptance Criteria:**
  - [ ] Single HTML file output
  - [ ] Offline viewable (embedded assets)
  - [ ] Professional styling
- **Subagent:** Coder

### Epic 6.2: Activation of Curriculum Learning
**Story 6.2.1: Wire Curriculum Learning in Trainer**
- **Task:** Activate existing curriculum infrastructure
- **Location:** Model zoo + trainer integration
- **Acceptance Criteria:**
  - [ ] Config enables curriculum mode
  - [ ] Training progresses through curriculum stages
  - [ ] Stage transitions logged
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
| CI/CD | Pipeline exists | No | Yes | `.github/workflows/` present |
| Test Coverage | Overall | ~70% | >80% | pytest-cov report |
| Multi-Game | Games implemented | 1 (Go) | 2+ (Go, Chess) | Registry count |
| Distributed | Multi-node validation | No | Yes | Integration test passes |
| PDE | Training integration | No | Yes | Config option works |
| Video Compression | Hyperprior complete | No | Yes | No TODO comments |

---

## Appendix A: Critical TODOs in Codebase

| File | Line | Description | Priority |
|------|------|-------------|----------|
| `scripts/encode_video.py` | 261 | Add hyperprior z encoding | High |
| `scripts/decode_video.py` | 420 | Properly decode hyperprior z_data | High |
| `src/training/self_play.py` | 419 | True parallel generation | Medium |
| `tests/games/test_sgf.py` | - | Variation parsing (skipped) | Medium |

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

*Last Updated: 2026-02-01*
*Version: 1.0.0*
*Author: Claude Code Agent Investigation*
