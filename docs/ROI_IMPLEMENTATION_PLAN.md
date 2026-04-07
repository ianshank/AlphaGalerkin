# AlphaGalerkin ROI Implementation Plan

## Context

AlphaGalerkin (v0.3.0 → v0.4.0) is a mature resolution-independent AI system with strong mathematical foundations (Galerkin Transformers, FNet, MCTS). Milestones 1-3 and 8 are complete. The project is positioning for SBIR submissions (Navy N252-088, DOE ASCR, NSF, AFWERX) at TRL 3-4. This plan prioritizes by **ROI = (SBIR readiness + technical risk reduction + downstream unblocking) / effort**.

**Status Update (2026-04-07):** Tier 1 items 1.1-1.4 are now COMPLETE. Tier 2 items 2.1, 2.3, and partial 2.4 are COMPLETE. Visualization module (from Tier 3.3) is COMPLETE. See commit history on `claude/create-implementation-plan-hFHXP` for details.

---

## Tier 1: Highest ROI (Do First)

### 1.1 Wire Physics Loss into Training Loop (M4.1)
**ROI**: Unlocks PDE self-play, SBIR demos, and the core novel capability  
**Effort**: S (1-2 days) — infrastructure exists, just needs wiring  
**Value**: Critical for SBIR — this IS the differentiator (MCTS + Galerkin for PDE)

**Files to modify:**
- `src/training/trainer.py` — add `physics_informed` branch in training step
- `src/training/losses/__init__.py` — ensure `CombinedAlphaGalerkinPhysicsLoss` is registered
- `config/train.yaml` — add `training.physics_informed: bool = false` field
- `src/training/config.py` — add Pydantic field for physics config

**Reuse:**
- `src/training/physics_loss.py` — `CombinedAlphaGalerkinPhysicsLoss` already implemented
- `src/training/loss_balancing.py` — ReLoBRaLo/GradNorm already implemented
- `src/pde/operators.py` — Poisson, Burgers, NS operators ready

**Tests:**
- `tests/training/test_trainer_physics.py` — new: physics loss gradient flow, config toggle, NaN guard
- Target: 85% coverage on new code paths

**Acceptance:** `python -m scripts.train training.physics_informed=true` runs without error, physics loss logged

---

### 1.2 SBIR Benchmark Suite Completion (M9.1 partial)
**ROI**: Directly enables SBIR proposal submission with quantitative results  
**Effort**: M (2-3 days) — extend existing baselines  
**Value**: Revenue-critical — proposals need comparative benchmarks

**Files to modify:**
- `src/research/baselines.py` — extend `DorflerAMRSolver` to 2D, add `NavierStokesBaselineSolver`
- `src/research/pde_benchmarks.py` — ensure benchmark runner handles new solvers
- `config/benchmarks/sbir_suite.yaml` — verify all 3 benchmarks have baselines
- `src/demos/sbir_demo.py` — create end-to-end demo script

**Reuse:**
- `src/physics/navier_stokes.py` — `NavierStokesOperator` with Taylor-Green vortex (exact analytical solution)
- `src/pde/geometry.py` — `LShapedDomain` for AMR benchmarking
- `src/research/reporter.py` — existing report generation

**Tests:**
- `tests/research/test_baselines_2d.py` — 2D AMR solver convergence
- `tests/research/test_ns_baseline.py` — NS solver against analytical solution
- `tests/demos/test_sbir_demo.py` — demo runs in <5 min, produces HTML report

**Acceptance:** `python -m src.demos.sbir_demo` produces comparison report for all 3 SBIR benchmarks

---

### 1.3 Loss Balancing Strategy Audit & Fix (Training Safety)
**ROI**: Prevents silent training failures, reduces technical risk  
**Effort**: S (0.5-1 day) — audit + fix if needed  
**Value**: High — training correctness is foundational

**Files to inspect/modify:**
- `src/training/loss_balancing.py` — verify `update()` in all 5 strategies (Static, ReLoBRaLo, GradNorm, Uncertainty, SoftAdapt)
- `tests/training/test_loss_balancing.py` — add missing strategy coverage

**Reuse:** Existing `create_loss_balancer` factory, existing Pydantic `LossBalancingConfig`

**Tests:** Property-based tests (Hypothesis) for each strategy's `update()` + `compute_weighted_loss()`

**Acceptance:** All 5 strategies pass roundtrip test: init → update(losses) → compute_weighted_loss() → valid gradients

---

### 1.4 PDE-MCTS Basis Selection Self-Play (M4.2)
**ROI**: The core SBIR novelty — MCTS-guided Galerkin basis selection  
**Effort**: M (2-3 days) — adapter exists, needs training loop wiring  
**Value**: Critical for patent claims and SBIR proposals

**Files to modify:**
- `src/pde/mcts_adapter.py` — verify `PDEGameAdapter` fully bridges to MCTS `GameInterface`
- `src/training/trainer.py` — add PDE game mode (config: `training.game=pde_basis`)
- `src/games/registry.py` — register PDE games alongside Go/Chess
- `config/train_pde.yaml` — new config for PDE basis selection training

**Reuse:**
- `src/pde/games/basis_selection.py` — `BasisSelectionGame` already implemented
- `src/pde/mcts_adapter.py` — `PDEGameAdapter` already bridges to MCTS
- `src/mcts/gumbel.py` — Gumbel AlphaZero MCTS ready
- `src/training/self_play.py` — `SelfPlayWorker` is game-agnostic

**Tests:**
- `tests/pde/test_mcts_training.py` — 2-step PDE self-play + training, error reduction logged
- `tests/integration/test_pde_e2e.py` — config-driven PDE training E2E

**Acceptance:** `python -m scripts.train --config-name=train_pde` runs, error reduction per episode logged

---

## Tier 2: High ROI (Do Next)

### 2.1 Chess Test Coverage to 85% (Active Sprint 2)
**ROI**: Unblocks Sprint 3-4, satisfies CI coverage gates  
**Effort**: M (2-3 days)  
**Value**: Quality gate + enables full-scale training confidence

**Files to create/modify:**
- `tests/games/test_chess.py` — edge cases (promotion, castling, en passant, stalemate, 50-move, threefold)
- `tests/games/test_chess_roundtrip.py` — exhaustive encode/decode fuzz for all 4672 actions
- `tests/data/test_collate.py` — chess experience batching, mask shapes
- `tests/training/test_chess_self_play.py` — multi-game generation, experience shapes

**Reuse:** Existing test patterns from Go tests, Hypothesis strategies

**Acceptance:** `pytest tests/games/ tests/training/ --cov=src/games --cov-fail-under=85`

---

### 2.2 Distributed Training Multi-Process Tests (M5.1)
**ROI**: Validates multi-node capability, required for cloud training  
**Effort**: M (2-3 days)  
**Value**: Enables Vertex AI production training, SBIR scalability claims

**Files to modify:**
- `tests/integration/test_distributed_multiprocess.py` — 4-process NCCL mock test
- `tests/vertex/test_launcher.py` — fix 5 skipped tests with improved SDK mocking
- `src/distributed/trainer.py` — any fixes discovered during testing

**Reuse:** Existing `src/distributed/` infrastructure, `torch.multiprocessing.spawn`

**Acceptance:** All distributed tests pass, 0 skipped Vertex launcher tests

---

### 2.3 Curriculum Learning Activation (M6.2)
**ROI**: Enables progressive training (9x9 → 13x13 → 19x19), improves convergence  
**Effort**: S (1-2 days) — infrastructure exists in `src/curriculum/`  
**Value**: Training quality improvement, SBIR methodology differentiator

**Files to modify:**
- `src/training/trainer.py` — wire `curriculum_schedule` from config
- `src/curriculum/` — verify stage transition logic
- `config/train.yaml` — add `curriculum_schedule` field

**Reuse:** `src/curriculum/` module, `TrainingConfig.curriculum_schedule` field (already defined)

**Tests:** `tests/training/test_curriculum_training.py` — stage transitions logged, resolution changes

---

### 2.4 BaseTrainer Refactor (M9.2)
**ROI**: DRY improvement enabling faster iteration on all training modes  
**Effort**: M (2-3 days)  
**Value**: Reduces maintenance burden, enables cleaner PDE/physics training

**Files to modify:**
- `src/training/base_trainer.py` — new: shared AMP, grad clip, LR scheduling, checkpoint
- `src/training/trainer.py` — inherit from `BaseTrainer`
- `src/training/operator_trainer.py` — inherit from `BaseTrainer` (if exists)

**Reuse:** Extract common patterns from existing `Trainer`

**Tests:** All existing trainer tests must pass (backwards compatible)

---

## Tier 3: Strategic (Do When Ready)

### 3.1 Full-Scale Chess Training (Sprint 4)
**Effort**: XL (1-2 weeks)  
**Depends on**: 2.1 (coverage), 2.3 (curriculum)  
**Value**: Elo validation, training pipeline stress test

### 3.2 ONNX Production Export (M9.2)
**Effort**: M (2-3 days)  
**Depends on**: 2.4 (BaseTrainer)  
**Value**: Edge deployment, SBIR TRL advancement

### 3.3 Visualization & HTML Reports (M6.1)
**Effort**: M (2-3 days)  
**Depends on**: 1.2 (SBIR benchmarks generate data to visualize)  
**Value**: SBIR proposal appendices, PoC reporting

### 3.4 Real-World Validation (M7)
**Effort**: XL (2 weeks)  
**Depends on**: 3.1 (trained models), 1.2 (benchmarks)  
**Value**: External credibility, Elo estimates, codec benchmarks

### 3.5 Parallel Self-Play with Multiprocessing
**Effort**: M (2-3 days)  
**Depends on**: 2.2 (distributed tests validate multiprocessing patterns)  
**Value**: Training throughput improvement for full-scale runs

---

## Execution Order (Critical Path)

```
Week 1:  1.3 (loss audit) → 1.1 (physics loss wiring) → 1.4 (PDE-MCTS self-play)
         1.2 (SBIR benchmarks) [parallel]
Week 2:  2.1 (chess coverage) + 2.2 (distributed tests) [parallel]
         2.3 (curriculum activation)
Week 3:  2.4 (BaseTrainer refactor)
         3.3 (visualization) [parallel]
Week 4+: 3.1 (full-scale training) → 3.4 (real-world validation)
         3.2 (ONNX export) [parallel]
```

---

## Cross-Cutting Requirements

- **No hardcoded values**: All new parameters via Pydantic config with validation
- **Backwards compatible**: New config fields have sensible defaults, existing configs unchanged
- **Modular/reusable**: Follow existing registry pattern (`create_registry`), factory functions
- **Dynamic code**: Use existing `GameRegistry`, `PDEOperatorRegistry`, loss registry patterns
- **85% test coverage**: Each new module tested with unit + integration tests
- **Property-based testing**: Use Hypothesis for mathematical invariants (loss, PDE operators)

---

## Verification Plan

```bash
# After each tier, validate:
ruff check src/ tests/
mypy src/ --strict
pytest tests/ -v --cov=src --cov-fail-under=85

# Tier 1 specific:
python -m scripts.train training.physics_informed=true --config-name=train_fast
python -m src.demos.sbir_demo
pytest tests/training/test_loss_balancing.py -v
python -m scripts.train --config-name=train_pde

# Tier 2 specific:
pytest tests/games/ --cov=src/games --cov-fail-under=85
pytest tests/distributed/ tests/vertex/ -v
pytest tests/training/test_curriculum_training.py -v
```
