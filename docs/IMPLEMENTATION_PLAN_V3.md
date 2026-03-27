# AlphaGalerkin Implementation Plan v3

## Context

AlphaGalerkin is a resolution-independent Go AI using Continuous Operator Learning (Galerkin Transformers & FNet). The project has reached v2.0 maturity with 213 source files, 207 test files, and 60% test coverage. Key milestones completed: zero-shot transfer (MSE 0.000209), Chess (97% coverage), CI/CD, Vertex AI, JAX backend, UCI engine integration, HuggingFace demos.

**This plan addresses:** the remaining 4 milestones (PDE integration, distributed validation, enhanced PoC, real-world validation), 3 open PRs (#23 Dockerization, #25 PDE RL, #26 PettingZoo), test coverage gap (60% → 85%), and cross-module integration hardening. All code follows existing patterns: Pydantic v2 config, structlog logging, decorator registries, einops tensors, no hardcoded values.

**Verified:** P0 stubs (emergency checkpoint, GTP player, parallel self-play) and P1 features (physics loss, curriculum learning) are already wired and functional in the trainer.

---

## Phase 0: Test Foundation & Coverage Baseline (60% → 72%)

**Goal:** Shared test infrastructure + close critical coverage gaps in 5 under-tested modules.

### 0.1 Shared Test Fixtures
Create reusable factory functions (no hardcoded values, config-driven):
- `tests/fixtures/__init__.py`
- `tests/fixtures/model_factory.py` — minimal `AlphaGalerkinModel` with configurable `d_model`, `n_heads`, `n_layers`
- `tests/fixtures/game_factory.py` — produce `GoGame`, `ChessGame`, or PDE adapter instances
- `tests/fixtures/pde_factory.py` — PDE configs, operators, game adapters
- `tests/deployment/conftest.py` — ONNX mocking fixtures
- `tests/tools/conftest.py` — CLI runner fixtures
- `tests/poc/conftest.py` — PoC scenario fixtures

**Pattern:** Follow `tests/curriculum/conftest.py`, `tests/analysis/conftest.py`. All factories accept Pydantic config overrides.

### 0.2 Fix Skipped Tests
- `tests/vertex/test_launcher.py` (~5 skips) — replace with `unittest.mock.MagicMock` spec'd against Vertex SDK
- `tests/games/test_sgf.py` (~1 skip) — implement variation parsing test
- `tests/e2e/test_cli_journey.py` (~2 skips) — fix `sys.path` via conftest

**Verify:** `pytest --co -q 2>&1 | grep -c "skip"` → 0

### 0.3 Deployment Module Tests (5 src, 2 test → 5 test)
- `tests/deployment/test_export_onnx.py` — mock `torch.onnx.export`, test dynamic shapes, opset selection
- `tests/deployment/test_quantize.py` — test dynamic/static quantization modes, `ValueError` for unsupported
- `tests/deployment/test_validate.py` — compare PyTorch vs ONNX outputs with tolerance

### 0.4 Tools Module Tests (4 src, 1 test → 4 test)
- `tests/tools/test_cli.py` — CLI subcommands via `click.testing.CliRunner`
- `tests/tools/test_colab.py` — Colab helper functions with mocked widgets
- `tests/tools/test_verify_invariance.py` — invariance verification script outputs

### 0.5 Experiments Module Tests (4 src, 1 test → 4 test)
- `tests/experiments/test_benchmark_fnet.py` — tiny model benchmark, verify timing format
- `tests/experiments/test_train_physics.py` — training loop entry with minimal config
- `tests/experiments/test_verify_transfer.py` — transfer verification with mocked model

### 0.6 Physics Solver Tests (5 src, 2 test → 5 test)
- `tests/physics/test_darcy.py` — mesh generation, solution properties, boundary conditions
- `tests/physics/test_heat.py` — time evolution, conservation, steady-state convergence
- `tests/physics/test_elasticity.py` — displacement, stress tensor symmetry

**Pattern:** Follow `tests/physics/test_poisson.py`. Use `hypothesis` for property-based testing.

### 0.7 CI Threshold Update
- `.github/workflows/ci.yml` — raise `--cov-fail-under=60` → `--cov-fail-under=70`

**Verification:**
```bash
pytest tests/ -m "not slow and not e2e" --cov=src --cov-report=term-missing -q  # ≥70%
pytest tests/ --co -q | grep skip  # 0 skips
ruff check src/ tests/
mypy src/ --strict --ignore-missing-imports
```

---

## Phase 1: PDE End-to-End Integration (Milestone 4)

**Goal:** Connect PDE framework → MCTS search → training pipeline for basis selection and mesh refinement self-play.

**Depends on:** Phase 0.1 (fixtures)

### 1.1 PDE Unit Tests
- `tests/pde/test_game.py` — `PDEGame` abstract base class methods
- `tests/pde/test_basis_selection.py` — state transitions, legal actions, terminal conditions
- `tests/pde/test_mesh_refinement.py` — refinement strategies, convergence
- `tests/pde/test_registry.py` — PDE operator registry

### 1.2 PDE Integration Tests
- `tests/integration/test_pde_mcts_e2e.py` — `BasisSelectionGame` → `PDEGameAdapter` → MCTS (10 sims) → verify valid policy
- `tests/integration/test_pde_mesh_refinement_e2e.py` — same for `MeshRefinementGame`
- `tests/integration/test_pde_training_pipeline.py` — full: PDE game → self-play → buffer → train → checkpoint

### 1.3 PDE Self-Play Trainer
- Create `src/pde/training.py` — `PDESelfPlayTrainer` composing existing `Trainer` infrastructure (replay buffer, checkpoint manager, loss balancer)
- Extend `config/schemas.py` — add `PDETrainingConfig` with fields: `pde_type`, `game_mode`, `max_steps`, `tolerance`, `n_collocation`
- `tests/pde/test_training.py` — verify loss decreases over tiny training run

### 1.4 PDE Documentation
- `docs/pde_guide.md` — operators, game modes, configuration, examples, training integration

**Verification:**
```bash
pytest tests/pde/ -v
pytest tests/integration/test_pde_*.py -v
python -c "from src.pde.mcts_adapter import PDEGameAdapter; print('OK')"
```

**Key files:** `src/pde/mcts_adapter.py`, `src/training/self_play.py`, `config/schemas.py`

---

## Phase 2: Coverage Push to 85% (parallel with Phase 1)

**Goal:** Systematically close remaining coverage gap.

**Depends on:** Phase 0

### 2.1 PoC Module (17 src, 5 test → 13 test)
- `tests/poc/test_cli.py` — CLI list, info, run via CliRunner
- `tests/poc/test_logging.py` — structured logging, context binding
- `tests/poc/test_scenarios_transfer.py` — transfer scenario with mocked model
- `tests/poc/test_scenarios_complexity.py` — complexity benchmark
- `tests/poc/test_scenarios_stability.py` — stability monitoring
- `tests/poc/test_tuning_config.py` — tuning config validation
- `tests/poc/test_tuning_sampler.py` — TPE, grid, random samplers
- `tests/poc/test_statistics_significance.py` — t-test, Mann-Whitney, bootstrap, effect sizes

### 2.2 Distributed Module (7 src, 4 test → 7 test)
- `tests/distributed/test_launcher.py` — torchrun/SLURM command generation with mocked subprocess
- `tests/distributed/test_model_zoo.py` — checkpoint save/load/list/prune
- `tests/distributed/test_worker.py` — distributed self-play worker protocol

### 2.3 Modeling Module (10 src, 5 test → 8 test)
- `tests/modeling/test_operator.py` — backend selection, `NotImplementedError` for unsupported
- `tests/modeling/test_model.py` — forward pass with both backends
- Additional tests for uncovered attention/layer files

### 2.4 Games Module (14 src, 8 test → 12 test)
- `tests/games/test_state.py` — `GameState` dataclass operations
- `tests/games/test_registry.py` — registration, lookup, listing
- `tests/games/test_interface.py` — abstract compliance via concrete implementations
- `tests/games/sgf/test_config.py` — SGF config validation

### 2.5 Data Module (4 src, 3 test → 4 test)
- `tests/data/test_physics_dataset.py` — loading, batching, normalization

### 2.6 CI Threshold Final
- `.github/workflows/ci.yml` — raise `--cov-fail-under=70` → `--cov-fail-under=85`

**Verification:** `pytest tests/ --cov=src --cov-report=term-missing` ≥ 85%, no module below 70%

---

## Phase 3: Cross-Module Integration Hardening (Milestone 5 partial)

**Goal:** Validate composed module workflows match production scenarios.

**Depends on:** Phases 1, 2

### 3.1 Integration Tests
- `tests/integration/test_full_training_pipeline.py` — config → model → self-play (5 games) → buffer → 3 train steps → checkpoint → resume → verify loss continuity
- `tests/integration/test_multigame_switching.py` — switch Go ↔ Chess configs; verify architecture adaptation
- `tests/integration/test_curriculum_progression.py` — board sizes 9→13→19; stage transitions; checkpoint preservation
- `tests/integration/test_physics_informed_training.py` — `physics_informed=True` activates loss; gradient flow; metrics output
- `tests/integration/test_onnx_roundtrip.py` — export → ONNX Runtime load → compare within 1e-5 for policy + value heads; dynamic batch (1, 8, 32)

### 3.2 Distributed Training Validation (Milestone 5)
- `tests/distributed/test_multiprocess.py` — spawn 2+ processes, gradient sync, parameter convergence
- `docs/distributed_guide.md` — multi-GPU/multi-node setup, SLURM examples

**Verification:**
```bash
pytest tests/integration/ -v --timeout=120
# Run 3x for flakiness check
```

---

## Phase 4: Open PR Integration & Feature Completion

**Goal:** Coordinate with open PRs, clean up stubs.

**Parallel with:** Phases 1-3

### 4.1 PR #25 (Core RL Framework for PDE)
- Verify 187 tests integrate without conflicts
- Ensure Pydantic configs extend `config/schemas.py` patterns
- Add integration test connecting PR #25 RL loop with Phase 1 PDE training

### 4.2 PR #23 (Dockerization)
- Verify Docker Compose uses dynamic env vars (no hardcoded ports/paths)
- SafeTensors migration is backwards-compatible with existing `.pt` checkpoints
- CI scanning integrates with `.github/workflows/ci.yml`

### 4.3 PR #26 (PettingZoo Demo)
- PettingZoo wrappers follow `GameInterface` protocol
- MARL environments use game registry pattern
- Demo scripts have tests

### 4.4 Stub Cleanup
- Remove `ShogiGame` reference from `src/games/__init__.py` docstring (not implemented — add to `src/games/ROADMAP.md`)
- Replace placeholder in `src/analysis/reviewer.py` with real game-state analysis logic
- Extend `tests/analysis/test_reviewer.py` for new logic

---

## Phase 5: DevOps Maturity (parallel with Phases 2-4)

### 5.1 Release Automation
- `.github/workflows/release.yml` — on `v*` tags: build wheel + sdist, publish, create GitHub Release

### 5.2 Dependency Management
- `.github/dependabot.yml` — weekly pip updates, auto-merge patches after CI

### 5.3 Container Security
- Add Trivy scan stage to CI for `docker/Dockerfile.vertex`

### 5.4 Repository Governance
- `CODEOWNERS`, `SECURITY.md`
- `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/PULL_REQUEST_TEMPLATE.md`

---

## Phase 6: Real-World Validation (Milestone 7)

**Goal:** External benchmarks proving production readiness.

**Depends on:** Phase 3

### 6.1 Go Engine Tournament
- `scripts/run_tournament.py` — GTP protocol vs GnuGo/KataGo
- `docs/benchmarks/go_tournament.md` — Elo estimate with confidence intervals

### 6.2 Video Codec Benchmarking
- `scripts/run_codec_benchmark.py` — BD-rate curves on Xiph.org sequences
- `docs/benchmarks/video_compression.md` — PSNR, SSIM, MS-SSIM vs H.265/VP9

### 6.3 Zero-Shot Transfer Ablation
- `scripts/run_transfer_ablation.py` — board sizes 5, 7, 9, 13, 19, 25
- `docs/benchmarks/transfer_ablation.md` — MSE curves, scaling analysis

---

## Dependency Graph

```
Phase 0 (Foundation, 60→72%)
    ├──→ Phase 1 (PDE E2E, Milestone 4)  ──────┐
    ├──→ Phase 2 (Coverage Push, 72→85%)  ──────┤──→ Phase 3 (Integration, Milestone 5)
    ├──→ Phase 4 (PR Integration)               │         │
    └──→ Phase 5 (DevOps)                       │         └──→ Phase 6 (Validation, Milestone 7)
                                                │
                                                └─────────────────────────────────────────────────
```

## Coverage Trajectory

| Phase     | Est. Coverage | CI Threshold |
|-----------|--------------|--------------|
| Current   | 60%          | 60%          |
| Phase 0   | 70-72%       | 70%          |
| Phase 1   | 73-75%       | 70%          |
| Phase 2   | 85%+         | 85%          |
| Phase 3   | 87-88%       | 85%          |

## Key Architectural Rules

1. **No hardcoded values** — all configs via Pydantic v2 extending `src/templates/config.py:BaseModuleConfig`
2. **Test factories over fixture proliferation** — `tests/fixtures/` with parameterized factories accepting config overrides
3. **Tiny models in integration tests** — `d_model=16, n_heads=2, n_layers=1` to keep CI < 2 min/test
4. **Backwards compatibility enforced by mypy** — no existing signatures change; new optional params get defaults
5. **Composition over inheritance** — `PDESelfPlayTrainer` reuses existing `Trainer` components (replay buffer, checkpoint manager, loss balancer)
6. **Slow tests marked** — `@pytest.mark.slow` or `@pytest.mark.gpu_required` for anything > 10s

## Critical Files Reference

| File | Role | Phases |
|------|------|--------|
| `src/training/trainer.py` | Central orchestrator (physics loss + curriculum already wired) | 1, 3 |
| `src/pde/mcts_adapter.py` | PDE ↔ MCTS bridge | 1 |
| `config/schemas.py` | All Pydantic config schemas | 0, 1, 2 |
| `.github/workflows/ci.yml` | CI pipeline, coverage threshold (line ~217) | 0, 2 |
| `src/training/self_play.py` | `SelfPlayWorker` + `ParallelSelfPlayWorker` | 1, 3 |
| `src/templates/config.py` | `BaseModuleConfig` — all configs extend this | All |
| `src/games/interface.py` | Abstract `GameInterface` protocol | 1, 4 |
| `tests/conftest.py` | Root fixtures (seed, device, backends) | 0 |
