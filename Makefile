# AlphaGalerkin — developer workflow automation
#
# Cross-platform (GNU Make on Linux/macOS, make via choco/scoop on Windows).
# All commands assume the virtual environment is activated or that the
# .venv/Scripts (Windows) or .venv/bin (Unix) prefix is used.
#
# Usage:
#   make lint          # ruff check + format check
#   make format        # ruff auto-format
#   make mypy          # strict type check (informational)
#   make test-fast     # fast unit tests (excludes slow/e2e/gpu)
#   make test-cert     # certificate module tests + 85% gate
#   make test-stoch    # stochastic module tests + 85% gate
#   make test-all      # full test suite (fast + slow)
#   make coverage      # global coverage with 85% gate
#   make gitleaks      # secret scan
#   make pre-commit    # all pre-commit hooks
#   make docs-serve    # MkDocs local dev server
#   make clean         # remove caches
#   make check         # lint + mypy + test-fast (pre-PR quick check)

.PHONY: lint format mypy test-fast test-cert test-stoch test-all coverage \
        gitleaks pre-commit docs-serve clean check gpu-smoke \
        demo pre-pr test-agents test-benchmarks test-core test-e2e \
        test-regression test-sanity test-security

# ---------------------------------------------------------------------------
# Tool resolution — prefer venv binaries, fall back to system
# ---------------------------------------------------------------------------
PYTHON ?= python
PYTEST ?= pytest
RUFF   ?= ruff
MYPY   ?= mypy
COV    ?= coverage

# Coverage tracer — pytrace avoids torch C-extension crashes on Windows/nightly
export COVERAGE_CORE ?= pytrace

# Disable wandb telemetry in tests
export WANDB_MODE ?= disabled

# Non-interactive matplotlib backend
export MPLBACKEND ?= Agg

# Deterministic hash seed
export PYTHONHASHSEED ?= 0

# ---------------------------------------------------------------------------
# Coverage thresholds (match ci.yml and regression-surface.yml)
# ---------------------------------------------------------------------------
GLOBAL_COV_THRESHOLD   ?= 85
CERT_COV_THRESHOLD     ?= 85
STOCH_COV_THRESHOLD    ?= 85

# ---------------------------------------------------------------------------
# Test-selection parity with CI
# ---------------------------------------------------------------------------
# `ci.yml` applies these SAME exclusions in both its `test-fast` and `coverage`
# jobs. They live here as one variable so `make test-fast` / `make coverage`
# measure what CI measures -- before this, the Makefile applied 3 of the 6
# --ignore paths and none of the 9 --deselect ids, a 115-test divergence in the
# targets `make pre-pr` chains to decide a PR is ready.
#
# Known duplication: this is a THIRD copy (ci.yml holds two). Collapsing all
# three onto one shared args file is tracked as backlog B7 -- deliberately not
# done here, because rewriting CI's test invocation risks a green PR for a
# cosmetic win. Keep this block in step with ci.yml by hand until then.
CI_TEST_EXCLUDES := \
	--ignore=tests/e2e/ \
	--ignore=tests/integration/ \
	--ignore=tests/demos/ \
	--ignore=tests/training/test_extended_config.py \
	--ignore=tests/notebooks/ \
	--ignore=tests/distributed/test_multiprocess.py \
	--deselect=tests/data/test_dataset.py::TestReplayDataset::test_iteration_with_dataloader \
	--deselect=tests/data/test_dataset.py::TestExperienceListDataset::test_with_dataloader \
	--deselect=tests/data/test_dataset.py::TestDatasetIntegration::test_batch_sampler_with_list_dataset \
	--deselect=tests/experiments/test_physics_loss.py::TestPhysicsLossComputeLaplacian::test_laplacian_of_linear \
	--deselect=tests/games/test_chess.py::TestChessEdgeCases::test_invalid_move_notation \
	--deselect=tests/games/test_chess.py::TestChessEdgeCases::test_illegal_move_notation \
	--deselect=tests/mcts/test_node.py::TestPruneExcept::test_prune_except_returns_child \
	--deselect=tests/mcts/test_search.py::TestMCTSTreeManagement::test_advance_reuses_subtree \
	--deselect=tests/training/test_self_play.py::TestParallelSelfPlayWorker::test_generate_games_sequential_fallback_on_error

# ---------------------------------------------------------------------------
# Lint & Format
# ---------------------------------------------------------------------------
lint:
	$(RUFF) check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
	$(RUFF) format --check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py

format:
	$(RUFF) format src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py

# ---------------------------------------------------------------------------
# Type Checking (informational — not a blocking gate)
# ---------------------------------------------------------------------------
mypy:
	$(MYPY) src/ --strict --ignore-missing-imports || true

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test-fast:
	$(PYTEST) tests/ \
		-m "not slow and not e2e and not gpu_required" \
		$(CI_TEST_EXCLUDES) \
		-q --no-header

test-sanity:
	$(PYTEST) tests/sanity/ -v

test-security:
	$(PYTEST) tests/security/ -v

test-benchmarks:
	$(PYTEST) tests/benchmarks/ -v

test-regression:
	$(PYTEST) tests/regression/ -v

test-core:
	$(PYTEST) tests/core/ -v

test-agents:
	$(PYTEST) tests/agents/ -v

test-e2e:
	$(PYTEST) tests/e2e/test_user_journey_*.py -v

test-cert:
	$(COV) run --branch \
		--include="*/src/pde/certificate/*" \
		-m pytest tests/pde/certificate/ -q -p no:cov
	$(COV) report \
		--include="*/src/pde/certificate/*" \
		--fail-under=$(CERT_COV_THRESHOLD)

test-stoch:
	$(COV) run --branch \
		--include="*/src/pde/stochastic/*,*/src/research/stochastic_galerkin_compare.py,*/src/poc/scenarios/stochastic_galerkin_compare.py,*/src/poc/scenarios/stochastic_galerkin_compare_config.py" \
		-m pytest tests/pde/stochastic tests/research/test_stochastic_galerkin_compare.py \
		tests/poc/test_stochastic_galerkin_compare_config.py tests/poc/test_stochastic_galerkin_compare_scenario.py \
		tests/scripts/test_run_stochastic_galerkin_compare.py tests/regression/test_related_work_guard.py -q -p no:cov
	$(COV) report \
		--include="*/src/pde/stochastic/*,*/src/research/stochastic_galerkin_compare.py,*/src/poc/scenarios/stochastic_galerkin_compare.py,*/src/poc/scenarios/stochastic_galerkin_compare_config.py" \
		--fail-under=$(STOCH_COV_THRESHOLD)

test-all:
	$(PYTEST) tests/ -q --no-header

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
coverage:
	$(PYTEST) tests/ \
		-m "not slow and not e2e and not gpu_required" \
		$(CI_TEST_EXCLUDES) \
		--cov=src \
		--cov-fail-under=$(GLOBAL_COV_THRESHOLD) \
		-q --no-header

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
demo:
	$(PYTHON) -m src.poc.cli run --config config/scenarios/stochastic_galerkin_compare_ci.yaml --demo

# ---------------------------------------------------------------------------
# GPU Validation
# ---------------------------------------------------------------------------
gpu-smoke:
	$(PYTEST) tests/pde/stochastic/test_gpu_smoke.py -v

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
gitleaks:
	gitleaks detect --config .gitleaks.toml --verbose

# ---------------------------------------------------------------------------
# Pre-commit (all hooks)
# ---------------------------------------------------------------------------
pre-commit:
	pre-commit run --all-files

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
docs-serve:
	mkdocs serve

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage.* coverage.xml 2>/dev/null || true

# ---------------------------------------------------------------------------
# Pre-PR Comprehensive Gate (lint + mypy + sanity + security + regression +
# benchmarks + core + agents + e2e + fast + coverage[85% global gate])
# ---------------------------------------------------------------------------
pre-pr: lint mypy test-sanity test-security test-regression test-benchmarks test-core test-agents test-e2e test-fast coverage
check: pre-pr
