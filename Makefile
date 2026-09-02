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
#   make test-substrate # refinement-substrate surface + 95% gate (needs [fem])
#   make test-all      # full test suite (fast + slow)
#   make coverage      # global coverage with 85% gate
#   make gitleaks      # secret scan
#   make pre-commit    # all pre-commit hooks
#   make docs-serve    # MkDocs local dev server
#   make clean         # remove caches
#   make check         # lint + mypy + test-fast (pre-PR quick check)
#   make docker-build  # build docker/Dockerfile (override DOCKER_IMAGE/CONTEXT)
#   make docker-test   # build, then run the image's own default CMD

.PHONY: lint format mypy test-fast test-cert test-stoch test-all coverage \
        gitleaks pre-commit docs-serve clean check gpu-smoke \
        demo pre-pr test-agents test-benchmarks test-core test-e2e \
        test-regression test-sanity test-security test-demos test-claude \
        test-substrate docker-build docker-test

# ---------------------------------------------------------------------------
# Tool resolution
#
# The Python-importing tools go through `$(PYTHON) -m` rather than their bare
# console scripts, because a bare `pytest`/`mypy`/`coverage` on PATH is bound to
# whichever interpreter installed it -- which need not be the one holding this
# project's dependencies. Measured in a working container: `pytest` resolved to
# a uv tool interpreter and every test target died at
# `ModuleNotFoundError: No module named 'hypothesis'` while `python -m pytest`
# ran the full suite. That silently broke `make pre-pr`, the documented pre-PR
# gate, for every target except `lint`. `$(PYTHON) -m` also still prefers an
# activated venv (its `python` is first on PATH), so this is strictly better at
# the thing the previous comment claimed. `ruff` stays a bare binary: it is a
# standalone executable that imports nothing from the environment.
#
# Every variable remains overridable: `PYTEST="uv run pytest" make test-fast`.
# ---------------------------------------------------------------------------
PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= ruff
MYPY   ?= $(PYTHON) -m mypy
COV    ?= $(PYTHON) -m coverage

# NOTE: no COVERAGE_CORE pin. The `pytrace` pin that used to live here was
# retired 2026-09-02 after the crash it guarded against was shown not to
# reproduce; coverage's default C tracer is ~3x faster and measures identically.
# See CHANGELOG.md's tracer-retirement entry for the CI evidence.

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
# Above the usual floor(measured)-2-capped-at-85 convention on purpose: the
# substrates package is 272 statements at 99%, so an 85 gate would carry 14
# points of slack. Must match ci.yml's test-extras step.
SUBSTRATE_COV_THRESHOLD ?= 95

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

# Mirrors ci.yml's "Run demo and notebook suites" step. These 226 tests are
# --ignore'd from test-fast (see CI_TEST_EXCLUDES) because they are slower than
# a unit test, so without this target `make pre-pr` is NARROWER than CI -- which
# is the drift that let them go unexecuted in CI for months in the first place.
test-demos:
	$(PYTEST) tests/demos/ tests/notebooks/ \
		-m "not gpu_required" \
		-q --no-header

# Mirrors ci.yml's "Validate .claude harness" step: deterministic, hermetic
# validation of the 9 skills / 5 subagents / 4 commands / hook / settings.json.
test-claude:
	$(PYTEST) tests/claude/ -q --no-header

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

# Mirrors ci.yml's test-extras "Coverage gate (src/research/substrates)" step.
# REQUIRES the optional [fem] extra: without scikit-fem the fem_required tests
# skip, the measured percentage collapses, and the gate fails for the wrong
# reason -- so this target is deliberately NOT chained into `pre-pr`, which must
# stay runnable on a base install. ALPHAGALERKIN_REQUIRE_EXTRAS=1 turns a
# missing scikit-fem into a loud collection error rather than that silent
# collapse. The heredoc'd rcfile drops skfem_tri.py from pyproject.toml's global
# coverage `omit` for this run only; a bare --cov of it otherwise reports 0.00%
# with "No data was collected".
test-substrate:
	@printf '[run]\nbranch = true\n\n[report]\nshow_missing = true\n' > .coveragerc.substrates
	ALPHAGALERKIN_REQUIRE_EXTRAS=1 $(PYTEST) \
		tests/refinement/test_substrate.py \
		tests/research/test_marking.py \
		tests/research/test_substrates_config.py \
		tests/research/test_substrates_sweep.py \
		tests/research/test_tensor_grid_substrate.py \
		tests/research/test_skfem_substrate.py \
		tests/research/test_amr_arena_interpretability.py \
		--cov=src/research/substrates \
		--cov-config=.coveragerc.substrates \
		--cov-branch --cov-fail-under=$(SUBSTRATE_COV_THRESHOLD) -q --no-header

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
# Chained into `pre-pr`, so it must not hard-fail on a machine without the
# binary -- but it must not pass QUIETLY either. A skipped scan reported as
# success is the "check described but not executed" failure mode this repo has
# had to correct repeatedly; the notice below names CI as the enforcing copy.
gitleaks:
	@if command -v gitleaks >/dev/null 2>&1; then \
		gitleaks detect --config .gitleaks.toml --verbose; \
	else \
		echo "SKIPPED: gitleaks is not installed -- this scan did NOT run."; \
		echo "         Install it (https://github.com/gitleaks/gitleaks) or rely"; \
		echo "         on the 'Secret scan (gitleaks)' step in ci.yml, which is"; \
		echo "         the enforcing copy."; \
	fi

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
# Docker
#
# `docker/Dockerfile` shipped on 2026-08-16 and, until 2026-09-02, was built by
# nothing -- no CI job, no target here, no test -- while CLAUDE.md's Next Steps
# simultaneously recorded that no Dockerfile existed. These targets make it
# runnable by hand; `tests/docs/test_dockerfile_context.py` is the part that
# runs in CI, and it needs no daemon: it asserts every COPY source survives
# `.dockerignore` and that the CMD's trees actually reach the image.
#
# DOCKER_IMAGE / DOCKER_CONTEXT are variables, not literals, so a caller can
# retag or build from an exported context without editing this file.
# ---------------------------------------------------------------------------
DOCKER ?= docker
DOCKER_IMAGE ?= alphagalerkin:dev
DOCKER_CONTEXT ?= .
DOCKERFILE ?= docker/Dockerfile

docker-build:
	$(DOCKER) build -f $(DOCKERFILE) -t $(DOCKER_IMAGE) $(DOCKER_CONTEXT)

# Runs the image's own default command (the sanity/security/benchmarks/
# regression selection baked into the Dockerfile's CMD), so this target cannot
# drift from what the image actually does on `docker run`.
docker-test: docker-build
	$(DOCKER) run --rm $(DOCKER_IMAGE)

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
pre-pr: lint mypy gitleaks test-claude test-sanity test-security test-regression test-benchmarks test-core test-agents test-e2e test-demos test-fast coverage
check: pre-pr
