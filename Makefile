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
        gitleaks pre-commit docs-serve clean check gpu-smoke

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
# Lint & Format
# ---------------------------------------------------------------------------
lint:
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/

format:
	$(RUFF) format src/ tests/

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
		--ignore=tests/e2e/ \
		--ignore=tests/integration/ \
		--ignore=tests/demos/ \
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
		--include="*/src/pde/stochastic/*" \
		-m pytest tests/pde/stochastic/ -q -p no:cov
	$(COV) report \
		--include="*/src/pde/stochastic/*" \
		--fail-under=$(STOCH_COV_THRESHOLD)

test-all:
	$(PYTEST) tests/ -q --no-header

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
coverage:
	$(PYTEST) tests/ \
		-m "not slow and not e2e and not gpu_required" \
		--ignore=tests/e2e/ \
		--ignore=tests/integration/ \
		--ignore=tests/demos/ \
		--cov=src \
		--cov-fail-under=$(GLOBAL_COV_THRESHOLD) \
		-q --no-header

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
demo:
	$(PYTHON) -m src.poc.cli run --scenario transfer_darcy_to_poisson --demo

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
# Pre-PR Comprehensive Gate (lint + mypy + sanity + security + regression + benchmarks + core + fast)
# ---------------------------------------------------------------------------
pre-pr: lint mypy test-sanity test-security test-regression test-benchmarks test-core test-agents test-e2e test-fast
check: pre-pr

