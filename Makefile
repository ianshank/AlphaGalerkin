# AlphaGalerkin Makefile
# Usage: make <target>

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
SRC := src/alphagalerkin
TESTS := tests/alphagalerkin
CONFIGS := configs/alphagalerkin

PYTHON := python
PIP := pip

# ---------------------------------------------------------------
# Help
# ---------------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
.PHONY: install
install: ## Install package and dependencies
	$(PIP) install --break-system-packages -e ".[dev]"

.PHONY: install-all
install-all: ## Install package with all optional dependencies
	$(PIP) install --break-system-packages -e ".[dev,vertex]"

# ---------------------------------------------------------------
# Quality
# ---------------------------------------------------------------
.PHONY: lint
lint: ## Run ruff linter
	ruff check $(SRC) $(TESTS)

.PHONY: lint-fix
lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix $(SRC) $(TESTS)

.PHONY: format
format: ## Run ruff formatter
	ruff format $(SRC) $(TESTS)

.PHONY: format-check
format-check: ## Check formatting without modifying files
	ruff format --check $(SRC) $(TESTS)

.PHONY: typecheck
typecheck: ## Run mypy type checking
	mypy $(SRC) --strict

# ---------------------------------------------------------------
# Testing
# ---------------------------------------------------------------
.PHONY: test
test: ## Run all tests
	pytest $(TESTS) -v

.PHONY: test-unit
test-unit: ## Run unit tests only
	pytest $(TESTS)/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	pytest $(TESTS)/integration -v

.PHONY: test-property
test-property: ## Run property-based tests only
	pytest $(TESTS)/property -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing --cov-report=html

.PHONY: test-fast
test-fast: ## Run tests excluding slow markers
	pytest $(TESTS) -v -m "not slow"

# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------
.PHONY: validate-config
validate-config: ## Validate default configuration
	$(PYTHON) -m alphagalerkin validate-config --config $(CONFIGS)/default.yaml

.PHONY: dry-run
dry-run: ## Dry run training with quick-test config
	$(PYTHON) -m alphagalerkin train --config $(CONFIGS)/training/quick_test.yaml --dry-run

# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
.PHONY: train
train: ## Train with default configuration
	$(PYTHON) -m alphagalerkin train --config $(CONFIGS)/default.yaml

.PHONY: train-quick
train-quick: ## Train with quick-test configuration
	$(PYTHON) -m alphagalerkin train --config $(CONFIGS)/training/quick_test.yaml

# ---------------------------------------------------------------
# CI pipeline
# ---------------------------------------------------------------
.PHONY: ci
ci: lint typecheck test ## Run full CI pipeline (lint + typecheck + test)

.PHONY: ci-fast
ci-fast: lint test-fast ## Run fast CI pipeline (lint + non-slow tests)

# ---------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------
.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: clean-all
clean-all: clean ## Remove all generated files including checkpoints and logs
	rm -rf checkpoints/ logs/ outputs/
	rm -rf /tmp/alphagalerkin_test_ckpts
