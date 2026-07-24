# Contributing to AlphaGalerkin

Thanks for your interest in contributing! This project favors **small, verified,
spec-driven changes** over large speculative ones. This guide points you at the
conventions already used in the repo rather than inventing new process.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 1. Development setup

```bash
git clone https://github.com/ianshank/AlphaGalerkin.git
cd AlphaGalerkin

python -m venv venv && source venv/bin/activate   # Linux/Mac
pip install -e ".[dev]"

# Install the git hooks so lint/format/type checks run before each commit
pre-commit install
```

See [`docs/getting-started.md`](docs/getting-started.md) for a clone-to-first-run
walkthrough and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the repository map.

## 2. Spec-driven development

Non-trivial features start as a **markdown spec** before any code:

1. Copy [`specs/TEMPLATE.spec.md`](specs/TEMPLATE.spec.md) to `specs/<feature>.spec.md`.
2. Fill in the **Data Contract** (Pydantic fields), **Acceptance Criteria**
   (Given/When/Then, each mapped to a test), and **Thresholds** (reusing the
   canonical `src.poc.config.MetricThreshold` — do **not** invent a parallel schema).
3. Write the tests, then the code, then an AQA test asserting spec ↔ config agreement.
4. Register the guarding test command in the **Regression Surface** table in
   [`CLAUDE.md`](CLAUDE.md).

The full workflow is documented in [`specs/README.md`](specs/README.md). The
`spec-new` skill scaffolds this for you.

## 3. Quality gates (run locally before pushing)

These mirror what CI enforces (`.github/workflows/ci.yml`). Run them before opening a PR:

```bash
# Lint + format (hard gates in CI)
ruff check src/
ruff format --check src/

# Type check (informational — CI runs mypy with continue-on-error)
mypy src/ --strict

# The regression surface for the code path you touched — see the table in CLAUDE.md.
# Example (PDE / scenario changes):
pytest tests/pde/ tests/poc/ -m "not gpu_required"

# Or run all the hooks at once
pre-commit run --all-files
```

### Coverage

CI enforces **85% branch coverage** globally, plus **per-module gates** (e.g.
`mcts ≥ 90`, `refinement ≥ 85`, `pde ≥ 75`, `distributed ≥ 60`, chess `≥ 80`). New
code needs tests that keep the changed module above its gate. The
[`coverage-gate`](.claude/) skill runs the exact per-module command for you.

> **Note:** coverage in this environment requires `COVERAGE_CORE=pytrace` (a torch
> wheel crashes the default C tracer).

## 4. Coding conventions

- **Style:** enforced by Ruff (line length 100; pydocstyle `D`, annotations `ANN`,
  bugbear `B`, and more — see `[tool.ruff.lint]` in `pyproject.toml`). Let
  `ruff format` do the formatting.
- **Types:** strict typing; `mypy --strict` with narrow, documented per-module
  carve-outs in `pyproject.toml`. New code should not add carve-outs without reason.
- **No hardcoded values:** configuration is Pydantic-validated. Surface magic
  numbers as typed config fields with docstrings.
- **Structured logging** via `structlog`, not `print`.
- **Tests:** property-based tests (Hypothesis) for mathematical operators;
  `@pytest.mark.gpu_required` for anything needing CUDA (auto-skipped on CPU CI).

## 5. Commits & pull requests

- **Commit messages:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`…), validated by Commitizen in
  the pre-commit hooks.
- **Changelog:** add a bullet under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md).
- **Branch:** work on a feature branch; open a PR against `main`.
- **PR description:** fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
  Keep PRs focused and reviewable.
- **Releases:** see [`RELEASING.md`](RELEASING.md) for the versioning/release process.

## 6. Where things live

The repository map, the two-domain (game-AI vs. PDE) split, and the
core-vs-experimental maturity labels are in [`ARCHITECTURE.md`](ARCHITECTURE.md).
Per-module developer guides are the `AGENT.md` files inside each `src/` package.

Questions? See [SUPPORT.md](SUPPORT.md).
