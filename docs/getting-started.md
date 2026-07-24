# Getting Started

This walks you from a fresh clone to running your first scenario and the tests
that guard the code you'll touch.

## Prerequisites

- Python **3.10+** (CI tests 3.10 / 3.11 / 3.12).
- PyTorch 2.0+ (CUDA 12.x recommended for GPU paths; CPU works for everything CI runs).

## 1. Clone and install

```bash
git clone https://github.com/ianshank/AlphaGalerkin.git
cd AlphaGalerkin

python -m venv venv && source venv/bin/activate    # Linux/Mac
pip install -e ".[dev]"

# Optional: install the pre-commit hooks (ruff, format, yamllint, commitizen…)
pre-commit install
```

Optional extras (install only what you need):

| Extra | Enables |
| --- | --- |
| `dev` | Test + lint toolchain (start here). |
| `test-extras` | FEM baseline, ONNX export/validate, PettingZoo. |
| `jax` / `jax-gpu` | JAX backend + cross-backend tests. |
| `picogk` | Leap 71 PicoGK voxel/SDF kernel (Noyron HX). |
| `lm-studio` | OpenAI-compatible local-LLM client (LLM-prior MCTS). |
| `docs` | MkDocs docs-site toolchain. |

## 2. Run your first scenario (CPU-safe)

The PoC framework runs configuration-driven scenarios:

```bash
# List available scenarios
python -m src.poc.cli list

# Quick validation suite (~5 min, CPU)
python -m src.poc.cli run --config config/scenarios/poc_quick.yaml

# Details for one scenario
python -m src.poc.cli info transfer
```

## 3. Verify resolution independence

```bash
python -m src.tools.verify_invariance --train-size 9 --infer-size 19
```

## 4. Run the tests for what you touched

CI enforces **85% branch coverage** globally plus per-module gates. Rather than
running the whole suite, run the **Regression Surface** command block for your
code path — the table is in [`CLAUDE.md`](https://github.com/ianshank/AlphaGalerkin/blob/HEAD/CLAUDE.md). Examples:

```bash
# Coverage in this environment needs pytrace (a torch wheel crashes the C tracer)
export COVERAGE_CORE=pytrace

# Solver wiring
pytest tests/alphagalerkin/test_solver.py -v

# MCTS evaluator protocol
pytest tests/mcts/test_evaluator.py -v

# PDE end-to-end
pytest tests/integration/test_pde_e2e.py -v

# CPU-only default surface (skip GPU tests)
pytest -m "not gpu_required"
```

GPU-only tests are marked `@pytest.mark.gpu_required` and auto-skip on CPU.

## 5. Before you commit

```bash
ruff check src/
ruff format --check src/
pre-commit run --all-files
```

## Where to go next

- [Architecture / repository map](https://github.com/ianshank/AlphaGalerkin/blob/HEAD/ARCHITECTURE.md)
- [Contributing](https://github.com/ianshank/AlphaGalerkin/blob/HEAD/CONTRIBUTING.md) and [spec-driven development](https://github.com/ianshank/AlphaGalerkin/blob/HEAD/specs/README.md)
- [Glossary](GLOSSARY.md)
