# AlphaGalerkin

**Resolution-independent operator learning + MCTS for board games (Go, Chess) and PDE solving.**

[![CI](https://github.com/ianshank/AlphaGalerkin/actions/workflows/ci.yml/badge.svg)](https://github.com/ianshank/AlphaGalerkin/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ianshank/AlphaGalerkin/graph/badge.svg)](https://codecov.io/gh/ianshank/AlphaGalerkin)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](.pre-commit-config.yaml)

AlphaGalerkin uses **Galerkin Transformers** and **Monte Carlo Tree Search** to
solve two classes of problems without retraining across resolutions:

1. **Board games** (Go, Chess) — zero-shot transfer between board sizes (train 9×9, play 19×19).
2. **PDE solving** — MCTS-guided adaptive mesh refinement and Galerkin basis selection for computational physics.

The two domains share one abstraction: **MCTS** (`src/mcts/`), adapted per domain
via `GameInterface` (games) and `src/pde/mcts_adapter.py` (PDEs). The methodological
delta — MCTS *multi-step look-ahead* for basis selection and error-driven refinement —
is unpublished: the AMR-RL literature is uniformly *single-step*, and the only prior
MCTS + finite-element work, **TreeMesh** ([arXiv:2111.07613](https://arxiv.org/abs/2111.07613)),
targets mesh *generation*, a distinct problem (see
[`docs/business/proposals/PRIOR_ART_REVIEW.md`](docs/business/proposals/PRIOR_ART_REVIEW.md)).

## What's here

| I want to… | Go to |
| --- | --- |
| Install and run something | [Getting Started](docs/getting-started.md) |
| Understand the codebase layout | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Browse all documentation | [docs/](docs/README.md) |
| Learn the terminology | [Glossary](docs/GLOSSARY.md) |
| Read the math | [Mathematical Foundation](docs/mathematical-foundation.md) |
| See applications | [Use Cases](docs/use-cases.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) · [specs/](specs/README.md) |
| Track changes | [CHANGELOG.md](CHANGELOG.md) |

## Key features

- **Resolution independence** — one model runs at any resolution (train 9×9,
  evaluate zero-shot at 19×19; measured MSE ≈ 4e-4, no retraining — honestly
  benchmarked against a CNN retrained at the target resolution,
  [`specs/transfer_baseline_compare.spec.md`](specs/transfer_baseline_compare.spec.md)).
- **O(N) attention** — Galerkin (Petrov-Galerkin projection) instead of O(N²) softmax.
- **Fast MCTS rollouts** — FNet FFT mixing (O(N log N)) for batch leaf evaluation.
- **Provable stability** — LBB / inf-sup condition monitored during training.
- **Spec-driven & agentic tooling** — every feature starts as a
  [spec](specs/README.md); [`.claude/`](.claude/) ships hooks, skills, and subagents.

## Installation

```bash
git clone https://github.com/ianshank/AlphaGalerkin.git
cd AlphaGalerkin
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.0+ (CUDA 12.x recommended for GPU paths).
Optional extras: `test-extras`, `jax` / `jax-gpu`, `picogk`, `lm-studio`, `docs`
(see [Getting Started](docs/getting-started.md#1-clone-and-install)).

## Quick start

```python
import torch
from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel

config = OperatorConfig(d_model=256, n_heads=8, n_galerkin_layers=6, n_softmax_layers=2)
model = AlphaGalerkinModel(config)

board = torch.randn(1, 17, 19, 19)          # (batch, planes, H, W)
output = model(board)
print(output.policy_logits.shape)            # (1, 362) — 361 moves + pass
print(output.value.item())                   # value in [-1, 1]
```

Run a configuration-driven PoC scenario (CPU-safe):

```bash
python -m src.poc.cli list                                       # list scenarios
python -m src.poc.cli run --config config/scenarios/poc_quick.yaml
```

More: [Getting Started](docs/getting-started.md) · [Use Cases](docs/use-cases.md).

## Architecture

A continuous embedding maps the discrete board to Fourier features on `[0,1]²`; a
Galerkin+FNet **strategy body** models global influence in O(N); a softmax
**tactical head** preserves injectivity for local reading; policy and value heads
produce the outputs.

- Repository map and layering: [ARCHITECTURE.md](ARCHITECTURE.md)
- C4 diagrams (Mermaid): [docs/architecture/c4_mermaid.md](docs/architecture/c4_mermaid.md)
- The math: [docs/mathematical-foundation.md](docs/mathematical-foundation.md)

## Testing

The project ships an extensive suite (**7,000+ test functions** across unit,
integration, property-based, E2E, and security categories) with an **85% branch
coverage** gate enforced in CI, plus per-module gates (e.g. `mcts ≥ 90`,
`refinement ≥ 85`, `pde ≥ 75`).

```bash
export COVERAGE_CORE=pytrace          # a torch wheel crashes the default C tracer
pytest -m "not gpu_required"          # CPU-only default surface
ruff check src/ && ruff format --check src/
```

Which tests guard which code path is documented in the **Regression Surface**
table in [`CLAUDE.md`](CLAUDE.md#regression-surface). See [CONTRIBUTING.md](CONTRIBUTING.md)
for the full workflow.

## Project status

Active development (`0.1.0`, pre-release). Shipped work and milestones are in
[`CHANGELOG.md`](CHANGELOG.md); the release process is in [`RELEASING.md`](RELEASING.md).
SBIR/commercialization material lives in [`docs/business/`](docs/business/README.md).

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md), the
[Code of Conduct](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) for
vulnerability reporting. Questions: [SUPPORT.md](SUPPORT.md).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use AlphaGalerkin in your research, please cite it (see
[`CITATION.cff`](CITATION.cff)):

```bibtex
@software{alphagalerkin2026,
  title  = {AlphaGalerkin: Resolution-Independent AI for Games and PDE Solving via MCTS-Guided Galerkin Methods},
  author = {Cruickshank, Ian},
  year   = {2026},
  url    = {https://github.com/ianshank/AlphaGalerkin}
}
```

## Acknowledgments

- AlphaGo / AlphaZero teams at DeepMind for foundational work
- The Galerkin Transformer and FNet authors for the mathematical framework
- The Go and scientific-ML research communities
