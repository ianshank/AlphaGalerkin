# AlphaGalerkin

**Resolution-independent operator learning + MCTS for board games (Go, Chess) and PDE solving.**

AlphaGalerkin uses Galerkin Transformers and Monte Carlo Tree Search to solve
board games and partial differential equations without retraining across
resolutions. The two domains share one abstraction — **MCTS** — adapted per
domain.

## Start here

- [Getting Started](getting-started.md) — install and run your first scenario.
- [Repository map](https://github.com/ianshank/AlphaGalerkin/blob/HEAD/ARCHITECTURE.md) — how the codebase is organized.
- [Glossary](GLOSSARY.md) — Galerkin / MCTS / PDE terminology.
- [Mathematical foundation](mathematical-foundation.md) — the Galerkin-projection and LBB-stability theory.

## Explore

- [Use cases](use-cases.md) — applications across both domains.
- [Architecture (C4)](architecture/c4_mermaid.md) — context / container / component diagrams.
- [Decision records](adr/README.md) — numbered ADRs.
- [Related work](related-work.md) — the novelty-boundary register.

## API reference

Auto-generated from docstrings for the core public packages:
[MCTS](api/mcts.md) · [PDE](api/pde.md) · [Modeling](api/modeling.md) ·
[Math kernel](api/math_kernel.md) · [Refinement](api/refinement.md).

## Contribute

See [CONTRIBUTING](https://github.com/ianshank/AlphaGalerkin/blob/HEAD/CONTRIBUTING.md)
and the [spec-driven workflow](https://github.com/ianshank/AlphaGalerkin/blob/HEAD/specs/README.md).
