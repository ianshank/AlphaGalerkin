# AlphaGalerkin Documentation

The docs are organized by **audience**. If you're new, start with
[Getting Started](getting-started.md); if you're placing a change, start with
[`ARCHITECTURE.md`](../ARCHITECTURE.md).

## Getting started & engineering

| Doc | For |
| --- | --- |
| [Getting Started](getting-started.md) | Clone → install → first run → run a module's tests. |
| [Contributing](../CONTRIBUTING.md) | Workflow, quality gates, coding conventions. |
| [Architecture / repo map](../ARCHITECTURE.md) | Every package, the two-domain split, gotchas. |
| [Glossary](GLOSSARY.md) | Galerkin / MCTS / PDE terminology. |
| [Specs](../specs/README.md) | Spec-driven development (contract-before-code). |
| [Releasing](../RELEASING.md) | Versioning, changelog, and release process. |
| [Templates](templates/) | Module + C4 implementation templates. |

## Architecture

| Doc | Content |
| --- | --- |
| [C4 model (Mermaid)](architecture/c4_mermaid.md) | Context / container / component diagrams. |
| [PDE game C4](architecture/pde_game_c4.md) | C4 for the PDE-solving-as-a-game framework. |
| [Components reference](architecture/components.md) | Component-level reference. |
| [Reusable tools](architecture/reusable_tools.md) | Shared tooling reference. |
| [Architecture Decision Records](adr/README.md) | Numbered ADRs (`0001`, `0002`, …). |

## Research

| Doc | Content |
| --- | --- |
| [Related work](related-work.md) | Novelty-boundary register (with executable guard). |
| [DOE Genesis](doe_genesis/) | Theory, MDP spec, data-management plan for the DOE research track. |
| [Chess engine benchmarking](CHESS_ENGINE_BENCHMARKING.md) | Chess engine uniqueness + match commands. |
| [Galerkin fusion head plan](GALERKIN_FUSION_HEAD_PLAN.md) | Cross-repo (Mouse-Droid-AGI) integration plan. |
| [Training data sources](TRAINING_DATA_SOURCES.md) / [training summary](training_summary.md) | Training references. |
| [PRD — chess self-play](prd/prd-chess-self-play.md) | Product requirements doc. |
| [Transfer results demo](demos/transfer_results.md) | Resolution-transfer demo results. |

## Roadmap & planning

| Doc | Content |
| --- | --- |
| [Implementation plan](IMPLEMENTATION_PLAN.md) | Next-phase implementation plan. |
| [Next steps plan](NEXT_STEPS_PLAN.md) | General roadmap. |
| [ROI implementation plan](ROI_IMPLEMENTATION_PLAN.md) | Tiered next-steps plan with validation gates. |
| [Prompt template](PROMPT_TEMPLATE.md) / [training implementation template](TRAINING_IMPLEMENTATION_TEMPLATE.md) | Agentic-coding templates. |

## Business & proposals

| Doc | Content |
| --- | --- |
| [Business hub](business/README.md) | SBIR/STTR proposal kit, commercialization, partners. |

## History / archive

Point-in-time material kept for provenance — **not** current reference:

- [`archive/`](archive/) — PR-specific runbooks (`PR86_HEADLINE_RUNS.md`), PR
  reviews, timestamped implementation plans, superseded architecture docs
  (`c4_model.md`, `C4_ARCHITECTURE.md`), and the old root `PLANNING.md`.
