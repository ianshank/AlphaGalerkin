# AlphaGalerkin — Architecture & Repository Map

This is the **canonical map** of the repository: what every top-level directory
and `src/` package is for, how the pieces depend on each other, and the naming
gotchas a new engineer will hit. It supersedes the older "Directory Structure"
block in [`CLAUDE.md`](CLAUDE.md).

> **Source-of-truth policy.** To stop this map from drifting the way the old one
> did, the `src/` package list below is enforced by a test
> ([`tests/docs/test_architecture_map.py`](tests/docs/test_architecture_map.py)):
> add or remove a `src/` package and CI fails until this file is updated.

## What AlphaGalerkin is

AlphaGalerkin applies **resolution-independent continuous operator learning**
(Galerkin Transformers, FNet mixing) and **Monte Carlo Tree Search** to two
problem domains that share one search abstraction:

- **Board games** (Go, Chess) — zero-shot transfer across board sizes.
- **PDE solving** — MCTS-guided adaptive mesh refinement and Galerkin basis
  selection for computational physics.

The two domains look unrelated in the directory tree, but the connective tissue
is **MCTS**: `src/mcts/` is the domain-agnostic search engine, and each domain
adapts its problem into that engine (`src/games/` via `GameInterface`,
`src/pde/` via `src/pde/mcts_adapter.py`).

## Documentation hierarchy (which doc owns what)

| Doc | Owns |
| --- | --- |
| [`README.md`](README.md) | Front door: value prop, install, quickstart, links out. |
| **`ARCHITECTURE.md`** (this file) | The repository map — packages, layering, gotchas. Source of truth for layout. |
| [`docs/README.md`](docs/README.md) | Index of the whole `docs/` tree by audience. |
| [`docs/architecture/c4_mermaid.md`](docs/architecture/c4_mermaid.md) | Diagram-level C4 architecture (Mermaid). |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records (numbered). |
| [`CLAUDE.md`](CLAUDE.md) | Agent operational context + the Regression Surface table. Points here for layout. |
| Root [`AGENT.md`](AGENT.md) + `src/*/AGENT.md` | Agent/developer guides; per-package conventions and "adding a new X" checklists. |
| [`specs/`](specs/README.md) | Per-feature specs (data contract + acceptance criteria + thresholds), written before code. |

## Top-level layout

| Path | What it is |
| --- | --- |
| `src/` | The main Python package (see the package map below). |
| `tests/` | Test suite mirroring `src/` plus cross-cutting suites (`integration/`, `e2e/`, `regression/`, `security/`, `docs/`). |
| `config/` | **The** configuration tree: training YAMLs + `scenarios/`, `agents/`, `baselines/`, `benchmarks/`, `presets/`, and Pydantic `schemas.py`. |
| `specs/` | Spec-driven-development specs (`TEMPLATE.spec.md` + per-feature specs). |
| `docs/` | Documentation tree — see [`docs/README.md`](docs/README.md) for the index. |
| `scripts/` | CLI entry-point scripts (train, evaluate, benchmark runners). |
| `notebooks/` | Jupyter notebooks (Colab, Darcy PoC, demos). |
| `dashboard/` | Gradio web UI. |
| `hf_space/` | Hugging Face Space **mirror** — its own `app.py`, `src/`, and `README.md` (see gotchas). |
| `results/` | Committed benchmark output artifacts (CSV/PNG). |
| `claude-code-platform/` | A nested sub-project (its own `pyproject.toml`/tests) — see gotchas. |
| `.github/` | CI workflows and community-health templates. |
| `.claude/` | Claude Code project scaffolding (hooks, skills, subagents, slash commands). |

## `src/` package map

Maturity legend: **core** = production solver/search path; **support** =
infrastructure the core depends on; **experimental** = PoC / prototyping / demo
code not on the production import path.

<!-- package-map:start -->
| Package | Domain | Maturity | Purpose |
| --- | --- | --- | --- |
| `src/mcts/` | shared | core | Monte Carlo Tree Search engine (PUCT + Gumbel), FNet-accelerated rollouts, `Evaluator`/`GameInterface` protocols. |
| `src/modeling/` | shared | core | Neural architectures — Galerkin & softmax attention, FNet blocks, LBB stability guard, the full model + heads. |
| `src/math_kernel/` | pde | core | Basis functions and integral approximations (Chebyshev/Fourier), Torch + optional JAX. |
| `src/pde/` | pde | core | PDE-solving-as-a-game framework: operators, geometry/SDF, `mcts_adapter`, `games/`, `stochastic/`, time-stepping. |
| `src/refinement/` | pde | core | Domain-free sequential-refinement engine (`RefinementGame` + adapter + registry) that `src/pde/` implements. |
| `src/alphagalerkin/` | pde | core | Unified `AlphaGalerkinSolver` wrapper matching the baseline protocol for apples-to-apples benchmarking. |
| `src/training/` | shared | core | Training infrastructure — trainers, losses, replay buffer, self-play, checkpointing. |
| `src/games/` | game-ai | core | Multi-game support — abstract `GameInterface`, registry, Go/Chess implementations. |
| `src/engines/` | game-ai | support | External chess-engine (UCI/Stockfish) integration, Elo, match orchestration. |
| `src/tournament/` | game-ai | support | Tournament scheduling/formats, Elo ratings, match management. |
| `src/analysis/` | game-ai | support | Game analysis/auditing — position evaluation, game review, pattern recognition. |
| `src/curriculum/` | game-ai | support | Curriculum-learning scheduler (board-size progression 9→13→19). |
| `src/physics/` | pde | support | Synthetic physics data generation (Poisson ground truth). |
| `src/research/` | pde | support | SBIR benchmark harness — FDM/AMR/PINN baselines, transfer validation, reporting. |
| `src/data/` | shared | support | Dataset loading (`PhysicsDataset`) and collation. |
| `src/backend/` | shared | support | PyTorch/JAX backend abstraction (migration-era; both coexist). |
| `src/distributed/` | shared | support | Multi-node DDP/NCCL distributed training infrastructure. |
| `src/deployment/` | shared | support | Model export/deploy — ONNX conversion, quantization, ONNX Runtime wrapper. |
| `src/agents/` | shared | support | Agentic layer — `BaseAgent` lifecycle, research-loop orchestrator, scaffold CLI. |
| `src/integrations/` | shared | support | Optional third-party integrations gated behind extras (e.g. `lm_studio/`). |
| `src/poc/` | shared | support | Proof-of-Concept scenario framework — config-driven runner, registry, scenarios, tuning, statistics. |
| `src/templates/` | shared | support | Reusable module-development scaffolding (base config, registry, logging, CLI). |
| `src/tools/` | shared | support | Utility tools/CLI (GTP engine, verification scripts). |
| `src/experiments/` | pde | experimental | Physics PoC experiments (Poisson supervised training, zero-shot transfer, FNet benchmark, CNN baseline). |
| `src/demos/` | shared | experimental | Interactive demos for the HF Space. |
| `src/prototyping/` | shared | experimental | Fast-prototyping utilities — **not** imported by core production paths (only its own tests + `hf_space/`). |
<!-- package-map:end -->

Plus `src/constants.py` (centralized numerical constants) and `src/__init__.py`.

## Layering (dependency direction)

```
             ┌──────────────────────────────┐
             │        src/mcts/ (search)     │  domain-agnostic
             └───────────────┬──────────────┘
              GameInterface   │   mcts_adapter
        ┌───────────────┐     │     ┌────────────────────────────┐
        │  src/games/   │     │     │   src/refinement/ (engine)  │
        │ (Go / Chess)  │     │     │        ▲  implements         │
        └───────┬───────┘     │     │  src/pde/  (operators/games)│
                │             │     │        ▲  wraps              │
                │             │     │  src/alphagalerkin/ (solver) │
                └─────────────┴─────┴────────────────────────────┘
                     src/modeling/ + src/training/ (shared)
```

## Naming & structure gotchas

- **`config/` (real) — there is no `configs/`.** The former orphan
  `configs/darcy_poc.yaml` was folded into `config/scenarios/`; `config/` with
  its Pydantic `schemas.py` is the single configuration root.
- **`tests/integration/` vs. `tests/integrations/`** — one letter apart, different
  meaning. `tests/integration/` = end-to-end / centaur integration tests;
  `tests/integrations/` = tests for `src/integrations/` (LM Studio, backends).
  These are **not** renamed (CI references both by path).
- **Two domains, one engine.** Chess/Go modules (`engines/`, `tournament/`,
  `analysis/`, `curriculum/`) sit beside the PDE core; the shared abstraction is
  MCTS, not a shared directory name.
- **`prototyping/` is non-production.** It is imported only by its own tests and
  the `hf_space/` mirror — never by the core `src/` production paths.
- **`hf_space/` mirrors `src/`** and ships its **own** `hf_space/README.md`,
  which is the landing page rendered on Hugging Face. It is intentionally
  **not** kept in sync with the root `README.md`; edit it separately.
- **`claude-code-platform/` is a repo-within-a-repo** with its own
  `pyproject.toml`, `CLAUDE.md`, and tests. It is out of scope for the root
  package build (`[tool.setuptools.packages.find]` only includes
  `src*`/`config*`/`dashboard*`) and for the docs site.
- **`results/` holds committed artifacts** (CSV/PNG) referenced by specs and the
  changelog; they are outputs kept in-tree deliberately, not build cruft.
- **Business/proposal prose lives under [`docs/business/`](docs/business/)**;
  the machine-readable SBIR *configs* stay in `config/proposals/` (loaded by
  `scripts/run_sbir_*.py`).

## Where to start

- New contributor: [`docs/getting-started.md`](docs/getting-started.md) → [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Adding a feature: write a spec ([`specs/README.md`](specs/README.md)) first.
- Understanding the math: [`docs/GLOSSARY.md`](docs/GLOSSARY.md) and the
  Mathematical Foundation section of the docs.
