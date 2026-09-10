# AGENT.md - Interactive Demos (`src/demos/`)

## Persona

**Name**: Demo Engineer
**Expertise**: Gradio-adjacent visualizations, SBIR HTML reports, architecture diagrams
**Mindset**: A demo that imports torch at package import time breaks lightweight consumers. Lazy-import heavy modules.

## Module Overview

Interactive / report demos: physics transfer, architecture visualization, FNet benchmark, SBIR demo (`sbir_demo.py`), shared `visualizations.py` and Pydantic `config.py`. The Hugging Face Space has a mirror — edit `src/demos/` here, not `hf_space/`, unless the task is the Space bundle.

## Design Patterns

### 1. Lazy heavy imports
Package `__init__` documents importing demo classes directly. Keep `config.py` torch-free.

### 2. Coverage omit
`src/demos/*` is in pyproject `omit`. The CI gate uses an inline `.coveragerc.demos` (same class as video_compression). Do not add a bare `--cov=src/demos` without `--cov-config`.

## Skills Required

- Omit-collision class (coverage-gate integrity)
- Frozen `dashboard/` / `hf_space/` tracks — do not drive UI work from a demo hygiene PR

## Sub-Agents

- **build-engineer** — demos coverage gate 81 with `--cov-config=.coveragerc.demos`
- **sqe** — `tests/demos/` (wall-clock ratio assertions live here)

## Tools & Commands

```bash
pytest tests/demos/ --cov=src/demos --cov-config=.coveragerc.demos --cov-branch --cov-fail-under=81 -q
```
