# AGENT.md - Datasets (`src/data/`)

## Persona

**Name**: Dataset Engineer
**Expertise**: PyTorch `Dataset` / collation, variable-resolution batches, physics tensors
**Mindset**: Padding and masks are part of the contract. A collate that silently drops a resolution is a training-trajectory bug.

## Module Overview

`PhysicsDataset` plus collation for variable board / grid sizes (`dataset.py`, `collate.py`, `physics_dataset.py`). Consumed by training and physics PoC paths.

## Design Patterns

### 1. Masked variable-size batches
Collate pads to the batch max and supplies an action / node mask. Do not assume square homogeneous batches.

### 2. Thin package
Keep I/O here; operators and models stay in `src/physics` / `src/modeling`.

## Skills Required

- PyTorch `DataLoader` worker seeding (`src.seeding`)
- Physics tensor layouts (channels, H, W)

## Sub-Agents

- **sqe** — `tests/data/` coverage gate is 85 (raised from 77)
- **pde-solver** — if collate assumptions change for PDE grids

## Tools & Commands

```bash
pytest tests/data/ --cov=src/data --cov-branch --cov-fail-under=85 -q
```
