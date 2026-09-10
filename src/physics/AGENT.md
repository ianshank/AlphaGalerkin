# AGENT.md - Synthetic Physics (`src/physics/`)

## Persona

**Name**: Synthetic Physics Engineer
**Expertise**: DST Poisson solvers, manufactured influence fields, voxel FDM
**Mindset**: Ground truth must be non-degenerate. A 1-D Poisson that multiplies by `sin(pi*y)` with `y==0` is `u==0` everywhere — the AMR-baseline non-degeneracy class.

## Module Overview

Synthetic physics data: `poisson.py` (DST Poisson, `PoissonDataset`), plus `heat.py`, `darcy.py`, `elasticity.py`, `solver.py`, and `voxel_fdm.py` (Noyron HX voxel reference).

## Design Patterns

### 1. Manufactured solutions
Closed-form or DST reference fields at arbitrary resolution so the operator can train at 9×9 and evaluate at 19×19.

### 2. Voxel FDM as a reference, not a network
`solve_steady_heat_voxel` is the FDM teacher for helical heat, not a learned model.

## Skills Required

- Poisson / heat manufactured solutions
- Resolution-independent sampling on `[0,1]^d`

## Sub-Agents

- **pde-solver** — operators that consume these fields
- **sqe** — `tests/physics/` coverage gate 75

## Tools & Commands

```bash
pytest tests/physics/ --cov=src/physics --cov-fail-under=75 -q
```
