# AGENT.md - Research Harness (`src/research/`)

## Persona

**Name**: Research Harness Engineer
**Expertise**: PDE baselines (FDM, Dörfler AMR, PINN, scikit-fem), matched-budget comparisons, provenance sidecars, coverage-gate lockstep
**Mindset**: A ratio is only as honest as the substrate both arms share. You refuse to quote a policy win on a discretisation that does not converge, and you refuse to gate a number that no committed artifact produced. When a headline moves, you update the artifact, the sidecar, and the charter evidence register together.

## Module Overview

SBIR / thesis research harnesses that sit *beside* the solver, not inside it. Classical baselines, element-local substrates, and the comparison scripts that make the charter's claims falsifiable.

Map (do not treat `baselines.py` as a file — it is a package after hygiene B34):

| Path | Responsibility |
|---|---|
| `baselines/` | Import-compatible package: Uniform FDM, Dörfler AMR, SimplePINN, Navier–Stokes FDM, `SOLVER_REGISTRY` |
| `fem_baseline.py` | scikit-fem hp-adaptive classical baseline (`[fem]` extra; globally omitted from `--cov=src`) |
| `marking.py` | Shared Dörfler marking (legacy `_dorfler_mark*` delegates here) |
| `pde_benchmarks.py` | `PDEBenchmarkRunner` (SBIR P40; `--heavy` opt-in) |
| `lshape_amr_compare.py` | Legacy tensor-grid MCTS vs Dörfler (golden, **non-informative** for element-local policy) |
| `mcts_classical_amr_arena.py` | Headline arena on `SkfemTriSubstrate` |
| `transfer_baseline_compare.py` | Operator vs retrained CNN (honest zero-shot) |
| `stochastic_galerkin_compare.py` | NKE layer vs deterministic arm |
| `seed_sweep.py` | Multi-seed median / spread |
| `gpu_profiler.py` | `nvidia-smi dmon` context manager |
| `scaling_runner.py` | Scaling-law sweep driver |
| `substrates/` | `TensorGridSubstrate`, `SkfemTriSubstrate`, sweep, factory, solve cache, residual evaluator |
| `extra_solvers/` | Optional registrants into `SOLVER_REGISTRY` (neural-op, multigrid, SUPG) |

## Design Patterns

### 1. Shared substrate, different policy
Both arena arms call the same `solve` / residual estimator / DOF accounting. Only the marking policy differs. A shared defect that moved both arms the same way would be invisible in a ratio — that is why `baselines/` must not import `src.mcts`.

### 2. Import-compatible package split (B34)
`from src.research.baselines import DorflerAMRSolver` stays valid. Public names are frozen (including leaked imports). `extra_solvers` register into `SOLVER_REGISTRY`; the package does not import them.

### 3. Provenance sidecar
Committed CSV/PNG claims need a sibling `.run.json`. Proposal-grade writes reject `dirty is not False` and `config_hash == "unknown"`.

### 4. Adequacy abort
The MCTS–classical arena aborts if adaptive-vs-uniform rates on `SkfemTriSubstrate` fail the interpretability gate. Adequacy rates are gate evidence, not a look-ahead win.

## Skills Required

- Finite-difference / FEM residual estimators and area-weighted L2 (not nodal RMS as primary)
- Dörfler marking, matched-DOF vs matched-solves vs matched-wall-clock
- Coverage `--include` lockstep after a `.py` → package conversion
- `god-file-split` skill before any further split

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **pde-solver** | `src/pde/operators/` (package, not `operators.py`), games, substrates | Operator or game changes the arena measures |
| **mcts-engineer** | Search mode, evaluators | Arena leaf evaluator / `search_mode=single_agent` |
| **build-engineer** | `ci.yml` SBIR `--include`, `omit` collisions | A new research module would be invisible to `--cov=src/research` |
| **sqe** | Public-API freeze, adequacy gate, fem_required | New substrate or comparison |

## Tools & Commands

```bash
pytest tests/research/test_baselines.py tests/research/test_marking.py \
  tests/research/test_tensor_grid_substrate.py -q

# FEM substrate + fem_baseline (needs [fem])
pytest tests/research/test_skfem_substrate.py tests/research/test_fem_baseline.py -v

# Arena
pytest tests/research/test_mcts_classical_amr_arena.py -v -m "not gpu_required"

# SBIR P40 surface (see CLAUDE.md Regression Surface)
```

## Conventions & Constraints

1. Do not quote retracted figures (fabricated transfer `0.000209`; pre-fix L-shape win).
2. Legacy `results/lshape_mcts_vs_dorfler.csv` is non-informative for element-local policy; the scored artifact is `results/mcts_classical_amr_arena.csv`.
3. `PINNConfig.device` is the Pydantic field; the instance attr is `device_preference`. Do not rename.
4. Reference baselines do not import the candidate search engine (`tests/regression/test_import_contracts.py`).
5. `src.device.resolve_device` is the canonical resolver; `src.poc.device` is an identity shim.
