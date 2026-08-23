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
is unpublished: no RL-for-AMR work uses an **explicit search tree over refinement
sequences**, and the only prior MCTS + finite-element work, **TreeMesh**
([arXiv:2111.07613](https://arxiv.org/abs/2111.07613)), targets mesh *generation*, a distinct
problem. The delta is deliberately narrow: VDGN
([arXiv:2211.00801](https://arxiv.org/abs/2211.00801)) already refines **anticipatorily** for
features that appear at later times, so what is unoccupied is the explicit search tree and its
transparent compute budget — not multi-step reasoning as such (see
[`docs/business/proposals/PRIOR_ART_REVIEW.md`](docs/business/proposals/PRIOR_ART_REVIEW.md)).

## What's here

| I want to… | Go to |
| --- | --- |
| Install and run something | [Getting Started](docs/getting-started.md) |
| Know what's in scope (and what isn't) | [Project charter](openspec/specs/project-charter/spec.md) |
| Understand the codebase layout | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Browse all documentation | [docs/](docs/README.md) |
| Learn the terminology | [Glossary](docs/GLOSSARY.md) |
| Read the math | [Mathematical Foundation](docs/mathematical-foundation.md) |
| See applications | [Use Cases](docs/use-cases.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) · [specs/](specs/README.md) |
| Track changes | [CHANGELOG.md](CHANGELOG.md) |

## Key features

- **Resolution independence** — one model runs at any resolution (train 9×9,
  evaluate zero-shot at 19×19; committed benchmark MSE ≈ **2.3e-3**, no retraining —
  honestly benchmarked against a CNN retrained at the target resolution, which is
  ~14× *more* accurate; the operator's value is zero-retraining, not peak accuracy.
  Artifacts: [`results/transfer_baseline_compare.csv`](results/transfer_baseline_compare.csv),
  [`config/baselines/transfer_ci.json`](config/baselines/transfer_ci.json);
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
Optional extras: `test-extras`, `fem`, `jax` / `jax-gpu`, `picogk`, `lm-studio`, `docs`
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

## Video Compression

### Codec Performance Benchmarking (Phase 0)

GPU-primary perf harness for `src/video_compression/`. Phase 0 of the self-hosted neural transcoder roadmap; the headline measurement gates every later phase (Phase 1 runtime backends, Phase 2 model zoo, Phase 3 MCTS rate control, Phase 4+ daemon and plugins).

The harness sweeps a Cartesian product of `{resolutions} × {batch_sizes} × {runtime_profiles} × {phases}`, captures per-cell throughput / latency-percentile / VRAM, and (optionally) compares against a recorded baseline with per-metric tolerance overrides.

```bash
# CPU smoke test (CI gate, ~10 s on a single core)
python -m scripts.benchmark_codec run --config config/perf/smoke.yaml

# Single-card headline on the 16 GB primary (RTX 5060 Ti at cuda:0)
python -m scripts.benchmark_codec run \
    --config config/perf/cuda0_headline.yaml \
    --output reports/perf/headline_$(git rev-parse --short HEAD).json

# Dual-card sweep across cuda:0 + cuda:1 (RTX 5060 Ti + RTX 5060)
python -m scripts.benchmark_codec run \
    --config config/perf/default.yaml \
    --output reports/perf/dual_$(git rev-parse --short HEAD).json

# Record a fresh baseline (commit to docs/perf/ — this is the regression-gate ground truth)
python -m scripts.benchmark_codec record-baseline \
    --config config/perf/default.yaml \
    --output docs/perf/baseline_v1.json \
    --hardware-tag rtx5060ti16-rtx5060-8

# Compare a run against a baseline
python -m scripts.benchmark_codec diff \
    --baseline docs/perf/baseline_v1.json \
    --report reports/perf/dual_$(git rev-parse --short HEAD).json
```

Baselines are JSON with explicit schema versioning (`PERF_BASELINE_DOCUMENT_SCHEMA_VERSION`); unversioned files migrate cleanly via `_migrate_baseline_document`. See [docs/perf/README.md](docs/perf/README.md) for the full recording / migration playbook.

The harness is **GPU-primary by design**: `device_preference="cuda"` is the default, and per-profile `device: "cuda:N"` lets a single sweep cover both cards of the reference rig. Set `device_preference: "cpu"` only for CI smoke; the headline measurement requires GPU.

### Phase 1 — Decoder Runtime Backends ✅ COMPLETE

Four decoder runtime backends implemented as Protocol-compliant modules in `src/video_compression/runtime/`:

| Backend | Registry Name | Key Feature | Precision |
|---|---|---|---|
| **PyTorch Eager** | `pytorch-eager` | Baseline, no compilation | FP32 |
| **torch.compile** | `pytorch-compiled` | Inductor graph fusion, CUDA graphs | FP32/FP16/BF16 + autocast |
| **ONNX Runtime** | `onnx-cuda` | In-memory ONNX export + CUDAExecutionProvider | FP32 |
| **TensorRT** | `tensorrt` | torch_tensorrt Dynamo IR, max throughput | FP32/FP16 (BF16→FP16) |

All backends register via `@register_runtime` decorator and are dispatched through the benchmark loop's `_runtime_name_for_profile()` mapping. No hardcoded values — optimization levels, opset versions, and compile modes are configurable, while precision support is backend-dependent and currently follows each runtime's implemented execution path.

```bash
# Run with TensorRT backend (requires CUDA + torch_tensorrt)
python -m scripts.benchmark_codec run --config config/perf/cuda0_headline.yaml

# Full runtime test suite (env-gated skips for missing deps)
pytest tests/video_compression/perf/ tests/video_compression/runtime/ -v
```

---

### Phase 2 — Model Zoo (R-D Lagrangian Sweep) ✅ Phase 2-D COMPLETE

Subpackage `src/video_compression/zoo/` orchestrates an R-D Lagrangian sweep across a heterogeneous-VRAM rig (e.g. `cuda:0=RTX 5060 Ti 16 GiB` + `cuda:1=RTX 5060 8 GiB`). Schedules an arbitrary λ-grid; ships an 8-point grid at [config/video_compression/zoo/lambda_grid.yaml](config/video_compression/zoo/lambda_grid.yaml).

**Phase 2-B** — core zoo schemas, manifest I/O, device planner, filesystem registry.  
**Phase 2-C** — `ZooTrainer` per-entry (fixed-λ, AMP, grad-clip, warmup, `parent_entry_id` warm-start).  
**Phase 2-D** — manifest-level sweep orchestrator + parallel dispatch + subprocess runner. Includes structural numerical stability fixes (monotonic `FactorizedPrior` CDF, GDN positivity, NaN-stable MS-SSIM).

| Module | Responsibility | Coverage |
|---|---|---|
| `config.py` | Pydantic schemas (`ModelZooEntryConfig`, `ModelZooManifestConfig`, `OptimizerConfig`, `SchedulerConfig`) — zero hardcoded values | 100% |
| `manifest.py` | JSON / YAML load / save dispatched by suffix; forward-compat migration via `_migrate_manifest_document` | 98% |
| `device_planner.py` | `scan_devices()` + `assign_devices()` with four strategies: `VRAM_AWARE` (best-fit pack on current headroom), `ROUND_ROBIN`, `SINGLE_DEVICE`, `MANUAL` | 100% |
| `storage.py` | Filesystem `VideoCodecZoo` registry (per-entry `checkpoint.pt` / `entry.json` / `metrics.json`); GCS backend gated for Phase D | 100% |
| `cli_helpers.py` | Shared CLI primitives: `load_dict`, `resolve_path`, `load_codec_config`, `resolve_entry`, `resolve_codec_config_for_entry`, `override_entry`, `resolve_device` | 100% |
| `sweep.py` | `ZooSweep.run()` (serial) + `run_parallel()` (one worker thread per device); `make_subprocess_entry_runner` with `CUDA_VISIBLE_DEVICES` pinning | 96% |

```bash
# Dry-run a manifest (no training, just plans the sweep)
python -m scripts.train_compression_zoo dry-run \
  --manifest config/video_compression/zoo/lambda_grid.yaml \
  --storage-root /tmp/zoo

# Train all entries in parallel (one worker per device)
python -m scripts.train_compression_zoo train \
  --manifest config/video_compression/zoo/lambda_grid.yaml \
  --storage-root ./zoo_outputs \
  --parallel

# Train a single entry by ID
python -m scripts.train_compression_zoo train \
  --manifest config/video_compression/zoo/lambda_grid.yaml \
  --storage-root ./zoo_outputs \
  --only-entry-id lam_0016

# Run the zoo subpackage tests + coverage gate
pytest tests/video_compression/zoo/ tests/scripts/test_train_compression_zoo.py \
  tests/scripts/test_train_compression_zoo_entry.py \
  tests/video_compression/training/test_zoo_trainer.py \
  --cov=src/video_compression/zoo --cov-fail-under=85 -v
```

---
## Testing

The project ships an extensive suite — **8,573 test functions, 9,770 collected**
after parametrisation (measured 2026-08-21, CPU surface) across unit, integration,
property-based, E2E and security categories — with an **85% branch coverage** gate
enforced in CI, plus **34 per-module gates** (e.g. `mcts ≥ 90`, `refinement ≥ 85`,
`poc ≥ 85`, `data ≥ 85`, `demos ≥ 81`, `pde ≥ 75`).

```bash
export COVERAGE_CORE=pytrace          # a torch wheel crashes the default C tracer
pytest -m "not gpu_required"          # CPU-only default surface
ruff check src/ && ruff format --check src/
```

Which tests guard which code path is documented in the **Regression Surface**
table in [`CLAUDE.md`](CLAUDE.md#regression-surface). See [CONTRIBUTING.md](CONTRIBUTING.md)
for the full workflow.

## Project status

Active development (`0.4.0-dev`, pre-release). Shipped work and milestones are in
[`CHANGELOG.md`](CHANGELOG.md); the release process is in [`RELEASING.md`](RELEASING.md).
SBIR/commercialization material lives in [`docs/business/`](docs/business/README.md).

## Mathematical Foundation

### Galerkin Projection

The key insight is treating attention as a **Petrov-Galerkin projection** for solving:

```
Find u ∈ U: ⟨Lu, v⟩ = ⟨f, v⟩  ∀v ∈ V
```

In attention form:

- **Q** (Query): Test function basis
- **K** (Key): Trial function basis
- **V** (Value): Function to project

The projection becomes:

```
Context = K^T V / n     (Monte Carlo integral)
Output = Q × Context    (Reconstruction)
```

### LBB Stability Condition

For convergence, we require the **inf-sup condition**:

```
inf_u sup_v ⟨Lu, v⟩ / (‖u‖ ‖v‖) ≥ β > 0
```

In practice: `dim(Key) ≥ dim(Query)` ensures stability.

### Resolution Transfer

Spectral methods enable zero-shot transfer:

1. **Fourier Encoding**: Position → frequency representation
2. **Spectral Filter**: Anti-alias when changing resolution
3. **Normalization**: Adjust Monte Carlo integral factor

---

## Performance

### Complexity Comparison

| Operation | Standard Attention | Galerkin Attention |
|-----------|-------------------|-------------------|
| 9×9 board | O(81² × d) | O(81 × d²) |
| 19×19 board | O(361² × d) | O(361 × d²) |
| Scaling | Quadratic in N | Linear in N |

### Benchmarks

| Model | Board Size | Inference (ms) | MCTS Sims/sec |
|-------|------------|----------------|---------------|
| Standard | 19×19 | 45 | 180 |
| Galerkin | 19×19 | 28 | 290 |
| Galerkin+FNet | 19×19 | 12 | 670 |

*Benchmarks on NVIDIA RTX 3090, batch size 1*

---

## Directory Structure

```
AlphaGalerkin/
├── src/
│   ├── core/              # Runtime protocols (Evaluator, Game, Operator, Solver) & Registry[T]
│   ├── modeling/          # Neural network components (GalerkinAttention, FNet, FNO, Stability)
│   ├── agents/            # Multi-physics agents, Lifecycle Hooks & Reusable Skills
│   │   ├── lifecycle_hooks.py # HookManager, LoggingHook, MetricsCollector, EarlyStopping
│   │   └── skills/        # BenchmarkSkill, SelfPlaySkill
│   ├── games/             # Game implementations (Chess, Go, PettingZoo)
│   ├── pde/               # PDE Game Framework & Operators (Poisson, Burgers, Navier-Stokes)
│   ├── research/          # SBIR benchmarking infrastructure & baseline solvers
│   ├── training/          # Training pipeline, checkpointing, ReLoBRaLo, constants
│   ├── engines/           # External engine integration (UCI, Match, Elo)
│   ├── math_kernel/       # Mathematical basis primitives (Fourier, Chebyshev)
│   ├── mcts/              # Monte Carlo Tree Search, Evaluators, Gumbel MCTS
│   └── tools/             # Utilities (GTP, CLI, Colab)
├── tests/                 # 7-Tier Test Pyramid (9,770 collected, >= 85% branch coverage gate)
│   ├── sanity/            # Dynamic public module import smoke & CLI help checks
│   ├── security/          # Pickle-RCE payload tests, path containment, GTP fuzzing
│   ├── benchmarks/        # O(N) attention scaling, MCTS throughput, FNet speedup
│   ├── regression/        # Mathematical invariants, backup sign checks, transfer ratio floors
│   ├── core/              # Core protocol conformance & generic registry unit tests
│   ├── claude/            # Deterministic validation of the .claude/ agentic harness
│   ├── demos/             # Benchmark & visualization demos (wired into CI 2026-08-21)
│   ├── notebooks/         # Notebook execution smoke (wired into CI 2026-08-21)
│   ├── pde/               # PDE operators, geometry, time-stepping
│   ├── training/          # Trainer, loss properties, numerical stability
│   ├── modeling/          # Attention properties, Fourier features
│   ├── games/             # Chess, Go, PettingZoo adapter
│   ├── engines/           # UCI, match, Elo tests
│   └── e2e/               # User journey end-to-end workflows
├── config/
│   ├── schemas.py         # Pydantic configs
│   ├── proposals/         # SBIR configs (Navy, DOE, NSF, AFWERX, DARPA D2P2)
│   └── benchmarks/        # sbir_suite.yaml
├── docs/
│   ├── architecture/      # C4 diagrams, gap analysis, hardening reports
│   ├── migration/         # v0.3 to v0.4 migration guide
│   └── proposals/         # SBIR templates, IP strategy, budgets
└── pyproject.toml
```

---

## SBIR Positioning

AlphaGalerkin addresses a **verified novelty gap**: the methodological
delta — MCTS *multi-step look-ahead* for basis selection and error-driven refinement —
is unpublished: no RL-for-AMR work uses an **explicit search tree over refinement
sequences**, and the only prior MCTS + finite-element work, **TreeMesh**
([arXiv:2111.07613](https://arxiv.org/abs/2111.07613)), targets mesh *generation*, a distinct
problem. The delta is deliberately narrow: VDGN
([arXiv:2211.00801](https://arxiv.org/abs/2211.00801)) already refines **anticipatorily** for
features that appear at later times, so what is unoccupied is the explicit search tree and its
transparent compute budget — not multi-step reasoning as such (see
[`docs/business/proposals/PRIOR_ART_REVIEW.md`](docs/business/proposals/PRIOR_ART_REVIEW.md)).
The SBIR reauthorization (S. 3971) extends the program through 2031 with backlogged FY2026 funds.

| Solicitation | Agency | Phase | Funding | Config |
|---|---|---|---|---|
| **AFWERX Open 26.1** | USAF | I | $75K / 3mo | `config/proposals/afwerx_open.yaml` |
| **NSF SBIR Pitch** | NSF | I | $305K / 12mo | `config/proposals/nsf_sbir.yaml` |
| **Navy N252-088** | NAVAIR | I | $150-250K / 6mo | `config/proposals/navy_n252_088.yaml` |
| **DOE ASCR C59-01** | DOE | I | $200-250K / 12mo | `config/proposals/doe_ascr_c59.yaml` |
| **DARPA Direct-to-Phase-II** | DARPA STO | II | $750K-$1.5M / 24mo | `config/proposals/darpa_d2p2.yaml` |

### Proposal Infrastructure
- **Registration**: [SAM.gov Guide](docs/business/proposals/SAM_REGISTRATION_GUIDE.md) (UEI, CAGE, NAICS 541715)
- **Timeline**: [Submission Calendar](docs/business/proposals/SUBMISSION_TIMELINE.md) with Gantt chart
- **Contacts**: [Program Offices](docs/business/proposals/PROGRAM_OFFICES.md) (Tier 1 + Tier 2)
- **Budgets**: [Budget Templates](docs/business/proposals/BUDGET_TEMPLATES.md) (DoD, NSF, AFWERX, DARPA)
- **IP Protection**: [IP Strategy](docs/business/proposals/IP_STRATEGY.md) (3 provisional patents, trade secrets)
- **Competitive Analysis**: [Landscape](docs/business/proposals/COMPETITIVE_LANDSCAPE.md) | [Differentiation](docs/business/proposals/DIFFERENTIATION_MATRIX.md)
- **Valuation**: [Framework](docs/business/proposals/VALUATION_FRAMEWORK.md) | [M&A Landscape](docs/business/proposals/MA_LANDSCAPE.md)

### Run SBIR Benchmarks
```bash
# End-to-end demo with convergence plots and comparison tables
python -m scripts.run_sbir_demo --config config/benchmarks/sbir_suite.yaml

# Custom output
python -m scripts.run_sbir_demo --output-dir outputs/navy_demo --formats json latex markdown

# Opt into heavy refinement levels (e.g. 65 536-DOF Poisson L-shaped)
# to demonstrate the P40's 24 GiB VRAM advantage. Default keeps CI fast.
python -m scripts.run_sbir_demo --heavy --output-dir outputs/sbir_demo_v2

# Tesla P40 high-resolution PINN vs NS-FDM comparison.
# Loads config/benchmarks/sbir_p40.yaml; every PINN parameter is
# config-driven and overridable via CLI flags.
python -m scripts.run_sbir_p40                              # default profile
python -m scripts.run_sbir_p40 --device cuda:1              # pin to a different GPU
python -m scripts.run_sbir_p40 --n-epochs 1000 --skip-cpu   # short GPU-only run
```

The P40 driver embeds **GPU utilisation telemetry** (mean SM-util %, mean
memory-util %, peak FB-MiB) in `SolverResult.metadata["gpu_profile"]`
when `nvidia-smi` is on PATH. Skips silently on CI / no-GPU hosts so the
same code path is safe everywhere.

---

## Next Steps

### Near-Term (v0.4)
- [x] ~~SBIR demo script~~ (`scripts/run_sbir_demo.py` with convergence plots, LaTeX/Markdown reports)
- [x] ~~BaseTrainer consolidation~~ (`src/training/base_trainer.py` with AMP, gradient clipping, LR scheduling)
- [x] ~~SBIR proposal infrastructure~~ (SAM guide, budgets, timeline, program offices, IP strategy)
- [x] ~~SBIR P40 benchmark hardening~~ (`scripts/run_sbir_p40.py` config-driven driver, `GpuUtilizationProfiler`, AMR escapes 18-DOF ceiling, NS-FDM Taylor-Green parity, PINN device knob)
- [ ] Multi-field PDE support (extending ModelOutput for vector fields)
- [ ] Migrate Trainer and OperatorTrainer to BaseTrainer inheritance
- [ ] PETSc/MFEM compatibility layer for DOE ASCR proposals
- [ ] Capture proposal-grade Tesla P40 numbers from `scripts/run_sbir_p40.py` once a sm_61-compatible PyTorch wheel is available

### Medium-Term (v0.5)
- [ ] 3D tetrahedral domain geometry support
- [ ] Distributed benchmark runner (multi-node SBIR suite)
- [ ] Uncertainty quantification for PDE solutions
- [ ] PettingZoo training loop for swarm games
- [ ] Pitch deck generation automation

### Long-Term (v1.0)
- [ ] SBIR Phase I proposal submissions (AFWERX, NSF, Navy)
- [ ] DARPA Direct-to-Phase-II package submission
- [ ] Production ONNX deployment pipeline
- [ ] Multi-physics coupling (fluid-structure interaction)
- [ ] Publication: "MCTS-Guided Galerkin Methods for Adaptive PDE Solving" (NeurIPS ML4PhysicalSciences)

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
