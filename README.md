# AlphaGalerkin

**Resolution-Independent AI for Games and PDE Solving using Continuous Operator Learning**

AlphaGalerkin uses Galerkin Transformers and MCTS to solve two classes of problems without retraining across resolutions:

1. **Board Games** (Go, Chess): Zero-shot transfer between board sizes (train 9x9, play 19x19)
2. **PDE Solving**: MCTS-guided adaptive mesh refinement and basis selection for computational physics

The core methodological delta — MCTS *multi-step look-ahead* for Galerkin basis selection and error-driven refinement — is unpublished (the AMR-RL literature is uniformly *single-step*; the only prior MCTS+finite-element work, TreeMesh, targets mesh *generation*, a distinct problem — see [`docs/business/proposals/PRIOR_ART_REVIEW.md`](docs/business/proposals/PRIOR_ART_REVIEW.md)), positioning AlphaGalerkin for SBIR funding in the $50B+ simulation market.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Mathematical Foundation](#mathematical-foundation)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

### The Problem

Both game AI and computational physics face a **resolution lock-in** problem:

- **Game AI**: A model trained on 19x19 Go cannot play 9x9 without retraining. CNNs are tied to fixed grid sizes.
- **PDE Solvers**: Mesh refinement is myopic (single-step error indicators). No multi-step planning exists for optimal mesh/basis selection.
- **Industry**: The $50B simulation software market lacks AI-guided adaptive methods with mathematical convergence guarantees.

### Our Solution

AlphaGalerkin treats any problem domain as a **continuous space** Omega = [0,1]^d, then applies:

- **Galerkin Attention**: O(N) complexity via Petrov-Galerkin projection (not O(N^2) softmax)
- **MCTS Planning**: Multi-step look-ahead for mesh refinement, basis selection, and move search
- **LBB Stability**: Provable convergence via inf-sup condition monitoring during training
- **Zero-Shot Transfer**: One model runs at any resolution — train on 9x9, evaluate zero-shot on 19x19 (measured MSE ~4e-4, no retraining). Honestly benchmarked against a CNN retrained at the target resolution (`specs/transfer_baseline_compare.spec.md`).
- **FFT Mixing**: O(N log N) FNet blocks for fast MCTS rollouts (5x+ speedup)

---

## Key Features

### Resolution Independence

```python
# Train on 9x9
model.fit(board_9x9_dataset)

# Play on 19x19 without retraining
model.adapt_resolution(source_size=9, target_size=19)
move = model.predict(board_19x19)
```

### O(N) Attention Complexity

Traditional attention is O(N²). Galerkin attention achieves O(N) through Petrov-Galerkin projection:

```
Standard:  O(361² × d) = O(130,321 × d)  for 19x19
Galerkin:  O(361 × d²) = O(361 × d²)     for 19x19
```

### Fast MCTS Rollouts

FNet mixing replaces attention with FFT operations:

```
Softmax Attention: O(N²)
FNet Mixing:       O(N log N)
Speedup:           ~5x for leaf evaluation
```

### Mathematical Rigor

Built on solid mathematical foundations:

- **Fredholm Integral Equations**: Model influence as Green's function solutions
- **LBB Stability**: Guaranteed convergence via inf-sup condition monitoring
- **Spectral Methods**: Proper anti-aliasing for resolution transfer

### Spec-Driven Development & Agentic Tooling

- **Specs before code** ([`specs/`](specs/README.md)): every feature starts as a markdown spec
  (data contract + acceptance criteria + `MetricThreshold`s), then tests, then code.
- **Claude Code project scaffolding** ([`.claude/`](.claude/)): a SessionStart bootstrap hook,
  reusable skills (`spec-new`, `regression-surface`, `coverage-gate`, `new-pde-operator`),
  persona subagents, and slash commands.
- **Multi-physics agents** ([`src/agents/`](src/agents/AGENT.md)): lifecycle hooks, opt-in
  timeouts, and `python -m src.agents.cli scaffold <name>` to generate a new agent from the spec
  template.
- **Noyron v2.2** ([`config/scenarios/noyron_basis_cpu.yaml`](config/scenarios/noyron_basis_cpu.yaml)):
  MCTS-guided Galerkin basis selection on Leap 71 helical SDF geometries —
  `python -m src.poc.cli run --config config/scenarios/noyron_basis_cpu.yaml`.

---

## Architecture

```
Input (Discrete Board)
        │
        ▼
┌───────────────────┐
│ Continuous        │  Maps grid to Fourier features on [0,1]²
│ Embedding         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Strategy Body     │  Galerkin Attention + FNet Mixing
│ (Global Influence)│  O(N) complexity
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Tactical Head     │  Softmax Attention (preserves injectivity)
│ (Local Reading)   │  For life & death calculations
└───────────────────┘
        │
        ├──────────────┐
        ▼              ▼
┌─────────────┐ ┌─────────────┐
│ Policy Head │ │ Value Head  │
│ (Move Dist) │ │ (Eval)      │
└─────────────┘ └─────────────┘
```

See [docs/architecture/c4_mermaid.md](docs/architecture/c4_mermaid.md) for comprehensive C4 architecture diagrams in Mermaid format, or [docs/archive/C4_ARCHITECTURE.md](docs/archive/C4_ARCHITECTURE.md) for ASCII-art versions.

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CUDA 12.6 recommended for GPU backends)
- Optional: CUDA 12.x+ for GPU training/inference
- Optional: `onnxruntime`, `onnxscript` (for the ONNX export/runtime path in `src/deployment/`)

### From Source

```bash
git clone https://github.com/ianshank/AlphaGalerkin.git
cd AlphaGalerkin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Dependencies

```
torch>=2.0.0
einops>=0.7.0
jaxtyping>=0.2.25
pydantic>=2.0.0
hydra-core>=1.3.0
structlog>=23.0.0
numpy>=1.24.0
```

---

## Quick Start

### Playing a Game via GTP

```bash
# Start GTP engine (connects to Go GUIs like Sabaki)
python -m src.tools.cli gtp --board-size 19

# Or with a trained model
python -m src.tools.cli gtp --model checkpoints/model.pt --board-size 19
```

### Using the Model in Python

```python
import torch
from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel

# Create model
config = OperatorConfig(
    d_model=256,
    n_heads=8,
    n_galerkin_layers=6,
    n_softmax_layers=2,
)
model = AlphaGalerkinModel(config)

# Create random board state (batch=1, channels=17, height=19, width=19)
board = torch.randn(1, 17, 19, 19)

# Get policy and value
output = model(board)
print(f"Policy shape: {output.policy_logits.shape}")  # (1, 362) - 361 moves + pass
print(f"Value: {output.value.item()}")  # [-1, 1]
```

### Resolution Transfer

```python
# Train on 9x9
model.training_resolution = 9
# ... training loop on 9x9 data ...

# Switch to 19x19 for inference (zero-shot!)
model.eval()
model.adapt_resolution(source_size=9, target_size=19)

# Now use on 19x19 board
board_19x19 = torch.randn(1, 17, 19, 19)
output = model(board_19x19)
```

### Running MCTS

```python
from src.mcts.search import MCTS
from src.mcts.evaluator import ModelEvaluator

# Create evaluator and search
evaluator = ModelEvaluator(model, device="cuda")
mcts = MCTS(
    evaluator=evaluator,
    n_simulations=800,
    c_puct=1.5,
)

# Run search
action_probs = mcts.search(game_state)
best_move = mcts.get_action(game_state, temperature=0)
```

### Chess Self-Play Training

```bash
# Train chess model (AlphaZero methodology)
python -m scripts.train_chess training.total_steps=1000 mcts.n_simulations=100

# Enable Stockfish benchmark evaluation
python -m scripts.train_chess \
  training.engine_eval_enabled=true \
  training.engine_eval_path=/path/to/stockfish \
  training.engine_eval_depth=5
```

```python
from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel

# Chess model: 119-channel input, 4672-action policy
config = OperatorConfig(
    d_model=256,
    n_heads=8,
    n_galerkin_layers=6,
    n_softmax_layers=2,
    input_channels=119,
    game_type="chess",
    action_space_size=4672,
)
model = AlphaGalerkinModel(config)
```

---

## Use Cases

### 1. Research: Resolution-Independent Learning

**Goal**: Study how Go knowledge transfers across board sizes.

```python
# Train on small boards (faster iteration)
train_sizes = [5, 7, 9]
for size in train_sizes:
    model.training_resolution = size
    train(model, get_dataset(size))

# Evaluate on all sizes including unseen ones
test_sizes = [5, 7, 9, 13, 19]
for size in test_sizes:
    model.adapt_resolution(train_sizes[-1], size)
    evaluate(model, get_dataset(size))
```

**Research Questions**:

- Does influence understanding transfer from 9x9 to 19x19?
- What's the minimum training size for effective 19x19 play?
- How does spectral filtering affect transfer quality?

### 2. Education: Learn Go Fundamentals on Small Boards

**Goal**: Teaching tool that demonstrates concepts learned on any board size.

```bash
# Start teaching mode on 9x9
python -m src.tools.cli gtp --board-size 9 --model teacher_model.pt

# Switch to 13x13 for intermediate lessons
python -m src.tools.cli gtp --board-size 13 --model teacher_model.pt

# Full 19x19 for advanced play
python -m src.tools.cli gtp --board-size 19 --model teacher_model.pt
```

### 3. Fast Prototyping: Accelerated MCTS

**Goal**: Rapid game analysis and move generation.

```python
from src.modeling.model import AlphaGalerkinFast

# Use FNet-only model for fast rollouts
fast_model = AlphaGalerkinFast(config, n_layers=4)

# 5x faster inference for MCTS leaf evaluation
fast_evaluator = ModelEvaluator(fast_model, device="cuda")
mcts = MCTS(evaluator=fast_evaluator, n_simulations=1600)

# More simulations in same time budget
move = mcts.get_action(game_state)
```

### 4. Hybrid Systems: Combine with Traditional Engines

**Goal**: Use AlphaGalerkin for global strategy, traditional engines for tactics.

```python
def hybrid_move_selection(game_state):
    # AlphaGalerkin for global influence assessment
    global_policy = alpha_galerkin.get_policy(game_state)

    # Traditional engine for local tactical verification
    tactical_moves = traditional_engine.get_tactical_moves(game_state)

    # Combine: prefer tactically sound moves with good global influence
    combined_scores = global_policy * tactical_weights
    return combined_scores.argmax()
```

### 5. Tournament Play: GTP-Compatible Engine

**Goal**: Compete in computer Go tournaments.

```bash
# Configure for tournament
python -m src.tools.cli gtp \
    --model tournament_model.pt \
    --board-size 19 \
    --device cuda \
    --simulations 1600

# GTP commands work with standard Go GUIs
# - Sabaki
# - GoGui
# - Lizzie
# - KaTrain
```

### 6. Analysis: Position Evaluation

**Goal**: Analyze professional games with influence visualization.

```python
def analyze_position(sgf_path):
    game = load_sgf(sgf_path)

    for move_num, position in enumerate(game.positions):
        output = model(position.tensor)

        # Policy shows likely next moves
        top_moves = output.policy_logits.topk(5)

        # Value shows winning probability
        win_prob = (output.value.item() + 1) / 2

        # LBB constant shows model confidence
        _, _, lbb = model(position.tensor, return_lbb=True)

        print(f"Move {move_num}: Win={win_prob:.1%}, LBB={lbb:.4f}")
```

### 7. Curriculum Learning: Progressive Board Sizes

**Goal**: Train efficiently by starting small and scaling up.

```python
curriculum = [
    (5, 1000),   # 5x5 for 1000 games
    (7, 2000),   # 7x7 for 2000 games
    (9, 5000),   # 9x9 for 5000 games
    (13, 10000), # 13x13 for 10000 games
    (19, 50000), # 19x19 for 50000 games
]

for board_size, n_games in curriculum:
    model.training_resolution = board_size
    self_play(model, n_games, board_size)

# Model learns progressively more complex positions
```

### 8. Embedded Systems: Lightweight Inference

**Goal**: Run on edge devices with limited compute.

```python
# Use minimal configuration
config = OperatorConfig(
    d_model=64,
    n_heads=4,
    n_galerkin_layers=2,
    n_softmax_layers=1,
    use_fnet_mixing=True,  # Faster than attention
)

model = AlphaGalerkinFast(config)

# Quantize for edge deployment
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Deploy on Raspberry Pi, Jetson, etc.
```

### 9. LLM-Prior MCTS for Out-of-Distribution PDEs

**Goal**: Guide MCTS basis selection with a *generalist* LLM (Qwen-14B
served by LM Studio) so the search survives PDE families a
domain-trained evaluator has never seen.

**Why this is interesting**: the project's existing `FNetEvaluator`
gives strong policy priors *inside* the training distribution and
collapses outside it. A generalist LLM with no PDE-specific training
won't beat the trained evaluator on Poisson, but it remains useful on
Burgers / biharmonic / Helmholtz where the trained head is silent. The
ablation scenario benchmarks all three arms (random / trained / LLM) on
both ID and OOD PDEs and reports the rollout-budget reduction and the
median final residual.

```bash
# 1. Install LM Studio and start the local server
#    (https://lmstudio.ai → load qwen2.5-14b-instruct → Local Server tab)
# 2. Install the optional [lm-studio] extra
pip install -e '.[lm-studio]'

# 3. Run the ablation (GPU-only — fails loud if CUDA is unavailable)
python -m src.poc.cli run --config config/scenarios/llm_prior_demo.yaml
```

The scenario is **GPU-only by policy**: `setup()` calls
`src.poc.device.resolve_device(config.device, context=...)` which raises
`RuntimeError` when CUDA is unavailable. Arm gating is graceful — when
LM Studio preflight fails (server unreachable, model not loaded,
insufficient VRAM) the LLM arm is dropped *with its acceptance
thresholds removed* so the rest of the scenario can still pass. Same
behaviour symmetrically when the trained-arm checkpoint is missing or
when zero LLM-call latency samples are recorded.

**Headline acceptance metrics** (Pydantic-thresholded, all configurable):

| Metric | Threshold | Statistic |
|---|---|---|
| `id_rollout_reduction_pct` | ≥ 25% | Mann-Whitney U on per-seed rollouts (random vs LLM) |
| `ood_llm_residual` | ≤ 1e-2 | Median final residual on the OOD PDE |
| `ood_trained_residual` | > 1e-1 | Trained evaluator's *expected failure* threshold |
| `llm_call_p95_latency_ms` | ≤ 3000 | 95th percentile of per-call wall-clock |

CPU CI runs the mocked tests only (`tests/integrations/`,
`tests/poc/test_llm_prior_ablation_*.py`). The GPU rig captures the
headline numbers via `pytest -m gpu_required` against a live LM Studio
endpoint with `LM_STUDIO_URL` set. See
[CLAUDE.md → Regression Surface](CLAUDE.md#regression-surface) for the
exact gates.

---

## API Reference

### Core Classes

#### `AlphaGalerkinModel`

Main model combining all components.

```python
model = AlphaGalerkinModel(config: OperatorConfig)
output = model(x: Tensor, return_lbb: bool = False) -> ModelOutput
```

#### `GalerkinAttention`

O(N) attention via Petrov-Galerkin projection.

```python
attn = GalerkinAttention(d_model: int, n_heads: int, ...)
out = attn(x: Tensor, return_lbb: bool = False)
```

#### `MCTS`

Monte Carlo Tree Search with neural guidance.

```python
mcts = MCTS(evaluator, n_simulations=800, c_puct=1.5, ...)
action_dist = mcts.search(game: GameInterface)
action = mcts.get_action(game, temperature=1.0)
```

#### `GTPEngine`

Go Text Protocol interface.

```python
engine = GTPEngine(model, board_size=19)
response = engine.process_command("genmove black")
```

### Configuration

```python
from config.schemas import OperatorConfig

config = OperatorConfig(
    # Model dimensions
    d_model=256,            # Hidden dimension
    n_heads=8,              # Attention heads
    d_ffn=1024,             # FFN dimension

    # Architecture
    n_galerkin_layers=6,    # Global influence layers
    n_softmax_layers=2,     # Local tactical layers
    use_fnet_mixing=True,   # Enable FFT mixing

    # Stability
    lbb_beta_threshold=1e-6,  # LBB stability threshold

    # Input
    input_channels=17,      # Board feature planes
    n_fourier_features=64,  # Positional encoding size
)
```

---

## Testing

The project has **2,700+** passing tests across unit, integration, E2E, property-based, and security categories.

### Run All Tests

```bash
# Full test suite
pytest tests/ -v

# Chess pipeline (78 tests with coverage gate)
pytest tests/games/test_chess*.py tests/training/test_*chess*.py \
  tests/security/test_chess_security.py tests/e2e/test_chess*.py \
  --cov=src/games/chess --cov-fail-under=80 -v

# Engine integration tests
pytest tests/engines/ -v

# Math kernel tests (property-based)
pytest tests/math_kernel/ -v

# Training tests
pytest tests/training/ -v
```

### Verify Resolution Invariance

```bash
# Test 9x9 -> 19x19 transfer
python -m src.tools.verify_invariance --train-size 9 --infer-size 19

# FNet benchmark (complexity verification)
python -m src.experiments.benchmark_fnet --sizes 81,169,361 --device cpu
```

### PoC Scenario Framework

```bash
# Quick validation suite (~5 min)
python -m src.poc.cli run --config config/scenarios/poc_quick.yaml

# Full validation suite (~30 min)
python -m src.poc.cli run --config config/scenarios/poc_full.yaml

# List available scenarios
python -m src.poc.cli list
```

### Noyron HX — Zero-Shot 3D Heat-Transfer Demo (Leap 71 integration)

Train an `AlphaGalerkin` PINN-style surrogate at low collocation-point density on
an SDF-bounded helical heat exchanger that mirrors Leap 71's downloadable Noyron
HX, then evaluate zero-shot at 4× density. The demo runs entirely on the
analytical helical-tube SDF (no `.NET` / PicoGK runtime required); the optional
`[picogk]` extra is reserved for runs against a downloaded Leap 71 STL.

```bash
# CPU smoke test (analytical reference, ~30 s)
python -m src.poc.cli run --scenario noyron_hx \
    --config config/scenarios/noyron_hx.yaml \
    scenarios.0.device=cpu

# GPU headline run (analytical reference, ~2 min on GPU)
python -m src.poc.cli run --scenario noyron_hx \
    --config config/scenarios/noyron_hx.yaml

# Voxel-FDM reference run (~15-30 min on GPU)
python -m src.poc.cli run --scenario noyron_hx \
    --config config/scenarios/noyron_hx.yaml \
    scenarios.0.ref_solver_kind=voxel_fdm
```

Success criteria: `mse_low < 5e-4`, `mse_high < 1e-3`, and
`transfer_ratio = mse_high / mse_low < 4`. The scenario also records
`accept_rate` (interior-bbox sampling efficiency), `train_time_s`, and
`eval_time_s` in `ScenarioResult.metrics`.

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/ --strict
```

---

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
│   ├── modeling/          # Neural network components
│   │   ├── attention.py   # Galerkin & Softmax attention
│   │   ├── embeddings.py  # Continuous embedding
│   │   ├── fnet.py        # FFT mixing blocks
│   │   ├── stability.py   # LBB stability guard
│   │   ├── model.py       # Full model + ChessPolicyHead
│   │   └── multiscale_fourier.py  # Multi-scale Fourier features
│   ├── games/             # Game implementations
│   │   ├── chess.py       # Chess (119ch, 4672 actions)
│   │   ├── go.py          # Go (resolution-independent)
│   │   ├── wrapper.py     # StatefulGameWrapper
│   │   ├── interface.py   # GameInterface protocol
│   │   └── pettingzoo_adapter.py  # PettingZoo multi-agent adapter
│   ├── pde/               # PDE Game Framework
│   │   ├── operators.py   # Poisson, Burgers, NavierStokes, Heat, L-shaped
│   │   ├── geometry.py    # Rectangular, L-shaped, CylinderFlow domains
│   │   ├── time_stepping.py  # ForwardEuler, RK4, CrankNicolson
│   │   ├── config.py      # Pydantic PDE configuration schemas
│   │   ├── game.py        # Abstract PDEGame base class
│   │   ├── registry.py    # PDE operator registry
│   │   ├── mcts_adapter.py  # PDE-to-MCTS bridge
│   │   └── games/
│   │       ├── basis_selection.py   # Galerkin basis selection
│   │       ├── mesh_refinement.py   # Adaptive mesh refinement
│   │       └── swarm_planning.py    # Multi-agent swarm control
│   ├── research/          # SBIR benchmarking infrastructure
│   │   ├── baselines.py   # FDM, Dorfler AMR, PINN solvers
│   │   └── pde_benchmarks.py  # PDEBenchmarkRunner + reports
│   ├── training/          # Training pipeline
│   │   ├── trainer.py     # Main loop + engine eval
│   │   ├── base_trainer.py  # Shared BaseTrainer ABC
│   │   ├── losses/        # Unified loss package (LossRegistry)
│   │   │   ├── alphagalerkin.py  # Policy CE + Value MSE + LBB
│   │   │   ├── operator.py      # L2Relative, H1, MSE
│   │   │   └── physics.py       # Residual + boundary + conservation
│   │   ├── checkpoint.py         # CheckpointManager
│   │   ├── checkpoint_migration.py  # Version-aware migration
│   │   ├── loss_balancing.py     # ReLoBRaLo, GradNorm, etc.
│   │   ├── self_play.py          # Game-agnostic self-play
│   │   └── evaluation.py         # Evaluator + engine eval
│   ├── engines/           # External engine integration
│   ├── math_kernel/       # Mathematical primitives
│   ├── mcts/              # Monte Carlo Tree Search
│   └── tools/             # Utilities (GTP, CLI)
├── tests/                 # 3000+ tests, 85% coverage gate
│   ├── pde/               # PDE operators, geometry, time-stepping, swarm
│   ├── research/          # Baselines, benchmarks
│   ├── training/          # Trainer, loss properties, numerical stability
│   ├── modeling/          # Attention properties, Fourier features
│   ├── games/             # Chess, Go, PettingZoo adapter
│   ├── engines/           # UCI, match, Elo tests
│   ├── security/          # Security tests
│   └── e2e/               # End-to-end smoke tests
├── config/
│   ├── schemas.py         # Pydantic configs
│   ├── proposals/         # SBIR configs (Navy, DOE, NSF, AFWERX, DARPA D2P2)
│   └── benchmarks/        # sbir_suite.yaml
├── scripts/
│   ├── run_sbir_demo.py   # End-to-end SBIR benchmark demo (--heavy opt-in for 65 536-DOF Poisson)
│   ├── run_sbir_p40.py    # Tesla P40 high-resolution PINN/NS-FDM comparison driver
│   ├── train.py           # Training CLI with Hydra
│   └── train_chess.py     # Chess training CLI
├── docs/
│   ├── architecture/      # C4 diagrams (Mermaid)
│   └── proposals/         # SBIR templates, IP strategy, budgets, competitive analysis
└── pyproject.toml
```

---

## SBIR Positioning

AlphaGalerkin addresses a **narrow, verified novelty gap**: MCTS *multi-step look-ahead* for error-driven adaptive refinement and Galerkin basis selection is unpublished — the RL-for-AMR canon is uniformly single-step policy RL, and the only prior MCTS+finite-element work (TreeMesh, arXiv:2111.07613) targets mesh *generation*, a distinct problem (see `docs/business/proposals/PRIOR_ART_REVIEW.md`; a blanket "no MCTS+FEM" claim would be false). The SBIR reauthorization (S. 3971) extends the program through 2031 with backlogged FY2026 funds.

The stochastic Galerkin operator-splitting extension (Kolmogorov forward equations on a Gaussian-mixture basis, `src/pde/stochastic/`; see `docs/related-work.md`) is **future work for proposal purposes** — an additive layer that does not alter the MCTS/self-play core and carries no LBB claims.

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

---

## Contributing

We welcome contributions! Please see our guidelines:

1. **Code Style**: Follow Google Python Style Guide
2. **Types**: Use strict typing with jaxtyping
3. **Tests**: Add property-based tests for mathematical operators
4. **Docs**: Update CLAUDE.md with architectural decisions

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests before committing
pytest tests/ -v
ruff check src/
mypy src/ --strict
```

---

## Created by

Ian Cruickshank

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use AlphaGalerkin in your research, please cite:

```bibtex
@software{alphagalerkin2026,
  title = {AlphaGalerkin: Resolution-Independent AI for Games and PDE Solving via MCTS-Guided Galerkin Methods},
  author = {Cruickshank, Ian},
  year = {2026},
  url = {https://github.com/ianshank/AlphaGalerkin}
}
```

---

## Acknowledgments

- AlphaGo/AlphaZero teams at DeepMind for foundational work
- Galerkin Transformer paper authors for the mathematical framework
- FNet paper authors for FFT mixing insights
- The Go AI research community
