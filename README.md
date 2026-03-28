# AlphaGalerkin

**Resolution-Independent Game AI using Continuous Operator Learning**

AlphaGalerkin is a novel approach to Go AI that replaces traditional Convolutional Neural Networks (CNNs) with Continuous Operator Learning, enabling zero-shot transfer between different board sizes and accelerating Monte Carlo Tree Search through FFT-based mixing.

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

Traditional Go AI systems like AlphaGo/AlphaZero use discrete CNNs that are tied to a specific board size. A model trained on 19x19 cannot play on 9x9 without retraining. This creates several limitations:

1. **No Transfer Learning**: Skills learned on smaller boards don't transfer to larger ones
2. **Redundant Training**: Must train separate models for each board size
3. **Fixed Resolution**: Cannot adapt to non-standard board sizes (13x13, 25x25, etc.)

### Our Solution

AlphaGalerkin treats the Go board as a **continuous domain** Ω = [0,1]² rather than a discrete grid. By using:

- **Galerkin Transformers**: Approximate integral operators with O(N) complexity
- **Fourier Positional Encoding**: Resolution-independent spatial representation
- **Spectral Adaptation**: Zero-shot transfer via proper frequency filtering

We achieve a model that:

- Trains on 9x9, plays on 19x19 without modification
- Learns "physics" of Go (influence, territory) rather than pixel patterns
- Accelerates MCTS with FFT-based mixing (5x+ speedup)

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

See [docs/architecture/c4_mermaid.md](docs/architecture/c4_mermaid.md) for comprehensive C4 architecture diagrams in Mermaid format, or [docs/architecture/C4_ARCHITECTURE.md](docs/architecture/C4_ARCHITECTURE.md) for ASCII-art versions.

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### From Source

```bash
git clone https://github.com/yourusername/AlphaGalerkin.git
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

The project has **5 300+** tests across unit, integration, E2E, and security categories, with **85 % branch coverage**.

### Run All Tests

```bash
# Full test suite (excludes slow/GPU/E2E)
pytest tests/ -q -m "not slow and not e2e and not gpu_required"
# → 5322 passed, 154 skipped, 0 failed

# With branch coverage report
pytest tests/ -q -m "not slow and not e2e and not gpu_required" \
  --cov=src --cov-branch --cov-report=term
# → TOTAL  85%

# Chess pipeline (78 tests with coverage gate)
pytest tests/games/test_chess*.py tests/training/test_*chess*.py \
  tests/security/test_chess_security.py tests/e2e/test_chess*.py \
  --cov=src/games/chess --cov-fail-under=80 -v

# Math kernel tests (property-based)
pytest tests/math_kernel/ -v

# Training tests (includes 46 new branch-coverage tests)
pytest tests/training/ -v

# PDE framework tests
pytest tests/pde/ -v

# Distributed training tests
pytest tests/distributed/ -v
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

### Code Quality

```bash
# Linting (all checks pass)
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
│   │   └── model.py       # Full model + ChessPolicyHead
│   ├── games/             # Game implementations
│   │   ├── chess.py       # Chess (119ch, 4672 actions)
│   │   ├── go.py          # Go (resolution-independent)
│   │   ├── wrapper.py     # StatefulGameWrapper
│   │   └── interface.py   # GameInterface protocol
│   ├── engines/           # External engine integration
│   │   ├── uci.py         # UCI protocol driver
│   │   ├── match.py       # EngineMatch orchestration
│   │   ├── elo.py         # EloCalculator + confidence
│   │   └── config.py      # UCIConfig, MatchConfig
│   ├── training/          # Training pipeline
│   │   ├── trainer.py     # Main loop + engine eval
│   │   ├── self_play.py   # Game-agnostic self-play
│   │   ├── evaluation.py  # Evaluator + engine eval
│   │   └── checkpoint.py  # CheckpointManager
│   ├── math_kernel/       # Mathematical primitives
│   ├── mcts/              # Monte Carlo Tree Search
│   └── tools/             # Utilities (GTP, CLI)
├── tests/
│   ├── games/             # Chess, Go, wrapper tests
│   ├── engines/           # UCI, match, Elo tests
│   ├── training/          # Trainer, self-play tests
│   ├── security/          # Security tests
│   └── e2e/               # End-to-end smoke tests
├── config/
│   ├── schemas.py         # Pydantic configs
│   └── train_chess.yaml   # Chess training config
├── scripts/
│   └── train_chess.py     # Chess training CLI
├── docs/
│   └── architecture/      # C4 diagrams
└── pyproject.toml
```

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
@software{alphagalerkin2024,
  title = {AlphaGalerkin: Resolution-Independent Go AI using Continuous Operator Learning},
  year = {2024},
  url = {https://github.com/yourusername/AlphaGalerkin}
}
```

---

## Acknowledgments

- AlphaGo/AlphaZero teams at DeepMind for foundational work
- Galerkin Transformer paper authors for the mathematical framework
- FNet paper authors for FFT mixing insights
- The Go AI research community
