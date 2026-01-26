# AlphaGalerkin Training Infrastructure Implementation Template

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: A complete training infrastructure for AlphaGalerkin that enables
self-play reinforcement learning with resolution-independent neural operators,
supporting zero-shot transfer between board sizes (9x9 → 19x19).
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All unit tests pass: `pytest tests/training/ -v`
- [ ] All integration tests pass: `pytest tests/integration/ -v`
- [ ] Type checking passes: `mypy src/training/ --strict`
- [ ] Linting passes: `ruff check src/training/`
- [ ] Training runs for 100 steps without error on 9x9 board
- [ ] Self-play generates valid games (legal moves only)
- [ ] Checkpoint save/load preserves model weights exactly
- [ ] Training resumes from checkpoint without loss of state
- [ ] Loss decreases over 1000 training steps (learning signal exists)
- [ ] Model evaluates on different board sizes after training (resolution independence preserved)
```

### 1.3 Problem Description (The "Three Paragraphs")

```
PARAGRAPH 1 - Core Problem:
The AlphaGalerkin system has complete inference infrastructure (neural operators,
MCTS, GTP engine) but lacks training capability. We need to implement a self-play
reinforcement learning loop that: (1) generates training data via MCTS self-play,
(2) stores experiences in a replay buffer, (3) trains the model on batched samples,
and (4) iteratively improves through policy iteration. The key coordination logic
involves synchronizing self-play game generation with training updates while
maintaining stable learning dynamics.

PARAGRAPH 2 - Data Flows & State:
Data flows: Self-play → Game Records → Replay Buffer → Batched Samples → Training Loop
→ Updated Weights → Back to Self-play. State to maintain: (a) Replay buffer with
prioritized sampling of (board_state, mcts_policy, game_outcome) tuples, (b) Training
step counter for LR scheduling, (c) Checkpoint state (model weights, optimizer state,
scheduler state, replay buffer snapshot), (d) LBB stability metrics throughout training.
The system must handle variable board sizes in the same replay buffer and training batch.

PARAGRAPH 3 - Failure Modes & Invariants:
Failure modes: (1) LBB condition violation causing training instability - mitigated by
StabilityGuard monitoring and regularization loss, (2) Policy collapse to uniform/
deterministic - mitigated by temperature scheduling and Dirichlet noise, (3) Value
head explosion - mitigated by gradient clipping and bounded targets [-1, 1],
(4) Replay buffer staleness - mitigated by continuous self-play and buffer refresh.
Invariants: LBB constant σ_min > β > 0, policy outputs sum to 1.0, value in [-1, 1],
all generated moves are legal, checkpoint restoration is byte-identical.
```

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.10+
- Required Dependencies (from existing pyproject.toml):
  - torch>=2.0.0
  - einops>=0.7.0
  - jaxtyping>=0.2.25
  - pydantic>=2.0.0
  - hydra-core>=1.3.0
  - structlog>=23.0.0
  - numpy>=1.24.0
- Security: No hardcoded paths, seeds, or credentials
- Compatibility:
  - Must use existing TrainingConfig from config/schemas.py
  - Must integrate with existing model classes in src/modeling/
  - Must use existing MCTS from src/mcts/
  - Must preserve resolution independence (no hard-coded board sizes)
- Testing: pytest + hypothesis for property-based tests
```

### 2.2 Soft Constraints (Preferences)

```
- Style: Follow existing codebase patterns (see src/modeling/ for reference)
- Type Hints: Use jaxtyping for tensor shapes (Float[Tensor, "batch seq dim"])
- Logging: Use structlog (already configured in project)
- Architecture:
  - Prefer composition over inheritance
  - Use Protocol for interfaces (see src/mcts/evaluator.py)
  - Use dataclasses/NamedTuple for data structures
  - Use einops for tensor operations
- Performance:
  - Support mixed precision training (torch.cuda.amp)
  - Batch operations where possible
  - Minimize CPU-GPU transfers
- Testing:
  - >90% coverage on core training logic
  - Property-based tests for mathematical invariants
  - Integration tests for full training loop
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Add new dependencies (tensorboard, wandb, tqdm) with justification
- Create new configuration fields in schemas.py if needed
- Restructure src/training/ directory organization
- Add utility functions to existing modules
- Choose specific sampling strategies for replay buffer
- Implement parallel self-play using multiprocessing or threading
- Add debug/visualization utilities
- Create additional test fixtures and helpers
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE (create/modify freely):
- src/training/**/*.py (new directory)
- src/data/**/*.py (new directory)
- tests/training/**/*.py (new directory)
- scripts/train.py (new file)
- config/schemas.py (extend TrainingConfig if needed)
- config/*.yaml (Hydra configs)
- CLAUDE.md (update with new commands)

IN SCOPE (modify carefully, preserve interfaces):
- src/modeling/model.py (add training-specific methods if needed)
- src/mcts/evaluator.py (extend for training integration)
- src/tools/cli.py (add training subcommand)
- pyproject.toml (add dependencies)

OUT OF SCOPE (read-only):
- src/math_kernel/*.py (core math is stable)
- src/modeling/attention.py (core architecture is stable)
- src/modeling/fnet.py (core architecture is stable)
- tests/math_kernel/*.py (existing tests)
- docs/*.md (except for training docs)
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation within src/training/, src/data/, tests/training/
- Adding type hints and docstrings
- Running tests and fixing failures
- Installing dependencies via pyproject.toml
- Refactoring for consistency with existing code style
- Creating test fixtures and mocks
- Adding logging statements

CONFIRM FIRST (ask before proceeding):
- Modifying existing model forward() signatures
- Changing existing configuration defaults
- Adding required (non-optional) config fields
- Modifying existing test assertions
- Any changes to src/math_kernel/

PROHIBITED (do not attempt):
- Removing existing functionality
- Breaking changes to GTP interface
- Hard-coding board sizes anywhere
- Removing type hints from existing code
- Disabling existing tests
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5 per component
- Max files to modify in single pass: 15
- Implementation order: Follow priority in component list
- Testing cadence: Run tests after each component completion
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:

# 1. Lint check
ruff check src/training/ src/data/ tests/training/

# 2. Type check
mypy src/training/ src/data/ --strict

# 3. Unit tests for new code
pytest tests/training/ -v --tb=short

# 4. Integration tests
pytest tests/integration/ -v --tb=short

# 5. Full test suite (ensure no regressions)
pytest tests/ -v

# 6. Smoke test training
python -m scripts.train --config-name=test --max_steps=10

# 7. Verify resolution independence preserved
python -m src.tools.verify_invariance --train-size 9 --infer-size 19
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
→ Run `ruff check --fix` for auto-fixable issues
→ Manually fix remaining issues
→ Re-run verification

ON TYPE ERROR:
→ Analyze error message
→ Add proper type annotations (prefer jaxtyping for tensors)
→ Avoid `type: ignore` unless absolutely necessary with comment
→ Re-run verification

ON TEST FAILURE:
→ Read failure output completely
→ Identify if implementation bug or test bug
→ Fix implementation first (tests are specification)
→ Only fix test if test itself is incorrect
→ Re-run verification

ON IMPORT ERROR:
→ Check pyproject.toml for missing dependency
→ Add dependency with version constraint
→ Re-run verification

ON REPEATED FAILURE (same error 3x):
→ Stop and document the issue
→ List attempted fixes
→ Provide analysis of root cause hypothesis
→ Request human guidance
```

### 4.3 Success Verification

```
Before reporting component completion:
1. All verification commands pass
2. New code has >90% test coverage
3. Docstrings present on all public functions
4. Type hints on all function signatures
5. No TODO comments left unaddressed
6. CLAUDE.md updated with new commands
7. Brief summary of implementation decisions provided
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory (CLAUDE.md Updates)

```markdown
# Additions to CLAUDE.md after implementation:

## Training Commands
- `python -m scripts.train`: Run training with default config
- `python -m scripts.train --config-name=fast`: Quick training for testing
- `python -m scripts.train trainer.checkpoint_path=<path>`: Resume from checkpoint

## Training Architecture Decisions
- [DATE]: Replay buffer uses reservoir sampling for uniform distribution
- [DATE]: Self-play uses temperature annealing (1.0 → 0.1 over game)
- [DATE]: Training loss = policy_CE + value_MSE + lbb_regularization
- [DATE]: Checkpoint includes: model, optimizer, scheduler, step, buffer_stats

## Training Known Issues
- [Issue]: [Description and workaround if any]
```

### 5.2 Information to Preserve Across Sessions

```
- Working training command with tested hyperparameters
- Checkpoint format version and compatibility notes
- Performance baselines (steps/sec, games/hour)
- Known numerical stability thresholds
- Successful training run configurations
```

### 5.3 Information That Can Be Re-derived

```
- Current training step (from checkpoint)
- Model architecture (from config)
- Test results (can re-run)
- Code structure (can scan)
```

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```bash
# 1. Verify environment
python --version  # Ensure 3.10+
pip list | grep torch  # Ensure torch installed

# 2. Run existing tests to establish baseline
pytest tests/ -v --tb=short

# 3. Verify existing code works
python -m src.tools.verify_invariance --train-size 9 --infer-size 19

# 4. Understand existing patterns
# Read these files for style reference:
# - src/modeling/model.py (class structure)
# - src/mcts/search.py (algorithm implementation)
# - config/schemas.py (configuration patterns)
# - tests/integration/test_model.py (testing patterns)
```

### 6.2 Implementation Order

```
PHASE 1: Core Training Components (HIGH Priority)
1. src/training/__init__.py - Package setup
2. src/training/loss.py - AlphaGalerkinLoss class
3. src/training/replay_buffer.py - ReplayBuffer with priority sampling
4. tests/training/test_loss.py - Loss function tests
5. tests/training/test_replay_buffer.py - Buffer tests

PHASE 2: Self-Play & Data (HIGH Priority)
6. src/training/self_play.py - SelfPlayWorker
7. src/data/__init__.py - Package setup
8. src/data/dataset.py - ReplayDataset
9. src/data/collate.py - Variable-size board collation
10. tests/training/test_self_play.py - Self-play tests
11. tests/training/test_dataset.py - Dataset tests

PHASE 3: Training Loop (HIGH Priority)
12. src/training/trainer.py - Main Trainer class
13. src/training/checkpoint.py - CheckpointManager
14. tests/training/test_trainer.py - Trainer tests
15. tests/training/test_checkpoint.py - Checkpoint tests

PHASE 4: CLI & Integration (MEDIUM Priority)
16. scripts/train.py - CLI entry point
17. config/train.yaml - Default training config
18. config/train_fast.yaml - Fast config for testing
19. tests/integration/test_training_loop.py - E2E tests

PHASE 5: Evaluation (LOW Priority)
20. src/training/evaluation.py - EvaluationPipeline
21. tests/training/test_evaluation.py - Evaluation tests
```

### 6.3 Completion Checklist

```
□ All success criteria from Section 1.2 met
□ All verification commands from Section 4.1 pass
□ CLAUDE.md updated with training commands and decisions
□ Each component has corresponding test file
□ Property-based tests for mathematical invariants
□ Integration test for full training loop
□ Smoke test runs 100 steps without error
□ Summary of implementation decisions documented
□ Known limitations documented
```

---

## SECTION 7: COMPONENT SPECIFICATIONS

### 7.1 Training Loss (`src/training/loss.py`)

```python
"""
AlphaGalerkinLoss: Composite loss for policy + value + LBB regularization

Interface:
    loss = AlphaGalerkinLoss(config: TrainingConfig)
    total_loss, loss_dict = loss(
        policy_logits: Float[Tensor, "batch actions"],
        value: Float[Tensor, "batch 1"],
        target_policy: Float[Tensor, "batch actions"],
        target_value: Float[Tensor, "batch 1"],
        lbb_constant: Float[Tensor, ""] | None
    )

Loss components:
    - Policy: CrossEntropy(policy_logits, target_policy)
    - Value: MSE(value, target_value)
    - LBB: -log(lbb_constant + eps) if lbb_constant provided

Returns:
    - total_loss: Weighted sum for backward()
    - loss_dict: {"policy": x, "value": y, "lbb": z, "total": t}

Properties to test:
    - Loss is non-negative
    - Gradients flow to all parameters
    - Loss decreases with correct predictions
    - LBB term increases when constant decreases
"""
```

### 7.2 Replay Buffer (`src/training/replay_buffer.py`)

```python
"""
ReplayBuffer: Circular buffer with optional priority sampling

Interface:
    buffer = ReplayBuffer(capacity: int, priority_alpha: float = 0.6)
    buffer.add(experience: Experience)
    batch = buffer.sample(batch_size: int) -> list[Experience]
    buffer.update_priorities(indices: list[int], priorities: list[float])

Experience dataclass:
    - board_state: Float[Tensor, "channels height width"]
    - board_size: int  # For resolution-aware batching
    - target_policy: Float[Tensor, "actions"]  # MCTS visit distribution
    - target_value: float  # Game outcome from this player's perspective
    - metadata: dict  # Optional: move_number, game_id, etc.

Properties to test:
    - Buffer respects capacity (old items evicted)
    - Sampling returns valid experiences
    - Priority sampling biases toward high-priority items
    - Buffer handles variable board sizes
    - Thread-safe for concurrent add/sample
"""
```

### 7.3 Self-Play Generator (`src/training/self_play.py`)

```python
"""
SelfPlayWorker: Generate training games via MCTS

Interface:
    worker = SelfPlayWorker(
        model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
        board_sizes: list[int] = [9, 13, 19]
    )
    games = worker.generate_games(n_games: int) -> list[GameRecord]

GameRecord dataclass:
    - board_size: int
    - moves: list[tuple[int, int] | None]  # None = pass
    - policies: list[Float[Tensor, "actions"]]  # MCTS distributions
    - outcome: float  # 1.0 = black wins, -1.0 = white wins, 0.0 = draw

Conversion to experiences:
    experiences = GameRecord.to_experiences() -> list[Experience]
    # Handles perspective flipping for value targets

Properties to test:
    - All generated moves are legal
    - Game terminates (two passes or board full)
    - Policies sum to 1.0
    - Outcome is in {-1.0, 0.0, 1.0}
    - Works for all specified board sizes
"""
```

### 7.4 Dataset & DataLoader (`src/data/dataset.py`, `src/data/collate.py`)

```python
"""
ReplayDataset: PyTorch Dataset wrapping ReplayBuffer

Interface:
    dataset = ReplayDataset(buffer: ReplayBuffer)
    item = dataset[idx]  # Returns Experience

VariableSizeCollator: Handles batching different board sizes

Interface:
    collator = VariableSizeCollator(pad_value: float = 0.0)
    batch = collator(experiences: list[Experience]) -> TrainingBatch

TrainingBatch dataclass:
    - board_states: Float[Tensor, "batch channels max_h max_w"]
    - board_sizes: Int[Tensor, "batch"]  # Original sizes for masking
    - target_policies: Float[Tensor, "batch max_actions"]
    - target_values: Float[Tensor, "batch 1"]
    - padding_mask: Bool[Tensor, "batch max_h max_w"]

Properties to test:
    - Collator handles mixed board sizes in batch
    - Padding mask correctly identifies padded regions
    - Original data recoverable from padded batch
"""
```

### 7.5 Trainer (`src/training/trainer.py`)

```python
"""
Trainer: Main training loop orchestrator

Interface:
    trainer = Trainer(
        model: AlphaGalerkinModel,
        config: AlphaGalerkinConfig,
        device: torch.device
    )
    trainer.train(n_steps: int)
    trainer.save_checkpoint(path: Path)
    trainer.load_checkpoint(path: Path)

Training loop (per step):
    1. If buffer needs data: run self-play games
    2. Sample batch from replay buffer
    3. Forward pass with mixed precision
    4. Compute loss (policy + value + LBB)
    5. Backward pass with gradient clipping
    6. Optimizer step with LR scheduling
    7. Log metrics (loss, LBB, learning rate)
    8. Checkpoint if interval reached

Properties to test:
    - Training step increments correctly
    - Loss is logged at each step
    - Checkpoint saves all required state
    - Training resumes correctly from checkpoint
    - LR schedule follows config
    - Gradient clipping is applied
"""
```

### 7.6 Checkpoint Manager (`src/training/checkpoint.py`)

```python
"""
CheckpointManager: Save/load training state

Interface:
    manager = CheckpointManager(checkpoint_dir: Path)
    manager.save(
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        buffer_stats: dict,
        config: AlphaGalerkinConfig
    )
    state = manager.load(path: Path) -> CheckpointState
    latest = manager.get_latest() -> Path | None

CheckpointState dataclass:
    - step: int
    - model_state_dict: dict
    - optimizer_state_dict: dict
    - scheduler_state_dict: dict
    - buffer_stats: dict
    - config: AlphaGalerkinConfig
    - timestamp: datetime
    - version: str  # For compatibility

Properties to test:
    - Save creates valid checkpoint file
    - Load restores exact state
    - get_latest finds most recent checkpoint
    - Version mismatch raises clear error
    - Corrupt file handled gracefully
"""
```

---

## SECTION 8: TESTING SPECIFICATIONS

### 8.1 Unit Tests Required

```python
# tests/training/test_loss.py
- test_policy_loss_correct_shape
- test_value_loss_correct_shape
- test_lbb_regularization_effect
- test_loss_weights_applied
- test_gradients_flow
- test_loss_decreases_with_correct_predictions

# tests/training/test_replay_buffer.py
- test_buffer_add_and_sample
- test_buffer_capacity_limit
- test_priority_sampling_bias
- test_buffer_thread_safety
- test_variable_board_sizes

# tests/training/test_self_play.py
- test_game_generation_valid_moves
- test_game_termination
- test_policy_normalization
- test_outcome_values
- test_multiple_board_sizes

# tests/training/test_trainer.py
- test_training_step_increments
- test_loss_logged
- test_checkpoint_interval
- test_resume_from_checkpoint
- test_lr_schedule_applied

# tests/training/test_checkpoint.py
- test_save_load_roundtrip
- test_get_latest_checkpoint
- test_version_compatibility
- test_corrupt_file_handling
```

### 8.2 Property-Based Tests (Hypothesis)

```python
# tests/training/test_properties.py

@given(board_size=st.integers(5, 25))
def test_loss_non_negative(board_size):
    """Loss should always be non-negative."""

@given(batch_size=st.integers(1, 64), capacity=st.integers(100, 10000))
def test_buffer_never_exceeds_capacity(batch_size, capacity):
    """Buffer size should never exceed capacity."""

@given(n_games=st.integers(1, 10))
def test_self_play_all_moves_legal(n_games):
    """All moves in generated games should be legal."""

@given(board_sizes=st.lists(st.integers(9, 19), min_size=1, max_size=8))
def test_collator_handles_mixed_sizes(board_sizes):
    """Collator should handle any mix of valid board sizes."""
```

### 8.3 Integration Tests

```python
# tests/integration/test_training_loop.py

def test_full_training_loop_100_steps():
    """Run 100 training steps and verify loss decreases."""

def test_training_with_checkpoint_resume():
    """Train, save, load, continue - verify continuity."""

def test_self_play_to_training_pipeline():
    """Generate games, add to buffer, train on them."""

def test_resolution_independence_after_training():
    """Train on 9x9, verify inference works on 19x19."""
```

---

## SECTION 9: CONFIGURATION FILES

### 9.1 Default Training Config (`config/train.yaml`)

```yaml
# Default training configuration
defaults:
  - _self_

# Model configuration
operator:
  d_model: 256
  d_key: 64
  d_value: 64
  n_heads: 8
  n_galerkin_layers: 6
  n_softmax_layers: 2
  n_fourier_features: 128

# MCTS configuration for self-play
mcts:
  n_simulations: 400
  c_puct: 1.5
  dirichlet_alpha: 0.03
  dirichlet_epsilon: 0.25
  temperature: 1.0
  temperature_drop_move: 30

# Training configuration
training:
  learning_rate: 2e-4
  weight_decay: 1e-4
  batch_size: 256
  gradient_clip: 1.0
  lr_scheduler: cosine
  warmup_steps: 1000
  total_steps: 100000
  n_self_play_games: 100
  replay_buffer_size: 500000
  policy_loss_weight: 1.0
  value_loss_weight: 1.0
  checkpoint_interval: 1000
  use_amp: true

# Experiment settings
experiment_name: alphagalerkin_train
seed: 42
log_level: INFO
log_lbb_metrics: true

# Training-specific additions
trainer:
  board_sizes: [9, 13, 19]
  games_per_iteration: 100
  min_buffer_size: 10000
  eval_interval: 5000
  eval_games: 50
```

### 9.2 Fast Test Config (`config/train_fast.yaml`)

```yaml
# Fast configuration for testing
defaults:
  - train
  - _self_

operator:
  d_model: 64
  n_galerkin_layers: 2
  n_softmax_layers: 1
  n_fourier_features: 32

mcts:
  n_simulations: 50

training:
  batch_size: 32
  total_steps: 100
  n_self_play_games: 10
  replay_buffer_size: 1000
  checkpoint_interval: 50
  warmup_steps: 10

trainer:
  board_sizes: [9]
  games_per_iteration: 5
  min_buffer_size: 100
  eval_interval: 50
  eval_games: 5

experiment_name: alphagalerkin_test
```

---

## SECTION 10: LOGGING & DEBUGGING

### 10.1 Structured Logging Setup

```python
"""
Logging configuration using structlog (already in project)

Log levels:
- DEBUG: Tensor shapes, intermediate values, timing
- INFO: Training step, loss values, checkpoints
- WARNING: LBB instability, buffer low, slow training
- ERROR: Training failures, invalid states

Structured fields:
- step: Current training step
- loss: Loss value
- loss_policy: Policy loss component
- loss_value: Value loss component
- loss_lbb: LBB regularization component
- lbb_constant: Current LBB stability constant
- lr: Current learning rate
- buffer_size: Current replay buffer size
- games_generated: Number of self-play games generated
"""
```

### 10.2 Debug Utilities

```python
# src/training/debug.py

def log_tensor_stats(name: str, tensor: Tensor) -> None:
    """Log min, max, mean, std of tensor."""

def check_gradient_health(model: nn.Module) -> dict:
    """Check for vanishing/exploding gradients."""

def visualize_policy(policy: Tensor, board_size: int) -> str:
    """ASCII visualization of policy distribution."""

def profile_training_step(trainer: Trainer) -> dict:
    """Profile time spent in each training phase."""
```

### 10.3 Metrics to Track

```
Per-step metrics:
- total_loss, policy_loss, value_loss, lbb_loss
- learning_rate
- gradient_norm (before clipping)
- lbb_constant (stability metric)
- step_time_ms

Per-iteration metrics:
- games_generated
- buffer_size
- mean_game_length
- outcome_distribution (black_wins, white_wins, draws)

Per-evaluation metrics:
- win_rate_vs_random
- win_rate_vs_previous
- mean_move_agreement (with MCTS policy)
```

---

## SECTION 11: NEXT STEPS

### Immediate Actions (After Template Approval)

```
1. Create directory structure:
   mkdir -p src/training src/data tests/training scripts config

2. Initialize packages:
   touch src/training/__init__.py src/data/__init__.py tests/training/__init__.py

3. Begin Phase 1 implementation:
   - Implement loss.py
   - Implement replay_buffer.py
   - Write corresponding tests
   - Run verification loop

4. Continue through phases in order
```

### Success Metrics for Phase 1

```
□ Loss module computes correct gradients
□ Replay buffer handles 10K+ experiences
□ All tests pass
□ Type checking passes
□ Ready for Phase 2
```

---

## APPENDIX A: EXISTING CODE INTERFACES

### Model Interface (from src/modeling/model.py)

```python
class AlphaGalerkinModel(nn.Module):
    def forward(
        self,
        board_state: Float[Tensor, "batch channels height width"],
        return_lbb: bool = False
    ) -> ModelOutput:
        # Returns: ModelOutput(policy_logits, value, lbb_constant)
```

### MCTS Interface (from src/mcts/search.py)

```python
class MCTS:
    def __init__(self, evaluator: Evaluator, config: MCTSConfig): ...
    def search(self, game_state: GameState, n_simulations: int) -> MCTSNode: ...
```

### Evaluator Interface (from src/mcts/evaluator.py)

```python
class FNetEvaluator:
    def __init__(self, model: AlphaGalerkinModel, device: torch.device): ...
    def evaluate(self, game_state: GameState) -> EvaluationResult: ...
    def evaluate_batch(self, states: list[GameState]) -> list[EvaluationResult]: ...
```

---

## APPENDIX B: FILE TEMPLATES

### Package Init Template

```python
"""AlphaGalerkin Training Module.

This module provides training infrastructure for the AlphaGalerkin
resolution-independent Go AI.
"""

from .loss import AlphaGalerkinLoss
from .replay_buffer import ReplayBuffer, Experience
from .self_play import SelfPlayWorker, GameRecord
from .trainer import Trainer
from .checkpoint import CheckpointManager

__all__ = [
    "AlphaGalerkinLoss",
    "ReplayBuffer",
    "Experience",
    "SelfPlayWorker",
    "GameRecord",
    "Trainer",
    "CheckpointManager",
]
```

### Test File Template

```python
"""Tests for [module name].

Property-based tests use Hypothesis.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st
import torch

from src.training.[module] import [Class]


class TestClassName:
    """Tests for ClassName."""

    def test_basic_functionality(self) -> None:
        """Test basic usage."""
        pass

    @given(st.integers(1, 100))
    def test_property(self, value: int) -> None:
        """Property: description."""
        pass
```

---

*End of Implementation Template*
