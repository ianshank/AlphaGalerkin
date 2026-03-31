# PRD: Chess Self-Play Training Pipeline

## Epic: Game-Agnostic Model + Chess Self-Play Training

### User Story

**As a** researcher using AlphaGalerkin,
**I want to** train the model on chess via pure self-play (AlphaZero methodology),
**So that** the model learns chess strategy entirely from self-play, with Stockfish used only as a benchmark.

### Background

The current AlphaGalerkin pipeline is hardcoded for Go (`SimpleGoGame`). Chess requires:

- 119 input channels (vs Go's 17)
- 4672 action space (vs Go's `n²+1`)
- Different Dirichlet alpha (0.3 vs 0.03)
- A policy head that outputs dense action logits (not position-based)

### Acceptance Criteria

#### AC1: Model Architecture Support

- **Given** a chess training configuration with `game_type=chess`, `input_channels=119`, `action_space_size=4672`
- **When** the model is instantiated
- **Then** it creates a `ChessPolicyHead` outputting 4672 logits and accepts 119-channel input

#### AC2: Self-Play Game Generation

- **Given** a chess `GameInterface` and trained model
- **When** `SelfPlayWorker.play_game()` is called
- **Then** it generates a complete chess game via MCTS self-play using `StatefulGameWrapper`
- **And** produces valid training experiences with correct policy shapes

#### AC3: Training Loop

- **Given** `config/train_chess.yaml`
- **When** `python -m scripts.train_chess` is run
- **Then** the full training pipeline executes: self-play → buffer → train → eval → checkpoint

#### AC4: Stockfish Benchmark Evaluation

- **Given** a training run in progress and Stockfish installed
- **When** evaluation interval is reached
- **Then** the model is evaluated against Stockfish at configurable depth
- **And** Elo estimate is logged

#### AC5: Resign Detection

- **Given** a self-play game in progress
- **When** the value head estimates < -0.9 for both sides for 10+ consecutive moves
- **Then** the game is terminated early as a loss for the resigning side

### Out of Scope

- Multi-GPU / TPU distributed training (use existing `DistributedContext` if available)
- Matching DeepMind's absolute Elo strength (we're proving the pipeline, not the compute)
- GUI visualization during training
- Opening book or endgame tablebase integration

### Success Metrics

| Metric | Target |
|---|---|
| Chess self-play completes without error | ✅ |  
| Model produces legal chess moves via MCTS | ✅ |
| Training loss converges over 1000+ steps | ✅ |
| Win rate vs random ≥ 90% after training | ✅ |
| Stockfish benchmark eval runs at interval | ✅ |
| Test coverage ≥ 80% on new modules | ✅ |
