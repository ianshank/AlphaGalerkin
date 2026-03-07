# AlphaGalerkin Chess Self-Play Training Pipeline — Sprint Plan

## Goal

Retrain AlphaGalerkin on chess via **pure self-play** (AlphaZero/MuZero methodology), matching the DeepMind specification: no pre-training against Stockfish, no rule injection — the agent learns chess entirely from self-play. Stockfish is used only as a **benchmark evaluation target** to measure Elo progress.

## Current State

| Component | Status | Gap |
|---|---|---|
| `ChessGame` (`src/games/chess.py`) | ✅ Complete | 119-channel tensor, 4672 action space, full rule engine |
| `StatefulGameWrapper` (`src/games/wrapper.py`) | ✅ New | Bridges stateless `GameInterface` → MCTS protocol |
| `SelfPlayWorker` (`src/training/self_play.py`) | ❌ Go-only | Hardcodes `SimpleGoGame`, Go board-size sampling |
| `Trainer` (`src/training/trainer.py`) | ❌ Go-only | Self-play worker bound to Go, no game injection |
| `PolicyHead` (`src/modeling/model.py`) | ❌ Go-only | Outputs `n+1` (board positions + pass), chess needs 4672 |
| Model config (`config/schemas.py`) | ❌ Go-only | `input_channels=17`, chess needs 119 |
| Training config (YAML) | ❌ Missing | No chess-specific training YAML |
| Stockfish benchmark eval | ⚠️ Partial | `EngineMatch` exists but `match.py` had MCTS arg bug (now fixed) |
| `ReplayBuffer` | ✅ Game-agnostic | Works for any experience shape |
| MCTS (`src/mcts/search.py`) | ✅ Game-agnostic | Works via `GameInterface` protocol |

## Architecture Changes Required

### Epic 1: Game-Agnostic Model Architecture (XL)

**Problem**: `PolicyHead` outputs `n+1` logits (one per board position + pass), which only works for Go. Chess has 4672 action types that don't map to board positions.

**Solution**: Create a `ChessPolicyHead` that outputs 4672 logits via a dense projection from global features, and make the model configurable per game.

| File | Change |
|---|---|
| `src/modeling/model.py` | Add `action_space_size` to `AlphaGalerkinModel.__init__()`, create `ChessPolicyHead` |
| `config/schemas.py` | Add `action_space_size`, `game_type` fields to `OperatorConfig` |

### Epic 2: Game-Agnostic Self-Play (L)

**Problem**: `SelfPlayWorker.play_game()` hardcodes `SimpleGoGame`, Go-specific action handling, and board-size sampling.

**Solution**: Accept a `GameInterface` and use `StatefulGameWrapper` internally.

| File | Change |
|---|---|
| `src/training/self_play.py` | Accept `game: GameInterface`, use `StatefulGameWrapper`, generalize action handling |
| `src/training/trainer.py` | Wire `game` into `SelfPlayWorker`, evaluation, and checkpoint |

### Epic 3: Chess Training Configuration (M)

| File | Change |
|---|---|
| `config/train_chess.yaml` | New chess-specific training config |
| `config/schemas.py` | Add chess-specific Dirichlet alpha (0.3 for chess vs 0.03 for Go) |

### Epic 4: Stockfish Benchmark Evaluation Integration (M)

| File | Change |
|---|---|
| `src/training/trainer.py` | Add periodic Stockfish eval during training loop |
| `src/engines/match.py` | Already fixed with `StatefulGameWrapper` |
| `scripts/train_chess.py` | New training CLI with chess + Stockfish eval |

### Epic 5: Comprehensive Test Suite (L)

| File | Change |
|---|---|
| `tests/training/test_chess_self_play.py` | New: chess-specific self-play tests |
| `tests/modeling/test_chess_model.py` | New: chess model forward pass tests |
| `tests/engines/test_match_chess.py` | New: chess engine match integration tests |
| `tests/games/test_wrapper.py` | New: `StatefulGameWrapper` tests |
| `tests/e2e/test_chess_training_e2e.py` | New: end-to-end chess training smoke test |

## Sprint Breakdown

### Sprint 1 (Current): Foundation — Model + Self-Play Generalization

- [ ] **E1.1** Add `action_space_size` and `game_type` to `OperatorConfig` schema
- [ ] **E1.2** Create `ChessPolicyHead` in `src/modeling/model.py`
- [ ] **E1.3** Make `AlphaGalerkinModel` select head based on `game_type`
- [ ] **E2.1** Generalize `SelfPlayWorker` to accept `GameInterface`
- [ ] **E2.2** Generalize `Trainer` to wire game instance through pipeline
- [ ] **E3.1** Create `config/train_chess.yaml`
- [ ] **E3.2** Create `scripts/train_chess.py` CLI entry point

### Sprint 2: Evaluation + Testing

- [ ] **E4.1** Wire Stockfish benchmark eval into training loop
- [ ] **E5.1** `StatefulGameWrapper` unit tests
- [ ] **E5.2** Chess model forward pass tests
- [ ] **E5.3** Chess self-play integration tests
- [ ] **E5.4** Chess engine match tests
- [ ] **E5.5** E2E chess training smoke test

### Sprint 3: Launch Training Run

- [ ] Launch multi-hour self-play training run
- [ ] Monitor Elo progress against Stockfish (depth 1→5)
- [ ] Checkpoint and resume support validation

## DeepMind Specification Alignment

| Parameter | AlphaZero Value | Our Value | Notes |
|---|---|---|---|
| Self-play games per iteration | 25,000 | 25-100 | Scaled for single-GPU |
| MCTS simulations/move | 800 | 100-400 | GPU-bound, tune for speed |
| Dirichlet alpha (chess) | 0.3 | 0.3 | Wider exploration than Go's 0.03 |
| Dirichlet epsilon | 0.25 | 0.25 | Root noise fraction |
| Temperature schedule | 1.0 → 0.0 at move 30 | 1.0 → 0.0 at move 30 | Match exactly |
| c_puct | 2.5 | 1.5-2.5 | Tune |
| Learning rate | 0.01 → 0.001 → 0.0001 | Step schedule | Match paper |
| Replay buffer | 1M games | 50K experiences | Scaled for single-GPU |
| Training steps | 700K | 5K-50K | Scale to hardware |
| Batch size | 4096 | 128-512 | Scale to GPU memory |
| Resign threshold | -0.9 value | -0.9 | Speeds up training |

## Risks & Mitigations

1. **GPU memory**: Chess 119-channel tensors are larger than Go's 17-channel. Mitigation: reduce batch size, use AMP.
2. **Training time**: Single-GPU training will be orders of magnitude slower than TPU pods. Mitigation: focus on proving the pipeline works, not matching DeepMind strength.
3. **Action space mismatch**: Policy head change is fundamental. Mitigation: thorough testing of `ChessPolicyHead` + MCTS integration.
