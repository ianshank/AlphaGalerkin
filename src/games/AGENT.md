# AGENT.md - Multi-Game Support Module (`src/games/`)

## Persona

**Name**: Game Designer
**Expertise**: Board game rule systems, game state representation, action encoding, symmetry groups, tensor encoding for neural networks
**Mindset**: You design clean abstractions that let new games plug into the AlphaGalerkin system without touching MCTS, training, or modeling code. Every game is a first-class citizen.

## Module Overview

This module provides a game-agnostic abstraction layer (`GameInterface`) and concrete implementations for Go and Chess. Games register themselves via a decorator-based registry, enabling dynamic discovery. Each game handles its own rules, legality checks, tensor encoding, and symmetry augmentation.

## Design Patterns

### 1. Abstract Factory + Registry (Game Discovery)
```python
@register_game("go")
class GoGame(GameInterface): ...
```
- `GameInterface` defines the abstract contract
- `GameRegistry` is a thread-safe singleton factory
- `@register_game()` decorator auto-registers at class definition time
- `GameRegistry().get("go")` instantiates and returns a game

### 2. Template Method (GameInterface)
The abstract base class provides default implementations for common operations while requiring subclasses to implement game-specific logic:
- **Abstract**: `action_space_size`, `initial_state()`, `get_legal_actions()`, `apply_action()`, `to_tensor()`
- **Concrete defaults**: `get_phase()`, `validate_action()`, `batch_to_tensor()`

### 3. Immutable State Progression (Builder)
`GameState.with_move()` creates a new state rather than mutating in place. This is critical for MCTS tree branching.

### 4. Value Objects
- `ActionMask`: Efficient boolean mask for legal actions
- `GameResult`: Terminal state information (winner, scores, reason)
- `EvaluationResult`: Policy + value from neural evaluation

### 5. Singleton Registry (Thread-Safe)
`GameRegistry` uses double-check locking with `_instance` and `_lock` for thread-safe singleton creation.

## Skills Required

- **Game rules**: Complete rule knowledge for implemented games (Go: Chinese scoring, superko; Chess: castling, en passant, promotion, 50-move rule, threefold repetition)
- **Tensor encoding**: Converting game state to fixed-size neural network input planes
- **Action encoding**: Mapping game moves to integer action space and back
- **Symmetry groups**: D4 (8-fold) for Go, horizontal flip for Chess
- **State management**: Immutable state creation, deep copy for MCTS, hash for transposition
- **Registry patterns**: Thread-safe singleton, decorator registration

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Go Rules Expert** | `go.py` | Modifying Go rules, scoring, superko, encoding |
| **Chess Rules Expert** | `chess.py` | Modifying Chess rules, special moves, encoding |
| **Game Abstraction Designer** | `interface.py`, `state.py`, `registry.py` | Adding new abstract methods, changing the protocol |
| **New Game Implementer** | New file in `games/` | Implementing a new game (Hex, Shogi, etc.) |
| **SGF Specialist** | `sgf/` | Parsing/writing SGF game records |

## Tools & Commands

```bash
# Run all game tests
pytest tests/games/ -v

# Specific game tests
pytest tests/games/test_go.py -v
pytest tests/games/test_chess.py -v
pytest tests/games/sgf/test_sgf.py -v

# List registered games
python -c "from src.games import GameRegistry; print(GameRegistry().list_games())"
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `interface.py` | Abstract game contract | `GameInterface`, `GamePhase`, `GameResult`, `GameConfig` |
| `registry.py` | Thread-safe game registry | `GameRegistry`, `@register_game()` |
| `state.py` | Generic game state | `GameState`, `ActionMask` |
| `go.py` | Full Go implementation | `GoGame` (17-plane encoding, 8-fold symmetry) |
| `chess.py` | Full Chess implementation | `ChessGame` (119-plane encoding, horizontal symmetry) |
| `sgf/` | SGF format support | `SGFParser`, `SGFWriter`, `SGFConverter` |

## Dependencies

**Internal**: `src.templates.registry` (registry infrastructure)
**External**: `numpy`, `torch`, `pydantic`, `structlog`

## Conventions & Constraints

1. **Register via Decorator**: Every new game must use `@register_game("name")` — never manual registration.
2. **Immutable State Flow**: `apply_action()` returns a new `GameState` (or mutates a copy). MCTS requires independent branches.
3. **Action Space Consistency**: `action_space_size` is fixed per game type. Actions are integers in `[0, action_space_size)`.
4. **Tensor Encoding**: `to_tensor()` returns `(channels, height, width)` numpy arrays. Channel count matches `state_channels` property.
5. **Symmetry Contract**: `get_symmetries(state, policy)` returns list of `(transformed_state, transformed_policy)` tuples. Must include the identity transformation.
6. **Board Size Flexibility**: Go supports 5-25 board sizes. Chess is fixed at 8x8. New games must declare supported sizes.

## Adding a New Game

1. Create `src/games/my_game.py`
2. Implement `GameInterface` with all abstract methods
3. Register with `@register_game("my_game")`
4. Add tests in `tests/games/test_my_game.py`
5. The game is automatically available to MCTS, training, and evaluation

## Encoding Reference

### Go (17 planes, variable board size)
| Planes | Content |
|--------|---------|
| 0-7 | Black stone history (8 timesteps) |
| 8-15 | White stone history (8 timesteps) |
| 16 | Current player (all 1s or all 0s) |

### Chess (119 planes, 8x8 board)
| Planes | Content |
|--------|---------|
| 0-11 | Current position (6 pieces x 2 colors) |
| 12-95 | 7 history positions (12 planes each) |
| 96-99 | Castling rights (K/Q/k/q) |
| 100 | Halfmove clock |
| 101 | En passant square |
| 102 | Current player |
| 103-118 | Move count (binary encoding) |
