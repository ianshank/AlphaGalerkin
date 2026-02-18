# Implementation Plan: Remaining Tier 1 & Tier 2 PettingZoo Games

## Status Summary

| Item | Game | GameInterface | PettingZoo Factory | Tests | Status |
|------|------|:---:|:---:|:---:|--------|
| 1.1 | Go | Done | Done | Done | **Complete** |
| 1.2 | Chess | Done (has bugs) | Missing | Missing | **90% — needs wrapper + fixes** |
| 1.3 | Connect Four | Missing | Missing | Missing | **0% — full implementation** |
| 1.4 | Tic-Tac-Toe | Missing | Missing | Missing | **0% — full implementation** |
| 2.1 | Othello | Done | Done | Done | **Complete** |
| 2.2 | Hex | Done | Done | Done | **Complete** |
| 2.3 | Checkers | Missing | Missing | Missing | **0% — full implementation** |

---

## Implementation Order

1. **Chess PettingZoo wrapper** (lowest effort — game exists, just needs factory + bug fix)
2. **Tic-Tac-Toe** (simplest new game — useful as smoke test, validates patterns)
3. **Connect Four** (moderate — non-square board, gravity mechanic, variable grid)
4. **Checkers** (most complex — multi-jump captures, kings, forced capture rules)

---

## Item 1: Chess PettingZoo Wrapper (Tier 1.2)

### Problem
`ChessGame` exists at `src/games/chess.py` with full rules (castling, en passant,
promotion, 119-plane AlphaZero encoding, 4672-action space). However:
- **Bug**: `get_result()` passes `scores={}` dict but `GameResult` expects
  positional args `score_black: float, score_white: float, move_count: int`
  (lines 913, 922-928, 932, 937, 945, 952, 955)
- No `chess_env()` factory in `src/pettingzoo/environments.py`
- No export in `src/pettingzoo/__init__.py`
- No PettingZoo-level tests

### Changes Required

#### 1a. Fix `src/games/chess.py` — `get_result()` GameResult calls
Replace all `GameResult(winner=..., reason=..., scores={...})` with proper
`GameResult(winner=..., score_black=..., score_white=..., reason=..., move_count=...)`.

Seven call sites to fix:
- Line 913: `game_ongoing` — add `score_black=0.0, score_white=0.0, move_count=state.move_number`
- Line 922-928: `checkmate` — convert `scores` dict to `score_black`/`score_white`, add `move_count`
- Line 932: `stalemate` — same pattern
- Line 937: `fifty_move_rule` — same pattern
- Line 945: `threefold_repetition` — same pattern
- Line 952: `insufficient_material` — same pattern
- Line 955: `unknown` — add missing fields

#### 1b. Add `chess_env()` factory to `src/pettingzoo/environments.py`
Follow Go/Othello/Hex pattern:
```python
from src.games.chess import ChessGame

def chess_env(
    render_mode: str | None = None,
    **config_kwargs: object,
) -> AlphaGalerkinAECEnv:
    game = ChessGame()
    config = PettingZooConfig(
        board_size=8,  # Chess is always 8×8
        render_mode=render_mode,
        agent_prefix="player",
        **config_kwargs,
    )
    return AlphaGalerkinAECEnv(game=game, config=config)
```

No `board_size` parameter — Chess is fixed 8×8.

#### 1c. Update `src/pettingzoo/__init__.py`
Add `chess_env` to imports and `__all__`.

#### 1d. Add `tests/pettingzoo/test_chess_env.py`
Tests:
- Factory creates valid environment
- Correct observation space shape: `(8, 8, 119)`
- Correct action space size: `4672`
- Reset produces valid observations
- Step with legal action succeeds
- Action mask matches legal actions
- Full game to completion (Scholar's mate or similar short game)
- Reward structure on checkmate
- Reward structure on stalemate

#### 1e. Verify existing `tests/games/test_chess.py` still passes
The GameResult fix should not break existing tests (or fix currently-broken ones).

### Files Changed
- `src/games/chess.py` (edit — fix 7 GameResult calls)
- `src/pettingzoo/environments.py` (edit — add chess_env factory)
- `src/pettingzoo/__init__.py` (edit — add chess_env export)
- `tests/pettingzoo/test_chess_env.py` (new)

---

## Item 2: Tic-Tac-Toe (Tier 1.4)

### Specification
- **Board**: 3×3 (fixed)
- **Players**: 2 (X=1, O=-1)
- **Actions**: 9 (one per cell, `action = row * 3 + col`)
- **Win condition**: 3-in-a-row (horizontal, vertical, diagonal)
- **Draw**: Board full with no winner
- **Tensor encoding**: 3 planes (own pieces, opponent pieces, player indicator)
- **Symmetries**: 8-fold D4 (4 rotations × 2 reflections), same as Othello
- **Resolution independence**: Not applicable (fixed 3×3), but useful as CI smoke test

### Implementation

#### 2a. `src/games/tictactoe.py`
```python
@register_game("tictactoe")
class TicTacToeGame(GameInterface):
    name = "tictactoe"
    description = "Tic-Tac-Toe (Noughts and Crosses)"
    min_board_size = 3
    max_board_size = 3
    default_board_size = 3
```

Key methods:
- `action_space_size` → 9 (3²)
- `state_channels` → 3
- `initial_state()` → empty 3×3 board, player 1
- `get_legal_actions()` → indices of empty cells
- `get_action_mask()` → boolean mask over 9 positions
- `apply_action()` → place piece, switch player
- `is_terminal()` → check 3-in-a-row (8 lines) or board full
- `get_winner()` → check all 8 winning lines
- `get_result()` → GameResult with proper fields
- `to_tensor()` → 3-plane encoding (own, opponent, turn indicator)
- `get_symmetries()` → 8-fold D4 symmetry

Win detection: check 3 rows + 3 columns + 2 diagonals = 8 lines.

#### 2b. `tictactoe_env()` factory in `src/pettingzoo/environments.py`
```python
def tictactoe_env(render_mode=None, **config_kwargs):
    game = TicTacToeGame()
    config = PettingZooConfig(board_size=3, render_mode=render_mode, ...)
    return AlphaGalerkinAECEnv(game=game, config=config)
```

#### 2c. Update `src/games/__init__.py`
Add `tictactoe` import to trigger registration.

#### 2d. Update `src/pettingzoo/__init__.py`
Add `tictactoe_env` to imports and `__all__`.

#### 2e. `tests/games/test_tictactoe.py`
Test classes (following Othello/Hex pattern):
- `TestTicTacToeRegistration` — registry lookup, game info
- `TestTicTacToeInitialization` — initial state, board shape, empty board
- `TestTicTacToeActionSpace` — size = 9, all legal from initial
- `TestTicTacToeLegalActions` — empty cells = legal, shrinks after moves
- `TestTicTacToeApplyAction` — piece placement, player switching
- `TestTicTacToeTerminal` — win in row/col/diagonal, draw (full board)
- `TestTicTacToeResult` — GameResult correctness
- `TestTicTacToeTensor` — shape (3, 3, 3), dtype, plane contents
- `TestTicTacToeSymmetries` — 8 transformations
- `TestTicTacToeFullGame` — play to completion, optimal play → draw

#### 2f. `tests/pettingzoo/test_tictactoe_env.py`
- Factory creates valid environment
- Observation/action space shapes
- Full game loop via `agent_iter()`
- Reward on win/draw

### Files Changed
- `src/games/tictactoe.py` (new)
- `src/games/__init__.py` (edit — add tictactoe import)
- `src/pettingzoo/environments.py` (edit — add tictactoe_env factory)
- `src/pettingzoo/__init__.py` (edit — add tictactoe_env export)
- `tests/games/test_tictactoe.py` (new)
- `tests/pettingzoo/test_tictactoe_env.py` (new)

---

## Item 3: Connect Four (Tier 1.3)

### Specification
- **Board**: 6 rows × 7 columns (default), variable N×M for resolution independence
- **Players**: 2 (piece=1, piece=-1)
- **Actions**: `num_cols` (column drop — piece falls to lowest empty row)
- **Win condition**: 4-in-a-row (horizontal, vertical, diagonal)
- **Draw**: Board full with no winner
- **Tensor encoding**: 3 planes (own pieces, opponent pieces, player indicator)
- **Symmetries**: 2-fold (identity + horizontal flip) — board is symmetric left-right
- **Resolution independence**: Variable grid sizes (e.g., train 6×7, evaluate 8×10)

### Design Decisions
- **Non-square board**: The `GameInterface` assumes `board_size` is a single int for
  square boards. Connect Four needs `board_height` and `board_width`. Options:
  - **(A) Encode as square**: Pad to max(rows, cols) = 7×7, with row 0 unused → wastes space
  - **(B) Use metadata**: `board_size = num_cols`, store `board_height` in `GameState.metadata`
  - **(C) Extend interface**: Add optional `board_height`/`board_width` properties

  **Recommendation: Option B.** `board_size` = `num_cols` (7), `metadata["board_height"]` = 6.
  The tensor is padded to `(3, max(H,W), max(H,W))` to maintain square shape for the
  neural network. This is consistent with the existing interface and keeps backward compatibility.

- **Action space**: `Discrete(num_cols)` — column index only. No pass action.

- **Variable sizes**: Support arbitrary `(rows, cols)` pairs via `board_size` (cols) and
  `metadata["board_height"]` (rows). Default is 6×7. For resolution independence demos,
  support ranges like 4×5 through 10×12.

### Implementation

#### 3a. `src/games/connect_four.py`
```python
@register_game("connect_four")
class ConnectFourGame(GameInterface):
    name = "connect_four"
    description = "Connect Four with variable grid sizes"
    min_board_size = 4   # minimum columns
    max_board_size = 15  # maximum columns
    default_board_size = 7  # standard 7 columns
```

Constants:
- `_DEFAULT_ROWS = 6`
- `_STATE_CHANNELS = 3`
- `_WIN_LENGTH = 4`

Key methods:
- `action_space_size` → `self._board_size` (number of columns)
- `state_channels` → 3
- `initial_state(board_size)` → empty board, `metadata["board_height"]` = rows
  Default rows = `board_size - 1` (maintain ~6:7 ratio) or configurable
- `get_legal_actions()` → columns where top row is empty
- `apply_action(state, col)` → find lowest empty row in column, place piece
- `is_terminal()` → check 4-in-a-row or board full
- `get_winner()` → scan for 4-in-a-row in all directions
- `to_tensor()` → 3-plane encoding, padded to square `(3, S, S)` where `S = max(rows, cols)`
- `get_symmetries()` → 2-fold: identity + horizontal flip (left-right mirror)

Win detection: scan all cells, check 4 directions (right, down, down-right, down-left).

Gravity mechanic: `apply_action(state, col)` finds `max(row)` where `board[row][col] == 0`.

#### 3b. `connect_four_env()` factory
```python
def connect_four_env(board_size=None, board_height=None, render_mode=None, **config_kwargs):
    game = ConnectFourGame()
    if board_height is not None:
        config_kwargs["board_height"] = board_height
    config = PettingZooConfig(board_size=board_size, render_mode=render_mode, ...)
    return AlphaGalerkinAECEnv(game=game, config=config)
```

#### 3c–3d. Registry + exports
Same pattern as Tic-Tac-Toe.

#### 3e. `tests/games/test_connect_four.py`
- Registration, initialization (6 rows × 7 cols default)
- Action space = 7, legal actions = all 7 columns initially
- Gravity: piece drops to bottom row
- Gravity: stacking pieces in same column
- Column full → not legal
- Win detection: horizontal, vertical, diagonal-up, diagonal-down
- Draw detection (full board, no 4-in-a-row)
- Variable board sizes (4×5, 8×10)
- Tensor shape and encoding
- Symmetry: 2-fold horizontal flip
- Full game to completion

#### 3f. `tests/pettingzoo/test_connect_four_env.py`
- Factory creates valid environment
- Observation/action space shapes
- Full game loop
- Variable board size (cross-resolution test)

### Files Changed
- `src/games/connect_four.py` (new)
- `src/games/__init__.py` (edit)
- `src/pettingzoo/environments.py` (edit)
- `src/pettingzoo/__init__.py` (edit)
- `tests/games/test_connect_four.py` (new)
- `tests/pettingzoo/test_connect_four_env.py` (new)

---

## Item 4: Checkers / Draughts (Tier 2.3)

### Specification
- **Board**: 8×8 (fixed), only dark squares used (32 playable squares)
- **Players**: 2 (piece=1/king=2, piece=-1/king=-2)
- **Pieces**: Regular pieces + kings (promoted upon reaching back rank)
- **Movement**: Diagonally forward (regular), diagonally any direction (kings)
- **Captures**: Jump over opponent piece to empty square, multi-jump chains
- **Forced capture**: If a capture is available, the player MUST capture
- **Win condition**: Capture all opponent pieces or block all opponent moves
- **Draw**: No captures and no pawn moves for 40 moves (optional rule)
- **Tensor encoding**: 5 planes (own pieces, own kings, opponent pieces, opponent kings, player)
- **Symmetries**: 2-fold (identity + horizontal flip)
- **Resolution independence**: Not applicable (fixed 8×8)

### Design Decisions
- **Action encoding**: Use `(from_square, to_square)` mapped to a flat index.
  32 playable squares × 4 possible landing squares = 128 max actions.
  But with multi-jumps, a single "action" is a full jump sequence.

  **Recommendation**: Encode single-step moves as `from_square * 4 + direction_index`.
  32 from-squares × 4 directions = 128 actions. For multi-jumps, each jump is a
  separate step (the game stays with the same player until the jump chain completes).
  This keeps the action space small and consistent.

  Alternative: Flat index `from * N + to` over dark squares: 32 × 32 = 1024.
  **Use the compact 128-action encoding** to match action space sizes of other games.

- **Multi-jump handling**: When a capture is made and the jumping piece can continue
  jumping, the same player moves again. The state's `metadata["must_continue_jump"]`
  tracks this. The `current_player` does NOT switch until the jump chain ends.

- **Forced capture**: `get_legal_actions()` returns only capture moves when captures
  are available. This is standard International/American Checkers rules.

### Implementation

#### 4a. `src/games/checkers.py`
```python
@register_game("checkers")
class CheckersGame(GameInterface):
    name = "checkers"
    description = "American Checkers (English Draughts)"
    min_board_size = 8
    max_board_size = 8
    default_board_size = 8
```

Constants:
- `_STATE_CHANNELS = 5` (own pieces, own kings, opp pieces, opp kings, turn)
- `_ACTION_SPACE = 128` (32 squares × 4 directions)
- Piece types: `EMPTY=0, BLACK_PIECE=1, BLACK_KING=2, WHITE_PIECE=-1, WHITE_KING=-2`

Key design:
- **Square mapping**: 32 playable dark squares mapped to indices 0-31.
  `dark_squares = [(r,c) for r in range(8) for c in range(8) if (r+c) % 2 == 1]`
- **Direction encoding**: 4 diagonal directions for single moves and captures
  - Direction 0: forward-left, Direction 1: forward-right
  - Direction 2: backward-left, Direction 3: backward-right
  - Regular pieces: only forward (directions 0-1 for moves, 0-3 for captures)
  - Kings: all 4 directions

Key methods:
- `action_space_size` → 128
- `state_channels` → 5
- `initial_state()` → standard Checkers setup (12 pieces per player on dark squares)
- `get_legal_actions()` → if captures available, return only captures (forced capture rule)
- `apply_action()` → move piece, capture if jump, promote if back rank, handle multi-jump
- `is_terminal()` → no legal moves for current player
- `get_winner()` → player with no pieces/moves loses
- `get_result()` → proper GameResult
- `to_tensor()` → 5-plane encoding
- `get_symmetries()` → 2-fold (identity + horizontal flip)

Multi-jump state machine:
```
apply_action(state, action):
    1. Move piece diagonally
    2. If capture: remove jumped piece
    3. If piece reaches back rank: promote to king
    4. If capture AND more captures available from landing square:
       - Set metadata["must_continue_jump"] = landing_square
       - Do NOT switch player
    5. Else:
       - Clear metadata["must_continue_jump"]
       - Switch player
```

#### 4b. `checkers_env()` factory
```python
def checkers_env(render_mode=None, **config_kwargs):
    game = CheckersGame()
    config = PettingZooConfig(board_size=8, render_mode=render_mode, ...)
    return AlphaGalerkinAECEnv(game=game, config=config)
```

#### 4c–4d. Registry + exports
Same pattern as other games.

#### 4e. `tests/games/test_checkers.py`
- Registration, initialization (12 pieces per side)
- Initial board setup (pieces on correct dark squares)
- Action space = 128
- Legal actions from initial position (forward moves only)
- Simple moves (diagonal forward)
- Simple captures (jump over opponent)
- Multi-jump captures (chain of jumps)
- Forced capture rule (when capture available, only captures returned)
- King promotion (reaching back rank)
- King movement (all 4 directions)
- King captures (backward jumps)
- Terminal detection (no pieces, no moves)
- Draw detection (if implemented)
- Tensor encoding (5 planes)
- Symmetry (2-fold)
- Full game to completion

#### 4f. `tests/pettingzoo/test_checkers_env.py`
- Factory, spaces, game loop, rewards

### Files Changed
- `src/games/checkers.py` (new)
- `src/games/__init__.py` (edit)
- `src/pettingzoo/environments.py` (edit)
- `src/pettingzoo/__init__.py` (edit)
- `tests/games/test_checkers.py` (new)
- `tests/pettingzoo/test_checkers_env.py` (new)

---

## Cross-Cutting Concerns

### Gap: GameResult Consistency
The `GameResult` dataclass requires `(winner, score_black, score_white, reason, move_count)`.
Chess currently passes `scores={}` dict — must fix. All new implementations must use
the correct positional arguments. Verify Othello and Hex implementations also comply.

### Logging
All new game implementations use `structlog.get_logger(__name__)` for structured logging,
following the pattern in `othello.py` and `hex.py`. Key log events:
- `game_initialized` (debug)
- `action_applied` (debug)
- `game_terminal` (info)

### Code Reuse Patterns
- Tic-Tac-Toe and Connect Four share win-detection logic (N-in-a-row scanning).
  Extract a shared `_check_n_in_a_row(board, n, directions)` helper if it simplifies
  both implementations. Otherwise keep inline — don't over-abstract for 2 uses.
- All games share the same 3-plane or 5-plane tensor encoding pattern.
- All games share the D4 symmetry code (Tic-Tac-Toe/Othello) or horizontal flip
  (Connect Four/Checkers/Chess).

### Backward Compatibility
- No changes to `GameInterface`, `GameState`, `ActionMask`, or `GameRegistry` contracts.
- New games register themselves via `@register_game()` — existing code unaffected.
- New `*_env()` factories added alongside existing ones — no breaking changes.
- PettingZooConfig unchanged — all games use the same config structure.

### Test Strategy
- Unit tests per game: 40-60 tests each (following Othello's 52 / Hex's 54 pattern)
- PettingZoo integration tests per game: 8-15 tests each
- All tests must run independently (`pytest tests/games/test_xxx.py -v`)
- No test should depend on GPU or external network
- Use `pytest.fixture` for game instances to avoid repeated setup

### Documentation Updates
- Update `CLAUDE.md` with new milestones and directory entries
- Update `docs/PETTINGZOO_DEMO_PLAN.md` status table
- Update `pyproject.toml` if new dependencies are needed (none expected)

---

## Estimated Scope

| Item | New Files | Edited Files | Est. Tests |
|------|-----------|-------------|------------|
| Chess wrapper | 1 | 3 | ~15 |
| Tic-Tac-Toe | 2 | 3 | ~45 |
| Connect Four | 2 | 3 | ~50 |
| Checkers | 2 | 3 | ~55 |
| **Total** | **7** | **~5 unique** | **~165** |

---

## Execution Order (Recommended)

```
Step 1: Fix Chess GameResult bug + add chess_env() + tests
Step 2: Implement Tic-Tac-Toe game + tictactoe_env() + tests
Step 3: Implement Connect Four game + connect_four_env() + tests
Step 4: Implement Checkers game + checkers_env() + tests
Step 5: Update CLAUDE.md, PETTINGZOO_DEMO_PLAN.md
Step 6: Run full test suite, verify all pass
Step 7: Commit and push
```
