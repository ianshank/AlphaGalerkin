# PettingZoo Demo Plan for AlphaGalerkin

## Motivation

AlphaGalerkin has proven resolution-independent zero-shot transfer on Go
(train on 9x9, evaluate on 19x19 with MSE 0.000209 — 240x better than
threshold). [PettingZoo](https://pettingzoo.farama.org/) is the standard
multi-agent RL environment library (the multi-agent analogue of Gymnasium).
Wrapping AlphaGalerkin behind PettingZoo's AEC API would:

1. Let anyone benchmark AlphaGalerkin against standard MARL baselines.
2. Showcase resolution independence on environments the community already knows.
3. Open the door to cooperative and imperfect-information games that stress
   different parts of the architecture.

This document maps PettingZoo environments to AlphaGalerkin's strengths,
ordered by implementation feasibility and demo value.

---

## AlphaGalerkin Strengths to Showcase

| Capability | What it proves | Best demo environments |
|---|---|---|
| Resolution independence (Fourier embeddings, no hard-coded board size) | Train small, play large — zero-shot | Go (variable board), Othello variants, scalable Connect Four |
| O(N) Galerkin attention | Scales to large state spaces without quadratic blowup | Go 19x19+, large-grid games |
| FFT mixing (FNet blocks) | Fast batch MCTS leaf evaluation | Any MCTS-driven game (Go, Chess, Connect Four) |
| Gumbel AlphaZero MCTS | State-of-the-art tree search | Chess, Go, Connect Four |
| Multi-game registry (`@register_game`) | Drop-in new games, same training loop | Every game below |

---

## Tier 1 — Direct Fits (PettingZoo wrappers only)

These are turn-based, two-player, perfect-information, grid-based games.
AlphaGalerkin already has the architecture for them. Work is limited to
writing a thin PettingZoo AEC wrapper and (for new games) a `GameInterface`
implementation.

### 1.1 Go (`go_v5`) — flagship demo

- **PettingZoo env**: `pettingzoo.classic.go_v5`
- **Status**: AlphaGalerkin already implements Go (`src/games/go.py`).
- **Work required**: Write a `PettingZooGoWrapper` that adapts
  `GoGame` → PettingZoo AEC API (observation dict with `action_mask`,
  agent iteration, reward on terminal).
- **Demo value**: **Very high**. Variable `board_size` (9, 13, 19) is
  PettingZoo Go's constructor parameter. This is the single best
  environment to demonstrate resolution-independent zero-shot transfer
  in a setting the RL community already benchmarks on.
- **Resolution independence angle**: Train on `go_v5(board_size=9)`,
  evaluate on `go_v5(board_size=19)` — directly comparable to baselines
  that must retrain per board size.

### 1.2 Chess (`chess_v6`)

- **PettingZoo env**: `pettingzoo.classic.chess_v6`
- **Status**: AlphaGalerkin already implements Chess (`src/games/chess.py`,
  119-plane AlphaZero encoding, 4672-action space).
- **Work required**: AEC wrapper. PettingZoo Chess uses the same AlphaZero
  encoding scheme, so observation/action mapping is nearly 1:1.
- **Demo value**: **High**. Chess is fixed 8x8, so it does not showcase
  resolution independence, but it demonstrates multi-game generality
  (same Galerkin backbone, same training loop, different game).
- **Unique angle**: Compare O(N) Galerkin attention vs. standard
  transformer attention on the same Chess benchmark.

### 1.3 Connect Four (`connect_four_v3`)

- **PettingZoo env**: `pettingzoo.classic.connect_four_v3`
- **Observation**: (6, 7, 2) — two binary planes on a 6×7 grid.
- **Action space**: Discrete(7) — column selection.
- **Work required**: Implement `ConnectFourGame(GameInterface)` +
  AEC wrapper. Straightforward: small board, simple rules, no captures.
- **Demo value**: **High**. Connect Four is a canonical MARL benchmark.
  Fast to train (small state space), good for validating the full pipeline
  end-to-end before tackling harder games. Solved game, so optimal play
  is known — measurable against perfect play.
- **Resolution independence angle**: The standard game is 6×7, but
  Connect Four generalizes to arbitrary (rows × cols). Implement
  variable grid size and demo train-on-6x7 → evaluate-on-10x12.

### 1.4 Tic-Tac-Toe (`tictactoe_v3`)

- **PettingZoo env**: `pettingzoo.classic.tictactoe_v3`
- **Observation**: (3, 3, 2) — two binary planes on 3×3.
- **Action space**: Discrete(9).
- **Work required**: Implement `TicTacToeGame(GameInterface)` +
  AEC wrapper. Trivial game logic.
- **Demo value**: **Medium**. Useful as a "hello world" integration test
  and CI smoke test. Should converge to perfect play in minutes. Not
  interesting for a research demo on its own, but valuable for verifying
  the PettingZoo adapter layer works correctly before scaling up.

---

## Tier 2 — High-Value Extensions (new GameInterface + moderate adaptation)

These require implementing a new game but fit cleanly into the existing
two-player, perfect-information, grid-based architecture.

### 2.1 Othello / Reversi

- **PettingZoo**: Available as `othello_v3` in the Atari family (pixel-based)
  and via third-party packages (board-based). A clean board-level
  implementation would be more useful.
- **Observation**: (8, 8, C) — binary planes for black/white stones.
- **Action space**: Discrete(65) — 64 board positions + pass.
- **Work required**: Implement `OthelloGame(GameInterface)` from scratch
  (simpler than Go — no liberty counting, no ko).
- **Demo value**: **Very high**. Othello naturally generalizes to NxN
  boards (6×6, 8×8, 10×10, 12×12). This makes it an ideal second
  showcase for resolution independence alongside Go. The game is
  well-studied with known strong baselines.
- **Resolution independence angle**: Train on 6×6 Othello → zero-shot
  evaluate on 8×8 and 10×10. This is a fresh result — no prior work
  has demonstrated cross-resolution Othello transfer.

### 2.2 Hex

- **PettingZoo**: Not in PettingZoo core, but a natural fit for the
  classic category and available in third-party packages.
- **Observation**: (N, N, 2) on a hex grid (representable as NxN square
  grid with adjacency adjustments).
- **Action space**: Discrete(N²).
- **Work required**: Implement `HexGame(GameInterface)`. Union-find for
  connectivity checking. Moderate complexity.
- **Demo value**: **Very high**. Hex is played on boards from 7×7 to 19×19
  (and beyond). It is one of the purest tests of resolution independence
  because the game mechanics are identical across sizes — no komi
  adjustments, no special rules. First player advantage is proven but
  the game is unsolved for N≥10.
- **Resolution independence angle**: Train on 7×7 → evaluate on 11×11
  and 19×19. The Galerkin operator's continuous kernel should transfer
  the connectivity-reasoning patterns across scales.

### 2.3 Checkers / Draughts

- **PettingZoo**: Was previously in PettingZoo classic (removed in recent
  versions). Available via third-party.
- **Observation**: (8, 8, C) — planes for regular pieces, kings, per player.
- **Action space**: Variable (multi-jump sequences make this non-trivial).
- **Work required**: Implement `CheckersGame(GameInterface)`. Moderate:
  multi-jump capturing, king promotion, forced capture rules.
- **Demo value**: **Medium-high**. Fixed 8×8 board (no resolution demo),
  but complements Chess as a second piece-movement game and tests the
  architecture on a game with mandatory captures and different branching
  factor.

---

## Tier 3 — Architecture-Stretching Demos (imperfect info / cooperation)

These require extending AlphaGalerkin beyond two-player perfect-information
games. Each tests a different architectural boundary.

### 3.1 Hanabi (`hanabi_v5`) — cooperative, imperfect information

- **PettingZoo env**: `pettingzoo.classic.hanabi_v5`
- **Players**: 2–5 (cooperative).
- **Observation**: Dict-based (card knowledge, hints, discard pile).
  Not naturally grid-shaped.
- **Action space**: Discrete(20) — play/discard card, reveal color/rank.
- **Architecture changes required**:
  - Extend `n_players` support beyond 2.
  - Handle imperfect information (belief modeling over hidden cards).
  - Observation is not a spatial grid — need to either (a) reshape into
    a pseudo-grid for the Galerkin backbone, or (b) add a 1D sequence
    pathway.
  - Replace win/loss reward with cooperative score (0–25).
- **Demo value**: **Very high** if successful. Hanabi is the premier
  cooperative imperfect-information benchmark. Showing that the Galerkin
  operator can learn implicit communication through action conventions
  would be a strong result.
- **Recommended approach**: Encode the observation as a (D, H, W) tensor
  where H×W is a pseudo-spatial layout of card slots, hint tokens, and
  firework piles. The Fourier positional encoding would give each "slot"
  a continuous coordinate, and attention can model inter-slot dependencies.

### 3.2 MPE Simple Adversary — mixed cooperative/competitive

- **PettingZoo env**: `mpe2.simple_adversary`
- **Players**: 3 (1 adversary + 2 cooperators).
- **Observation**: Continuous vectors (positions, velocities, landmarks).
- **Action space**: Discrete(5) — cardinal directions + no-op.
- **Architecture changes required**:
  - Continuous observation space (no grid).
  - Simultaneous actions (not turn-based).
  - 3-player with mixed objectives.
  - Would need a parallel-action variant of the training loop.
- **Demo value**: **Medium**. Interesting as a "can Galerkin attention
  learn spatial reasoning in continuous domains?" test, but the
  environments are small enough that standard MLPs solve them.
  Not the strongest showcase for AlphaGalerkin's advantages.

### 3.3 Atari Pong / Space Invaders — pixel-based multi-agent

- **PettingZoo envs**: `pettingzoo.atari.pong_v3`,
  `pettingzoo.atari.space_invaders_v2`
- **Observation**: (210, 160, 3) RGB pixels.
- **Action space**: Discrete(6–18).
- **Architecture changes required**:
  - Replace board-state encoder with a visual encoder (CNN front-end
    feeding into Galerkin attention).
  - Handle simultaneous actions and frame stacking.
  - Reward shaping for non-terminal intermediate rewards.
- **Demo value**: **Medium-high** for Space Invaders (cooperative angle),
  lower for Pong (already well-solved). The resolution independence story
  is interesting here: can a model trained on downscaled 84×84 frames
  generalize to full 210×160? This would test Fourier positional encoding
  on natural images rather than board states.

### 3.4 Knights Archers Zombies (`knights_archers_zombies_v10`) — cooperative real-time

- **PettingZoo env**: `pettingzoo.butterfly.knights_archers_zombies_v10`
- **Players**: 2–4 (cooperative, heterogeneous agents).
- **Observation**: RGB image or vector state.
- **Action space**: Discrete(6) — rotate, move, attack.
- **Architecture changes required**: Similar to Atari — visual encoder,
  simultaneous actions, cooperative reward, heterogeneous agent types.
- **Demo value**: **Medium**. More interesting than MPE for testing
  multi-agent coordination with Galerkin attention, but the visual
  complexity is high and baselines are weak (the environment is
  intentionally hard).

---

## Recommended Implementation Order

```
Phase 1: PettingZoo Integration Layer
├── 1a. Generic AEC wrapper (src/pettingzoo/wrapper.py)
│       Adapts any GameInterface → PettingZoo AEC env
├── 1b. Go wrapper + round-trip test
│       Validate with go_v5(board_size=9) and go_v5(board_size=19)
└── 1c. Chess wrapper + round-trip test

Phase 2: New Perfect-Information Games
├── 2a. Connect Four (GameInterface + wrapper + tests)
├── 2b. Othello (GameInterface + wrapper + tests)
│       ★ Resolution independence demo: 6×6 → 8×8 → 10×10
└── 2c. Tic-Tac-Toe (GameInterface + wrapper + smoke test)

Phase 3: Resolution Independence Showcase
├── 3a. Variable-size Connect Four (extend standard 6×7 to NxM)
├── 3b. Hex (GameInterface + wrapper + variable board sizes)
│       ★ Train 7×7 → evaluate 11×11, 19×19
└── 3c. Benchmark suite: measure zero-shot transfer gap across all games

Phase 4: Architecture Extensions (stretch)
├── 4a. Hanabi (cooperative + imperfect info)
│       Requires belief modeling and pseudo-spatial encoding
├── 4b. Atari environments (pixel input pathway)
│       Requires CNN front-end + Galerkin backbone
└── 4c. MPE / Butterfly (simultaneous actions, >2 players)
```

---

## What Each Phase Proves

| Phase | Thesis validated |
|---|---|
| 1 | AlphaGalerkin plugs into the standard MARL ecosystem. Anyone can `pip install` and benchmark. |
| 2 | The `GameInterface` + `@register_game` pattern is genuinely game-agnostic. One training loop, multiple games. |
| 3 | Resolution independence is not Go-specific. It transfers to Othello, Hex, and scaled Connect Four — any grid-based perfect-information game. |
| 4 | The Galerkin operator generalizes beyond board games to cooperative, imperfect-information, and pixel-based domains. |

---

## Highest-Impact Demos (Pick 3)

If the goal is to produce the most compelling demo material with minimum
effort, these three deliver the most:

1. **Go via PettingZoo** (Phase 1b) — Flagship. Wraps existing code.
   Everyone in the MARL community knows Go. Variable board sizes are
   built into PettingZoo's Go constructor, making the resolution
   independence demo effortless to reproduce.

2. **Othello with cross-resolution transfer** (Phase 2b + 3c) — Novel
   result. No prior work has shown zero-shot Othello transfer across
   board sizes. Simple rules (easier to implement than Go), clean
   resolution scaling (6→8→10→12), and a well-studied game with strong
   baselines to compare against.

3. **Hex with cross-resolution transfer** (Phase 3b) — The purest
   resolution independence test. Identical mechanics at every scale, no
   confounding rule changes. Training on 7×7 and evaluating on 19×19
   would be a striking result for the continuous operator learning thesis.

---

## File Structure

```
src/
  pettingzoo/
    __init__.py
    wrapper.py          # Generic GameInterface → AEC adapter
    go_env.py           # Go-specific PettingZoo env
    chess_env.py         # Chess-specific PettingZoo env
    connect_four_env.py  # Connect Four PettingZoo env
    othello_env.py       # Othello PettingZoo env
    hex_env.py           # Hex PettingZoo env
  games/
    connect_four.py      # ConnectFourGame(GameInterface)
    othello.py           # OthelloGame(GameInterface)
    hex.py               # HexGame(GameInterface)
    tictactoe.py         # TicTacToeGame(GameInterface)
tests/
  pettingzoo/
    test_wrapper.py      # Generic wrapper tests
    test_go_env.py       # Go round-trip tests
    test_connect_four.py # Connect Four tests
    test_othello.py      # Othello tests
    test_hex.py          # Hex tests
  games/
    test_connect_four.py
    test_othello.py
    test_hex.py
    test_tictactoe.py
scripts/
  demo_pettingzoo.py     # CLI to run any registered game as PettingZoo env
```

---

## Dependencies

```toml
# Add to pyproject.toml
[project.optional-dependencies]
pettingzoo = [
    "pettingzoo>=1.24.0",
    "gymnasium>=0.28.0",
]
```

---

## Key Design Decision: Wrapper Direction

Two options exist for PettingZoo integration:

**Option A: AlphaGalerkin wraps PettingZoo** — Use PettingZoo's existing
environments as the ground truth, and write adapters that let AlphaGalerkin's
MCTS + neural network play within them.

**Option B: PettingZoo wraps AlphaGalerkin** — Expose AlphaGalerkin's
`GameInterface` implementations as PettingZoo environments, so external
agents can play against AlphaGalerkin or use the environments for
benchmarking.

**Recommendation: Both, but start with Option B.** AlphaGalerkin's game
implementations are richer (superko, full scoring, etc.) and
resolution-independent. Exposing them as PettingZoo environments lets the
community use them. Then add Option A adapters where PettingZoo has
environments we don't want to reimplement (e.g., Hanabi via RLCard).
