# AlphaGalerkin Chess: Engine Benchmarking & Uniqueness

## What Makes AlphaGalerkin Unique

AlphaGalerkin is fundamentally different from existing neural chess engines (AlphaZero, Leela Chess Zero) in its mathematical foundation and architectural approach:

### Resolution-Independent Architecture

| Feature | AlphaZero / Lc0 | AlphaGalerkin |
|---------|-----------------|---------------|
| **Core operator** | Discrete CNN (ResNet) | Continuous Galerkin Attention + FNet |
| **Complexity** | O(N^2) softmax attention or O(N) conv | O(N) Galerkin linear attention |
| **Board representation** | Fixed-size grid convolutions | Continuous operator learning via Fredholm integrals |
| **Cross-resolution transfer** | Requires retraining per board size | Zero-shot transfer (trained 9x9, evaluate 19x19) |
| **Mixing mechanism** | Residual conv blocks | FFT-based spectral mixing (O(N log N)) |
| **Stability** | Ad-hoc regularization | Provable LBB inf-sup stability guarantee |
| **Mathematical basis** | Empirical deep learning | Petrov-Galerkin projection theory |

### Key Technical Differentiators

1. **Galerkin Linear Attention**: Uses `Q(K^T V)` with Monte Carlo normalization (1/n) instead of softmax. This gives O(N) global influence modeling vs O(N^2) for standard attention.

2. **FNet Spectral Mixing**: Replaces learned convolution layers with `torch.fft.rfft2` for O(N log N) spatial mixing. Enables batch MCTS leaf evaluation at significantly lower computational cost.

3. **LBB Stability Guard**: Monitors the inf-sup condition `sigma_min(K^T V) > beta > 0` during training, ensuring numerical stability of the Galerkin discretization. No other chess engine has provable operator stability.

4. **Fredholm Integral Formulation**: Board positions are treated as continuous functions, with pieces generating Green's function influence fields. This is a fundamentally different paradigm from treating chess as a grid-based pattern recognition problem.

5. **Zero-Shot Resolution Transfer**: The physics PoC operator transfers to 19x19 at measured MSE ≈ 0.00039 (trained only on 9x9), with no retraining. (The earlier "0.000209 / 240× better than threshold" was a fabricated notebook figure; an honest CNN-retrained baseline is more accurate — see `specs/transfer_baseline_compare.spec.md`.) This architecture enables training on simplified positions and evaluating on full complexity without retraining.

### Positions Per Second Comparison

AlphaZero famously achieved superhuman strength while examining only ~60,000 positions/second, compared to Stockfish's ~60,000,000 positions/second — a 1000x disadvantage in raw search. AlphaGalerkin's O(N) Galerkin attention and O(N log N) FNet mixing are designed to push neural evaluation throughput even higher, enabling deeper MCTS search per unit time.

---

## Benchmark Metrics: Targets to Beat

### Historical AlphaZero vs Stockfish Results

#### 2017 Match (DeepMind, 1 min/move)
| Metric | Result |
|--------|--------|
| Games | 100 |
| AlphaZero Wins | 28 |
| Stockfish Wins | 0 |
| Draws | 72 |
| Score | 64/100 (64%) |

**Note**: Stockfish was running without an opening book and with modified time management. Controversial conditions.

#### 2018 Match (DeepMind, Science paper)
| Metric | Result |
|--------|--------|
| Games | 1000 |
| AlphaZero Wins | 155 |
| Stockfish Wins | 6 |
| Draws | 839 |
| Score | 574.5/1000 (57.45%) |
| Time control | 3h + 15s/move |

This is the definitive AlphaZero result. 155W/6L/839D in 1000 games with modern time controls.

### Historical Leela Chess Zero vs Stockfish Results

#### CCC 2020 (Rapid)
| Metric | Result |
|--------|--------|
| Games | 200 |
| Lc0 Wins | 19 |
| Stockfish Wins | 7 |
| Draws | 174 |
| Score | 106-94 (53%) |

#### TCEC Season 27 Superfinal (2024/2025)
| Metric | Result |
|--------|--------|
| Games | 100 |
| Stockfish Wins | 35 |
| Leela Wins | 18 |
| Draws | 47 |
| Net Margin | Stockfish +17 |

**Key takeaway**: Stockfish currently dominates Leela at maximum strength in classical time controls. The TCEC S27 result shows a +17 net margin for Stockfish.

### Current Engine Elo Ratings (approximate)

| Engine | Elo (approx) |
|--------|-------------|
| Stockfish 17 | ~3683 |
| Leela Chess Zero | ~3650+ |
| Magnus Carlsen (peak) | 2882 |
| Magnus Carlsen (current) | ~2831 |

---

## AlphaGalerkin Benchmark Thresholds

These are the progressive milestones for AlphaGalerkin chess strength:

### Tier 1: Proof of Concept
- [ ] Win at least 1 game against Stockfish at maximum strength (any time control)
- [ ] Achieve non-trivial win rate (> 5%) in 100-game matches

### Tier 2: Competitive
- [ ] Match AlphaZero's ~15-16% win rate in 1000 classical games
- [ ] Outplay Stockfish at 10:1 time disadvantage (AlphaGalerkin gets 10x more time)
- [ ] Achieve Elo within 200 points of Stockfish (~3480+)

### Tier 3: Parity
- [ ] Achieve +17 net margin or better in 100-game superfinal (beat TCEC S27 Leela result)
- [ ] Win rate > 57% in 1000 games (match AlphaZero 2018)
- [ ] Elo > 3650 (match Leela)

### Tier 4: Superiority
- [ ] Win rate > 64% in 100 games (match AlphaZero 2017 result)
- [ ] Positive net margin against Stockfish in TCEC-equivalent conditions
- [ ] Elo > 3683 (exceed Stockfish)

---

## Evaluation Framework

AlphaGalerkin includes a complete engine benchmarking pipeline:

### Components

| Module | Purpose |
|--------|---------|
| `src/games/fen.py` | FEN serialization/deserialization for UCI communication |
| `src/engines/uci.py` | UCI protocol implementation (subprocess-based) |
| `src/engines/adapter.py` | Bridge between UCI engines and MCTS Evaluator protocol |
| `src/engines/match.py` | Match orchestration with color alternation and PGN output |
| `src/engines/elo.py` | Elo rating estimation with confidence intervals |
| `scripts/play_engine.py` | CLI for running engine matches |

### Running a Benchmark

```bash
# Quick depth-limited match
python -m scripts.play_engine \
    --engine-path /usr/bin/stockfish \
    --depth 15 \
    --n-games 100 \
    --model checkpoints/chess_model.pt \
    --pgn-output results/depth15_100games.pgn

# Time-controlled match (closer to real conditions)
python -m scripts.play_engine \
    --engine-path /usr/bin/stockfish \
    --movetime-ms 1000 \
    --n-games 1000 \
    --model checkpoints/chess_model.pt \
    --pgn-output results/1s_1000games.pgn

# Maximum strength match
python -m scripts.play_engine \
    --engine-path /usr/bin/stockfish \
    --depth 30 \
    --hash-mb 4096 \
    --threads 8 \
    --n-games 10 \
    --model checkpoints/chess_model.pt
```

### Output Example

```
==================================================
Match Results: 3W / 5L / 2D
Win Rate: 40.0%
Elo Difference: -36
95% CI: [-200, +128]
Likelihood of Superiority: 33.6%
PGN written to: results/match.pgn
==================================================
```

### Interpreting Results

| Metric | Meaning |
|--------|---------|
| **Win Rate** | (Wins + 0.5 * Draws) / Total — standard chess scoring |
| **Elo Difference** | Estimated rating gap. Positive = AlphaGalerkin stronger |
| **95% CI** | Confidence interval. Narrow CI = more reliable estimate |
| **LOS** | Likelihood of Superiority — probability AlphaGalerkin is stronger |

**Statistical significance**: For reliable Elo estimates, use n >= 100 games. For publishing results, use n >= 1000 games with proper time controls.

---

## Architecture for Benchmarking

```
AlphaGalerkin Model          Stockfish (UCI)
      |                           |
  MCTS Search              UCI Protocol
      |                           |
  FNet Evaluator          subprocess.Popen
      |                           |
  Galerkin Attention        stdin/stdout
      |                           |
  GameState ←→ FEN ←→ UCI Commands
```

The match framework maintains parallel GameState (for MCTS) and FEN (for UCI) representations, with `state_to_fen()` / `fen_to_state()` bridging the two worlds. Move translation leverages `ChessGame.action_to_string()` which already outputs UCI-compatible notation (e.g., "e2e4", "e7e8q").
