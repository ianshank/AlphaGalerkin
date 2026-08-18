# FORGE Integration Analysis for AlphaGalerkin

## Overview

[FORGE](https://github.com/ianshank/FORGE) (Fast Open-source Runtime for Generalist Environments) is a high-performance Rust simulation platform (130K+ steps/sec) with Python bindings, procedural worlds, multi-agent support, built-in MCTS, and Gymnasium/PettingZoo interfaces. This document evaluates how FORGE could serve AlphaGalerkin's mission of resolution-independent AI planning via continuous operator learning.

---

## 1. New Game Target via `GameInterface`

**The fit:** AlphaGalerkin's architecture is game-agnostic -- anything implementing the `GameInterface` protocol (Go, Chess, PDE games) can plug into the MCTS search, self-play, and training pipeline. FORGE environments expose discrete observation/action spaces through standard Gymnasium/PettingZoo APIs.

**What this enables:** A `ForgeGameAdapter` (analogous to the existing `PDEGameAdapter`) could wrap any FORGE environment as a `GameInterface`. AlphaGalerkin's neural MCTS (with Galerkin attention and FNet mixing) would then plan over FORGE's procedural worlds -- resource gathering, crafting sequences, multi-agent coordination -- all using the same training infrastructure (replay buffer, loss balancing, checkpointing).

**Why it matters for SBIR:** This demonstrates AlphaGalerkin isn't a board-game-only system. Planning over complex, open-ended environments with crafting trees, combat, and multi-agent dynamics is directly relevant to DoD planning problems (logistics, multi-asset coordination). It broadens the "generalist planner" narrative.

---

## 2. Resolution-Independence Validation on Variable-Size Worlds

**The fit:** AlphaGalerkin's core thesis is train-small, deploy-large (train on 9x9, infer on 19x19). FORGE generates worlds at arbitrary sizes -- 16x16, 32x32, 64x64, 128x128 -- with the same procedural rules.

**What this enables:** Train AlphaGalerkin's continuous operator network on small FORGE worlds (e.g., 16x16), then evaluate zero-shot on large worlds (64x64, 128x128). The grid observations (`grid_view` with 7 channels) are essentially 2D spatial fields -- exactly the kind of data Galerkin attention and Fourier features are designed for. This is a much more compelling demonstration of resolution independence than board games alone, since the environment complexity scales with size (more resources, longer plans, more agents).

**Why it matters:** The Physics PoC achieved MSE 0.000209 on Poisson equations. Replicating zero-shot transfer on a rich planning task would be a qualitatively different (and more impressive) validation.

---

## 3. High-Speed Rollout Engine for MCTS

**The fit:** AlphaGalerkin's MCTS uses neural network evaluations for leaf nodes, but still needs a forward model to simulate actions during tree search. FORGE's engine runs at <8 microseconds per step with deterministic, zero-allocation execution.

**What this enables:** Use FORGE's Rust engine as the state simulator inside AlphaGalerkin's MCTS tree expansion. Instead of Python-speed game simulation, each `node.expand()` call delegates to FORGE's compiled Rust core via PyO3. This could dramatically increase the number of MCTS simulations per second, especially for the Gumbel-AlphaZero variant which needs many simulations for sequential halving.

**Concrete path:** AlphaGalerkin's `MCTSNode.expand()` and `MCTS._simulate()` call `game.get_next_state(state, action)`. If the `ForgeGameAdapter` delegates this to `forge_env.step()`, the hot loop runs in Rust.

---

## 4. Curriculum Learning Synergy

**The fit:** Both systems have curriculum infrastructure but with different strengths:

- **AlphaGalerkin:** `BoardSizeCurriculum` -- step-based board size scheduling
- **FORGE:** 6-tier adaptive task curriculum with rolling success windows and composable task DSL

**What this enables:** FORGE's task curriculum could replace or augment AlphaGalerkin's simpler step-based curriculum. Instead of just scaling board size, you'd scale task complexity:

| Tier | Task Type |
|------|-----------|
| 1 | Navigate to location |
| 2 | Collect resources |
| 3 | Craft tools |
| 4 | Multi-step plans |
| 5 | Multi-agent coordination |
| 6 | Adversarial scenarios |

The adaptive scaling based on rolling success rates is more sophisticated than AlphaGalerkin's fixed step thresholds.

---

## 5. Multi-Agent Self-Play at Scale

**The fit:** AlphaGalerkin's `SelfPlayWorker` and `ParallelSelfPlayWorker` generate training experiences through self-play. FORGE natively supports multi-agent scenarios via PettingZoo Parallel API with communication channels.

**What this enables:** Instead of two-player board game self-play, run N-agent cooperative/competitive self-play in FORGE worlds. All agents share the same AlphaGalerkin network (weight sharing), but with per-agent observations. The `ParallelSelfPlayWorker` + FORGE's `ForgeParallelEnv` could generate diverse multi-agent experiences for the replay buffer. The communication tokens (vocabulary-based message passing) add a novel dimension -- the network learns when and what to communicate.

---

## 6. Deterministic Reproducibility for PoC Scenarios

**The fit:** AlphaGalerkin's PoC framework (`ScenarioRunner`) runs transfer, complexity, and stability benchmarks with pass/fail thresholds. FORGE guarantees byte-identical determinism (same seed + actions = same outcome).

**What this enables:** FORGE environments as reproducible PoC scenarios:

- **`ForgeTransferScenario`:** Train on 16x16 worlds, evaluate on 64x64 with exact reproducibility.
- **`ForgeComplexityScenario`:** Benchmark O(N) Galerkin attention vs O(N^2) softmax on increasing world sizes.

Determinism means every run is exactly reproducible, which is critical for SBIR proposal claims.

---

## 7. PDE Connection: Terrain as Continuous Fields

**The fit:** FORGE's procedural terrain uses multi-octave Perlin noise to generate continuous elevation/biome fields discretized onto grids. AlphaGalerkin's PDE framework solves continuous equations on meshes.

**What this enables:** FORGE's terrain generation is mathematically a PDE-like problem (Perlin noise is a smoothed random field). The MCTS-guided mesh refinement game could optimize where to allocate resolution on FORGE terrain -- spend more grid points on complex mountain/valley interfaces, fewer on flat plains. This connects AlphaGalerkin's PDE research (adaptive mesh refinement, Dorfler marking) to a practical spatial planning problem.

---

## 8. Evaluation and Benchmarking Harness

**The fit:** FORGE includes `forge-eval` (evaluation harness) and `forge-replay` (trajectory storage). AlphaGalerkin has `Evaluator` (win rate, policy agreement) and `ResultCollector`.

**What this enables:** FORGE's evaluation harness could provide standardized benchmarks: task completion rate, average reward, episode length, resource efficiency. These metrics complement AlphaGalerkin's existing evaluation (which is board-game-centric). Trajectory replay from `forge-replay` could feed into AlphaGalerkin's `ReplayBuffer` for offline training or analysis.

---

## 9. ONNX Deployment Target

**The fit:** AlphaGalerkin has a full ONNX export pipeline (export, quantize, runtime). FORGE has a WebAssembly deployment path and HTTP server.

**What this enables:** Train with AlphaGalerkin's PyTorch pipeline -> export to ONNX -> deploy alongside FORGE's WASM environment in a browser demo. The `demo_ui/` in FORGE already has real-time visualization with SSE streaming. An ONNX-quantized AlphaGalerkin model running inference in the browser while FORGE simulates the world would be a compelling interactive demo for SBIR reviewers.

---

## Summary Table

| FORGE Component | AlphaGalerkin Integration Point | Value |
|---|---|---|
| `ForgeGymnasiumEnv` | `GameInterface` protocol | New game domain beyond board games |
| Variable world sizes | Resolution-independence PoC | Transfer validation on rich environments |
| Rust engine (<8us/step) | `MCTS._simulate()` hot loop | 10-100x faster tree search |
| 6-tier task curriculum | `BoardSizeCurriculum` | Adaptive complexity scaling |
| PettingZoo multi-agent | `ParallelSelfPlayWorker` | N-agent cooperative/competitive training |
| Deterministic simulation | `ScenarioRunner` PoC framework | Reproducible benchmarks |
| Perlin noise terrain | PDE mesh refinement game | Spatial AMR on realistic fields |
| `forge-eval` + `forge-replay` | `Evaluator` + `ReplayBuffer` | Richer metrics, offline training data |
| WASM + demo UI | ONNX export pipeline | Interactive browser demos |
| MangoMAS bridge | Training pipeline | Constitutional constraints, curiosity |

---

## Bottom Line

FORGE gives AlphaGalerkin a fast, rich, deterministic environment that stress-tests the core thesis (resolution-independent planning via continuous operators) on problems far more complex than board games -- which is exactly what SBIR reviewers want to see.
