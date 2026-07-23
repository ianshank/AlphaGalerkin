# AlphaGalerkin C4 Architecture (Mermaid Format)

This document provides a comprehensive C4 architecture model for the AlphaGalerkin system using Mermaid diagrams.
The C4 model consists of four levels: System Context, Containers, Components, and Code.

The system supports three primary use cases:

1. **Board Game AI**: Resolution-independent Go and Chess with zero-shot transfer between board sizes
2. **PDE Solving**: MCTS-guided adaptive mesh refinement and Galerkin basis selection
3. **SBIR Commercialization**: Benchmark infrastructure, proposal configs, IP strategy, and competitive positioning for $50B+ simulation market

This architecture reflects v0.3.x including SBIR proposal infrastructure, classical solver baselines, domain geometry abstractions, time-stepping integration, and swarm planning.

---

## Level 1: System Context Diagram

The System Context diagram shows how AlphaGalerkin fits into the broader ecosystem, highlighting the key users and external systems.

```mermaid
C4Context
    title System Context - AlphaGalerkin

    Person(researcher, "Go/PDE Researcher", "ML researcher studying resolution-independent learning, continuous operators, and physics-informed neural networks")
    Person(developer, "Developer", "Implements and experiments with Go AI and PDE solving algorithms")
    Person(player, "Go Player", "Uses the system to play games and analyze positions")
    Person(scientist, "Computational Scientist", "Uses PDE Game Framework for adaptive numerical methods")

    System(alphagalerkin, "AlphaGalerkin", "Resolution-independent AI using Continuous Operator Learning. Supports Go playing with zero-shot transfer AND PDE solving via MCTS-guided basis selection and mesh refinement.")

    System_Ext(go_gui, "Go GUI", "Visual interface for playing and analyzing games (Sabaki, GoGui, Lizzie, KaTrain)")
    System_Ext(go_engine, "Go Rules Engine", "Validates moves and manages game state (gym-go, PettingZoo)")
    System_Ext(compute, "Compute Infrastructure", "GPU clusters for training (CUDA, distributed training)")
    System_Ext(visualization, "Scientific Visualization", "Matplotlib, ParaView for PDE solution visualization")

    Rel(researcher, alphagalerkin, "Runs PoC experiments, validates mathematical claims", "Python CLI")
    Rel(developer, alphagalerkin, "Trains models, implements features", "Python API")
    Rel(player, go_gui, "Plays games, analyzes positions", "GUI")
    Rel(scientist, alphagalerkin, "Solves PDEs with adaptive methods", "Python CLI/API")
    Rel(go_gui, alphagalerkin, "Sends moves, receives evaluations", "GTP Protocol")
    Rel(alphagalerkin, go_engine, "Validates moves, queries legal actions", "Python API")
    Rel(alphagalerkin, compute, "Executes training, performs inference", "PyTorch/CUDA")
    Rel(alphagalerkin, visualization, "Exports solutions for visualization", "NumPy arrays")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Key Interactions

- **Researchers** validate core mathematical claims (zero-shot transfer, O(N) complexity, LBB stability)
- **Developers** train models and implement new features using the Python API
- **Go Players** interact via GTP-compatible GUIs
- **Computational Scientists** use MCTS-guided PDE solving for adaptive basis/mesh refinement
- **External Systems** provide game rules validation, compute resources, and visualization

---

## Level 2: Container Diagram

The Container diagram shows the high-level technical building blocks of AlphaGalerkin.

```mermaid
C4Container
    title Container Diagram - AlphaGalerkin System

    Person(user, "User", "Researcher, Developer, Go Player, or Computational Scientist")

    Container_Boundary(alphagalerkin, "AlphaGalerkin System") {
        Container(cli, "CLI Entrypoints", "Python Scripts", "Command-line interface for training, benchmarking, PDE solving, and experiments")
        Container(gtp_server, "GTP Server", "Python", "Go Text Protocol server for game playing and analysis")

        Container(neural_operator, "Neural Operator Model", "PyTorch", "Core continuous operator learning model with Galerkin attention and FNet mixing")

        Container(mcts_engine, "MCTS Search Engine", "Python", "Monte Carlo Tree Search with neural network guidance for move/action selection")

        Container(ag_solver, "AlphaGalerkin Solver", "Python/PyTorch", "Unified PDE solver wrapping MCTS + PDE games behind BaseSolver. Three evaluator modes: 'random'/'uniform' (RandomEvaluator) and 'trained' (FNetEvaluator over a checkpoint-loaded AlphaGalerkinModel, cached on the solver instance and resolved through the LRU device-resolution helper).")

        Container(pde_framework, "PDE Game Framework", "Python/PyTorch", "Treats PDE solving as sequential decision-making with MCTS-guided basis selection and mesh refinement")

        Container(training_pipeline, "Training Pipeline", "PyTorch", "Self-play, replay buffer, physics-informed loss, adaptive loss balancing, and checkpoint management")

        Container(math_kernel, "Math Kernel", "NumPy/PyTorch", "Mathematical primitives: multi-scale Fourier features, Galerkin projection, spectral filtering")

        Container(poc_framework, "PoC Framework", "Python/Pydantic", "Scenario-based validation of mathematical claims with reproducible experiments")

        Container(data_layer, "Data Layer", "PyTorch Dataset", "Board state preprocessing, variable-size batching, physics data generation")

        Container(research, "Research & Benchmarking", "Python/SciPy", "SBIR baselines (FDM, AMR, PINN), benchmark runner, convergence reports")

        Container(geometry, "Domain Geometry & Time-Stepping", "Python/PyTorch", "Rectangular, L-shaped, cylinder domains; Forward Euler, RK4, Crank-Nicolson")

        Container(swarm, "Swarm Planning", "Python/NumPy", "Multi-agent swarm game with potential field avoidance and coverage optimization")

        Container(sbir_infra, "SBIR Proposal Infrastructure", "Markdown/YAML", "SAM.gov guide, budget templates, submission timeline, program offices, IP strategy, competitive landscape, M&A analysis")

        Container(deployment, "Deployment", "ONNX/PyTorch", "ONNX export, quantization, runtime inference wrapper for edge deployment")

        Container(lm_studio_evaluator, "LM Studio Evaluator (LLM Prior)", "Python/openai-SDK", "Optional MCTS evaluator backed by an OpenAI-compatible local LLM (Qwen-14B in LM Studio by default). Implements src/mcts/evaluator.py::Evaluator structurally with illegal-action masking + temperature softmax. Bounded exponential-backoff retries split SDK errors into retryable (APIConnectionError / APITimeoutError / RateLimitError / InternalServerError) and non-retryable (Authentication / BadRequest / NotFound / etc.). Preflight checks server reachable + model in /v1/models + free-VRAM floor before accepting traffic. Gated behind the [lm-studio] optional extra; SDK imported lazily so the base install is unaffected.")

        Container(codec_zoo, "Codec Model Zoo (Phase 2)", "Python/PyTorch", "Dual-GPU model zoo for the R-D Lagrangian sweep. Pydantic schemas, device planner (VRAM_AWARE), ZooTrainer with fixed-lambda and warm starts, and manifest-level parallel sweep orchestrator. Includes structural stability fixes for FactorizedPrior, GDN, and MS-SSIM.")

        ContainerDb(checkpoint_store, "Model Checkpoints", "File System", "Stores trained model weights and training state")
        ContainerDb(results_store, "Experiment Results", "JSON/YAML", "Stores PoC scenario results and metrics")
    }

    System_Ext(go_gui, "Go GUI", "GTP Client")
    System_Ext(compute, "GPU Cluster", "CUDA Infrastructure")
    System_Ext(lm_studio_server, "LM Studio Server", "OpenAI-compatible local LLM server (Qwen-14B). Runs on the same host or a CUDA-attached host; reachable on http://127.0.0.1:1234/v1 by default.")

    Rel(user, cli, "Runs commands", "CLI")
    Rel(go_gui, gtp_server, "Sends GTP commands", "GTP/TCP")

    Rel(cli, training_pipeline, "Initiates training")
    Rel(cli, poc_framework, "Runs experiments")
    Rel(cli, pde_framework, "Solves PDEs")
    Rel(gtp_server, neural_operator, "Gets policy/value")
    Rel(gtp_server, mcts_engine, "Performs search")

    Rel(mcts_engine, neural_operator, "Evaluates positions")
    Rel(mcts_engine, pde_framework, "Guides basis/mesh selection")
    Rel(ag_solver, mcts_engine, "Drives PDE-game episodes")
    Rel(ag_solver, pde_framework, "Builds basis_selection / mesh_refinement game")
    Rel(ag_solver, neural_operator, "Loads checkpoint via FNetEvaluator (trained mode)")
    Rel(ag_solver, checkpoint_store, "Reads model_state_dict (cached per solver instance)")
    Rel(cli, ag_solver, "Runs PDE benchmarks")
    Rel(pde_framework, neural_operator, "Uses for solution approximation")
    Rel(pde_framework, math_kernel, "Uses PDE operators, residuals")
    Rel(training_pipeline, neural_operator, "Trains weights")
    Rel(training_pipeline, pde_framework, "Generates PDE training data")
    Rel(neural_operator, math_kernel, "Uses basis functions, projections")

    Rel(training_pipeline, data_layer, "Loads training data")
    Rel(poc_framework, neural_operator, "Validates claims")
    Rel(poc_framework, data_layer, "Generates physics data")

    Rel(training_pipeline, checkpoint_store, "Saves/loads models")
    Rel(poc_framework, results_store, "Persists results")

    Rel(neural_operator, compute, "GPU execution")

    Rel(poc_framework, lm_studio_evaluator, "Selects LLM arm in llm_prior_ablation scenario")
    Rel(lm_studio_evaluator, mcts_engine, "Implements MCTS Evaluator protocol")
    Rel(lm_studio_evaluator, lm_studio_server, "JSON-mode chat.completions over HTTP", "OpenAI SDK")
    Rel(lm_studio_evaluator, compute, "Preflight VRAM probe via torch.cuda.mem_get_info")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Container Responsibilities

| Container | Responsibility | Key Technologies |
|-----------|----------------|------------------|
| **CLI Entrypoints** | User interface for training, benchmarking, PDE solving, and experiments | Python, argparse, Hydra |
| **GTP Server** | Game playing interface compatible with Go GUIs | Python, GTP Protocol |
| **Neural Operator Model** | Resolution-independent position evaluation | PyTorch, Galerkin Attention, FNet |
| **MCTS Search Engine** | Tree search with neural guidance for games and PDEs | Python, NumPy, Gumbel AlphaZero |
| **PDE Game Framework** | AlphaZero-style PDE solving via basis/mesh refinement | PyTorch, autodiff, PDE operators |
| **Training Pipeline** | Model training with physics-informed loss and adaptive balancing | PyTorch, ReLoBRaLo, GradNorm |
| **Math Kernel** | Mathematical foundations and operators | NumPy, SciPy, FFT, multi-scale Fourier |
| **PoC Framework** | Validates mathematical claims through experiments | Pydantic, structlog |
| **Data Layer** | Data loading and preprocessing | PyTorch Dataset, padding/masking |
| **Research & Benchmarking** | SBIR baseline comparisons and reports | SciPy, FDM, AMR, PINN, YAML configs |
| **Domain Geometry** | Complex domain abstractions and time integration | Rejection sampling, RK4, Crank-Nicolson |
| **Swarm Planning** | Multi-agent coverage optimization | Potential fields, PettingZoo adapter |
| **SBIR Proposal Infrastructure** | Submission readiness for 5 solicitations | SAM guide, budgets, timeline, IP, competitive analysis |
| **Deployment** | Model export and edge inference | ONNX, quantization, runtime wrapper |

---

## Level 3: Component Diagram - Neural Operator Model

This diagram shows the internal components of the Neural Operator Model container.

```mermaid
C4Component
    title Component Diagram - Neural Operator Model

    Container_Boundary(neural_operator, "Neural Operator Model") {
        Component(model, "AlphaGalerkinModel", "PyTorch nn.Module", "Main model orchestrating all components")

        Component(embedding, "Continuous Embedding", "PyTorch Layer", "Maps discrete board to continuous domain with Fourier features")

        Component(galerkin_attention, "Galerkin Attention", "PyTorch Layer", "O(N) attention via Petrov-Galerkin projection for global influence")

        Component(softmax_attention, "Softmax Attention", "PyTorch Layer", "Traditional attention for local tactical reading")

        Component(fnet_block, "FNet Mixing Block", "PyTorch Layer", "O(N log N) FFT-based mixing for fast rollouts")

        Component(stability_guard, "LBB Stability Guard", "PyTorch Module", "Monitors inf-sup condition during training")

        Component(policy_head, "Policy Head", "PyTorch Layer", "Outputs move probability distribution")

        Component(value_head, "Value Head", "PyTorch Layer", "Outputs position evaluation [-1, 1]")

        Component(adapter, "Resolution Adapter", "Python Module", "Spectral filtering for zero-shot transfer")
    }

    Component_Ext(math_kernel, "Math Kernel", "Basis functions, projections")
    Component_Ext(training_loss, "Training Loss", "Policy CE + Value MSE + LBB regularization")

    Rel(model, embedding, "Embeds input")
    Rel(embedding, galerkin_attention, "Feeds features")
    Rel(galerkin_attention, fnet_block, "Mixes features")
    Rel(fnet_block, softmax_attention, "Refines features")
    Rel(softmax_attention, policy_head, "Generates policy")
    Rel(softmax_attention, value_head, "Generates value")

    Rel(galerkin_attention, stability_guard, "Monitors LBB")
    Rel(model, adapter, "Adapts resolution")

    Rel(embedding, math_kernel, "Uses Fourier basis")
    Rel(galerkin_attention, math_kernel, "Uses Galerkin projection")
    Rel(adapter, math_kernel, "Uses spectral filtering")

    Rel(stability_guard, training_loss, "Adds regularization term")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Component Descriptions

| Component | Responsibility | Mathematical Foundation |
|-----------|----------------|-------------------------|
| **Continuous Embedding** | Maps discrete grid to Fourier features on [0,1]² | Fourier positional encoding |
| **Galerkin Attention** | O(N) global influence modeling | Petrov-Galerkin projection, Monte Carlo integral |
| **Softmax Attention** | Local tactical reading with injectivity | Standard attention mechanism |
| **FNet Mixing** | Fast feature mixing via FFT | Spectral methods, O(N log N) |
| **Stability Guard** | Ensures well-posed learning | LBB inf-sup condition: dim(K) ≥ dim(Q) |
| **Policy Head** | Move distribution prediction | Cross-entropy loss |
| **Value Head** | Position evaluation | MSE loss |
| **Resolution Adapter** | Zero-shot board size transfer | Anti-aliasing, frequency filtering |

---

## Level 3: Component Diagram - Training Pipeline

This diagram shows the internal components of the Training Pipeline container.

```mermaid
C4Component
    title Component Diagram - Training Pipeline

    Container_Boundary(training_pipeline, "Training Pipeline") {
        Component(trainer, "Trainer", "Python Class", "Main training loop orchestration")

        Component(self_play, "Self-Play Engine", "Python Module", "Generates training games using MCTS")

        Component(replay_buffer, "Experience Replay Buffer", "Python Class", "Stores and samples game experiences")

        Component(loss_fn, "AlphaGalerkin Loss", "PyTorch Module", "Combined loss: policy CE + value MSE + LBB regularization")

        Component(checkpoint_mgr, "Checkpoint Manager", "Python Module", "Saves/loads models with rotation")

        Component(evaluator, "Model Evaluator", "Python Module", "Win rate and policy agreement metrics")

        Component(optimizer, "Optimizer", "PyTorch", "Adam optimizer with learning rate scheduling")
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Model being trained")
    Component_Ext(mcts, "MCTS Engine", "Used for self-play")
    Component_Ext(data_layer, "Data Layer", "Batching and collation")
    ComponentDb_Ext(checkpoint_store, "Checkpoint Store", "Model persistence")

    Rel(trainer, self_play, "Generates games")
    Rel(self_play, mcts, "Uses for move selection")
    Rel(self_play, replay_buffer, "Stores experiences")

    Rel(trainer, replay_buffer, "Samples batches")
    Rel(replay_buffer, data_layer, "Uses collation")

    Rel(trainer, neural_operator, "Forward pass")
    Rel(neural_operator, loss_fn, "Computes loss")
    Rel(loss_fn, optimizer, "Backpropagates")
    Rel(optimizer, neural_operator, "Updates weights")

    Rel(trainer, evaluator, "Evaluates periodically")
    Rel(evaluator, neural_operator, "Tests model")

    Rel(trainer, checkpoint_mgr, "Saves checkpoints")
    Rel(checkpoint_mgr, checkpoint_store, "Persists to disk")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Training Pipeline Components

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| **Trainer** | Main training loop with logging | Python class with Hydra config |
| **Self-Play Engine** | Generates training data via MCTS | Parallel game execution |
| **Replay Buffer** | Experience storage and sampling | Uniform and prioritized replay |
| **Loss Function** | Multi-objective optimization | Policy CE + Value MSE + LBB term |
| **Checkpoint Manager** | Model persistence with best tracking | File I/O with rotation policy |
| **Model Evaluator** | Performance metrics | Win rate, policy agreement, engine Elo |
| **Optimizer** | Weight updates | Adam with warmup and decay |

---

## Level 3: Component Diagram — Chess Engine Integration

This diagram shows the chess engine evaluation subsystem added for Stockfish benchmarking.

```mermaid
C4Component
    title Component Diagram - Chess Engine Integration

    Container_Boundary(engine_subsystem, "Engine Integration Subsystem") {
        Component(uci_adapter, "UCI Adapter", "Python Module", "Universal Chess Interface protocol driver for Stockfish/Leela")

        Component(engine_match, "EngineMatch", "Python Class", "Orchestrates N-game matches: color alternation, time control, PGN output")

        Component(elo_calculator, "EloCalculator", "Python Class", "Bayesian Elo estimation with confidence intervals and LOS")

        Component(uci_config, "UCIConfig", "Pydantic Model", "Engine path, depth, nodes, movetime, hash, threads")

        Component(match_config, "MatchConfig", "Pydantic Model", "N games, time control, max moves, opening FEN")
    }

    Component_Ext(evaluator, "Evaluator", "evaluate_vs_engine()")
    Component_Ext(trainer, "Trainer", "_run_engine_evaluation()")
    Component_Ext(wandb, "W&B Logger", "eval/engine/* metrics")
    Component_Ext(chess_game, "ChessGame", "119-channel state, 4672 actions")

    Rel(trainer, evaluator, "Calls at eval_interval")
    Rel(evaluator, engine_match, "Creates match")
    Rel(engine_match, uci_adapter, "Sends UCI commands")
    Rel(engine_match, elo_calculator, "Computes Elo from results")
    Rel(engine_match, chess_game, "Uses for game state")
    Rel(engine_match, uci_config, "Configured by")
    Rel(engine_match, match_config, "Configured by")
    Rel(trainer, wandb, "Logs elo_diff, win_rate, LOS")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Engine Integration Components

| Component | Responsibility | Key Interfaces |
|-----------|----------------|----------------|
| **UCI Adapter** | Stockfish/engine protocol communication | `send_command()`, `parse_bestmove()` |
| **EngineMatch** | Match orchestration with color alternation | `play_match() → MatchResult` |
| **EloCalculator** | Bayesian Elo from W/L/D counts | `estimate_elo_difference() → EloEstimate` |
| **UCIConfig** | Engine binary path and search limits | Pydantic validated, depth/nodes/movetime |
| **MatchConfig** | Match format settings | N games, time control, PGN output |

---

## Level 3: Component Diagram - PoC Framework

This diagram shows the internal components of the Proof-of-Concept Framework container.

```mermaid
C4Component
    title Component Diagram - PoC Framework

    Container_Boundary(poc_framework, "PoC Framework") {
        Component(cli_poc, "PoC CLI", "Python argparse", "Command-line interface: run, list, info, compare")

        Component(registry, "Scenario Registry", "Python Module", "Discovers and manages available scenarios")

        Component(runner, "Scenario Runner", "Python Module", "Executes scenarios with parallel support")

        Component(config_mgr, "Config Manager", "Pydantic", "Validates and loads scenario configurations")

        Component(results_collector, "Results Collector", "Python Module", "Aggregates and persists experiment results")

        Component(logger, "Structured Logger", "structlog", "High-signal logging with context")

        Component(scenario_transfer, "Transfer Scenario", "Python Class", "Validates zero-shot transfer (9x9 → 19x19)")

        Component(scenario_complexity, "Complexity Scenario", "Python Class", "Validates O(N) vs O(N²) complexity")

        Component(scenario_stability, "Stability Scenario", "Python Class", "Monitors LBB condition during training")

        Component(scenario_noyron_hx, "Noyron HX Scenario", "Python Class", "Zero-shot 3D heat-transfer transfer on a Leap 71 helical SDF (analytical or voxel-FDM reference)")

        Component(scenario_noyron_basis, "Noyron Basis Scenario (v2.2)", "Python Class", "MCTS Galerkin basis selection on a Leap 71 helical SDF via pde_basis_helical; manufactured target + monotone-reduction thresholds")
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Model under test")
    Component_Ext(physics_data, "Physics Data Generator", "Synthetic Poisson equation data")
    Component_Ext(picogk_domain, "PicoGKDomain (SDF-backed)", "Helical tube SDF + bisection-projected boundary sampler")
    ComponentDb_Ext(results_store, "Results Store", "JSON/YAML files")

    Rel(cli_poc, registry, "Lists scenarios")
    Rel(cli_poc, runner, "Executes scenarios")
    Rel(cli_poc, config_mgr, "Loads configs")

    Rel(registry, scenario_transfer, "Registers")
    Rel(registry, scenario_complexity, "Registers")
    Rel(registry, scenario_stability, "Registers")
    Rel(registry, scenario_noyron_hx, "Registers")
    Rel(registry, scenario_noyron_basis, "Registers")

    Rel(runner, scenario_transfer, "Runs")
    Rel(runner, scenario_complexity, "Runs")
    Rel(runner, scenario_stability, "Runs")
    Rel(runner, scenario_noyron_hx, "Runs")
    Rel(runner, scenario_noyron_basis, "Runs")

    Rel(scenario_transfer, neural_operator, "Tests transfer")
    Rel(scenario_complexity, neural_operator, "Benchmarks complexity")
    Rel(scenario_stability, neural_operator, "Monitors stability")
    Rel(scenario_noyron_hx, neural_operator, "Tests 3D zero-shot transfer")
    Rel(scenario_noyron_hx, picogk_domain, "Samples interior + boundary")
    Rel(scenario_noyron_basis, picogk_domain, "Samples helical collocation points")

    Rel(scenario_transfer, physics_data, "Generates test data")

    Rel(runner, results_collector, "Collects results")
    Rel(results_collector, results_store, "Persists results")
    Rel(results_collector, logger, "Logs outcomes")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### PoC Framework Components

| Component | Responsibility | Purpose |
|-----------|----------------|---------|
| **PoC CLI** | User interface for running experiments | `run`, `list`, `info`, `compare` commands |
| **Scenario Registry** | Discovery and management of scenarios | Auto-registration, metadata tracking |
| **Scenario Runner** | Parallel execution of experiments | Worker pool, timeout handling |
| **Config Manager** | Configuration validation | Pydantic schemas, YAML/Python configs |
| **Results Collector** | Aggregation and persistence | JSON/YAML output, comparison tools |
| **Transfer Scenario** | Zero-shot transfer validation | Train 9x9 → eval 19x19, MSE < 0.05 |
| **Complexity Scenario** | O(N) complexity verification | Timing benchmarks, scaling analysis |
| **Stability Scenario** | LBB condition monitoring | Singular value tracking, β > 0 check |
| **Noyron HX Scenario** | Zero-shot 3D heat-transfer on Leap 71 helical SDF | Train at 4k pts → eval at 16k pts, MSE < 1e-3, transfer_ratio < 4 |

#### Noyron HX Scenario — Architecture Notes

**Single-device by design at the current model size.** During the headline GPU run (4096 collocation points, `d_model=64`, 3 Galerkin layers, batch_size=1, 200 epochs) per-GPU utilization was observed at 1–10% on a Blackwell-class card. The bottleneck is per-step Adam overhead and Python/CUDA launch latency, not compute. Adding `DistributedDataParallel` would put NCCL all-reduce on the critical path of every step and slow training, not speed it up.

**Revisit DDP only if the surrogate grows materially**: `n_train_pts ≥ 100k`, `d_model ≥ 512`, or `batch_size ≥ 32`. The torchrun infrastructure in [src/distributed/launcher.py](src/distributed/launcher.py) and the GPU-strict resolver in [src/poc/device.py](src/poc/device.py) already exist if/when that threshold is crossed.

**Multi-GPU host strategy in the meantime**: run independent replicas (one per device) via `CUDA_VISIBLE_DEVICES=N`. Each replica trains its own model and produces an independent `ScenarioResult`; useful for seed-variance estimates or for running different `ref_solver_kind` modes side-by-side.

---

## Level 3: Component Diagram - Math Kernel

This diagram shows the mathematical primitives that underpin the system.

```mermaid
C4Component
    title Component Diagram - Math Kernel

    Container_Boundary(math_kernel, "Math Kernel") {
        Component(fourier_basis, "Fourier Basis Functions", "NumPy/PyTorch", "Resolution-independent positional encoding")

        Component(galerkin_projection, "Galerkin Projection", "PyTorch", "Monte Carlo integral approximation for operators")

        Component(spectral_filter, "Spectral Filter", "PyTorch FFT", "Anti-aliasing for resolution transfer")

        Component(lbb_checker, "LBB Condition Checker", "NumPy/PyTorch", "Computes inf-sup constant β")

        Component(fredholm_kernel, "Fredholm Kernel", "Math Functions", "Green's function formulation for influence")

        Component(monte_carlo_integral, "Monte Carlo Integral", "NumPy", "Numerical integration with 1/n normalization")
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Uses math primitives")
    Component_Ext(poc_framework, "PoC Framework", "Validates mathematical properties")

    Rel(neural_operator, fourier_basis, "Encodes positions")
    Rel(neural_operator, galerkin_projection, "Projects values")
    Rel(neural_operator, spectral_filter, "Adapts resolution")
    Rel(neural_operator, lbb_checker, "Monitors stability")

    Rel(galerkin_projection, monte_carlo_integral, "Approximates integral")
    Rel(galerkin_projection, fredholm_kernel, "Uses Green's function")

    Rel(poc_framework, lbb_checker, "Validates β > 0")
    Rel(poc_framework, spectral_filter, "Tests transfer quality")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Math Kernel Components

| Component | Mathematical Foundation | Purpose |
|-----------|------------------------|---------|
| **Fourier Basis** | $\phi_k(x) = e^{2\pi i k \cdot x}$ | Resolution-independent encoding |
| **Galerkin Projection** | $\langle Lu, v \rangle = \langle f, v \rangle$ | O(N) operator approximation |
| **Spectral Filter** | Low-pass filter in frequency domain | Anti-aliasing for transfer |
| **LBB Checker** | $\inf_u \sup_v \frac{\langle Lu, v \rangle}{\|u\| \|v\|} \geq \beta$ | Stability guarantee |
| **Fredholm Kernel** | $u(x) = \int K(x,y) f(y) dy$ | Influence field modeling |
| **Monte Carlo Integral** | $\frac{1}{n} \sum_{i=1}^n f(x_i)$ | Numerical integration |

---

## Level 3: Component Diagram - PDE Game Framework

This diagram shows the internal components of the PDE Game Framework container.

```mermaid
C4Component
    title Component Diagram - PDE Game Framework

    Container_Boundary(pde_framework, "PDE Game Framework") {
        Component(pde_game, "PDEGame", "Abstract Base Class", "Unified interface treating PDE solving as sequential decision-making")

        Component(pde_state, "PDEState", "Dataclass", "Immutable state: coordinates, solution, residuals, error estimate, DoF budget")

        Component(pde_operators, "PDE Operators", "PyTorch Module", "Poisson, Burgers, Heat, Advection-Diffusion with autodiff residuals")

        Component(basis_game, "BasisSelectionGame", "PDEGame Implementation", "Galerkin basis function selection as action space")

        Component(mesh_game, "MeshRefinementGame", "PDEGame Implementation", "Adaptive h/p/hp refinement as action space")

        Component(operator_registry, "Operator Registry", "Singleton", "Thread-safe discovery and registration of PDE operators")

        Component(pde_config, "PDE Configuration", "Pydantic", "PDEConfig, BasisSelectionConfig, MeshRefinementConfig, PDEGameConfig")
    }

    Component_Ext(mcts_engine, "MCTS Engine", "Searches action space")
    Component_Ext(neural_operator, "Neural Operator", "Evaluates states, predicts policy/value")
    Component_Ext(training_loss, "Physics-Informed Loss", "Residual + boundary + conservation losses")

    Rel(pde_game, pde_state, "Manages state transitions")
    Rel(basis_game, pde_game, "Implements")
    Rel(mesh_game, pde_game, "Implements")
    Rel(basis_game, pde_operators, "Computes residuals")
    Rel(mesh_game, pde_operators, "Computes error indicators")

    Rel(mcts_engine, pde_game, "Searches for optimal actions")
    Rel(neural_operator, pde_state, "Encodes state to features")
    Rel(pde_operators, training_loss, "Provides PDE residuals")

    Rel(pde_game, pde_config, "Configured by")
    Rel(pde_operators, operator_registry, "Registered with")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### PDE Game Framework Components

| Component | Responsibility | Key Interface |
|-----------|----------------|---------------|
| **PDEGame** | Abstract base for PDE-as-game formulation | `get_valid_actions()`, `apply_action()`, `get_reward()` |
| **PDEState** | Immutable state representation | `coords`, `solution`, `residuals`, `error_estimate`, `dof` |
| **PDE Operators** | PDE-specific residual and boundary computation | `residual()`, `source_term()`, `boundary_value()` |
| **BasisSelectionGame** | Galerkin basis selection actions | Add/remove/modify Fourier, polynomial, RBF bases |
| **MeshRefinementGame** | Adaptive mesh refinement actions | h-refine, p-refine, coarsen elements |
| **Operator Registry** | Plugin system for PDE types | `register()`, `get()`, `list()` |
| **PDE Configuration** | Validated config schemas | Pydantic models with domain/boundary validation |

---

## Level 3: Component Diagram - Adaptive Loss Balancing

This diagram shows the loss balancing strategies for multi-objective optimization.

```mermaid
C4Component
    title Component Diagram - Adaptive Loss Balancing

    Container_Boundary(loss_balancing, "Adaptive Loss Balancing") {
        Component(loss_balancer, "LossBalancer", "Abstract Base Class", "Interface for adaptive weight computation")

        Component(relobralo, "ReLoBRaLo", "LossBalancer Implementation", "Relative Loss Balancing with Random Lookback for stable training")

        Component(gradnorm, "GradNorm", "LossBalancer Implementation", "Gradient normalization with learnable task weights")

        Component(uncertainty, "UncertaintyWeighting", "LossBalancer Implementation", "Homoscedastic uncertainty with learned log-variance")

        Component(softadapt, "SoftAdapt", "LossBalancer Implementation", "Softmax weighting based on loss improvement rates")

        Component(static, "StaticWeighting", "LossBalancer Implementation", "Fixed weights for baseline comparison")

        Component(loss_terms, "LossTerms", "Dataclass", "Structured loss output with weights and individual terms")

        Component(balancing_config, "BalancingConfig", "Pydantic", "Strategy selection, hyperparameters, weight bounds")
    }

    Component_Ext(physics_loss, "Physics-Informed Loss", "Residual, boundary, conservation losses")
    Component_Ext(alphagalerkin_loss, "AlphaGalerkin Loss", "Policy CE + Value MSE + LBB regularization")

    Rel(relobralo, loss_balancer, "Implements")
    Rel(gradnorm, loss_balancer, "Implements")
    Rel(uncertainty, loss_balancer, "Implements")
    Rel(softadapt, loss_balancer, "Implements")
    Rel(static, loss_balancer, "Implements")

    Rel(loss_balancer, loss_terms, "Produces")
    Rel(loss_balancer, balancing_config, "Configured by")

    Rel(physics_loss, loss_balancer, "Balanced by")
    Rel(alphagalerkin_loss, loss_balancer, "Balanced by")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Loss Balancing Components

| Component | Algorithm | Use Case |
|-----------|-----------|----------|
| **ReLoBRaLo** | Random lookback with softmax normalization | Default for PDE solving, handles scale differences |
| **GradNorm** | Gradient magnitude balancing via shared layer | Multi-task learning with gradient conflicts |
| **UncertaintyWeighting** | Learned log-variance per task | When task uncertainties vary significantly |
| **SoftAdapt** | Improvement rate tracking | When some losses plateau while others improve |
| **StaticWeighting** | Fixed weights | Baseline comparison, known good ratios |

---

## Level 3: Component Diagram - Multi-Scale Fourier Features

This diagram shows the spectral bias mitigation components.

```mermaid
C4Component
    title Component Diagram - Multi-Scale Fourier Features

    Container_Boundary(fourier_features, "Multi-Scale Fourier Features") {
        Component(multiscale, "MultiScaleFourierFeatures", "PyTorch Module", "Parallel frequency banks at multiple scales for capturing fine and coarse features")

        Component(adaptive, "AdaptiveFourierFeatures", "PyTorch Module", "Attention-weighted combination of frequency banks")

        Component(progressive, "ProgressiveFourierFeatures", "PyTorch Module", "Curriculum learning with progressive frequency activation")

        Component(positional, "PositionalEncoding", "PyTorch Module", "Standard sinusoidal encoding for transformer sequences")

        Component(spatial, "SpatialPositionalEncoding", "PyTorch Module", "2D grid encoding for image-like inputs")

        Component(fourier_config, "FourierFeaturesConfig", "Pydantic", "Scale ranges, feature counts, learnable flag")
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Uses for input encoding")
    Component_Ext(pde_operators, "PDE Operators", "Encodes collocation points")

    Rel(multiscale, fourier_config, "Configured by")
    Rel(adaptive, multiscale, "Extends with attention")
    Rel(progressive, multiscale, "Extends with curriculum")

    Rel(neural_operator, multiscale, "Embeds positions")
    Rel(neural_operator, spatial, "Encodes spatial grids")
    Rel(pde_operators, multiscale, "Encodes coordinates")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Fourier Features Components

| Component | Purpose | Mathematical Foundation |
|-----------|---------|-------------------------|
| **MultiScaleFourierFeatures** | Capture both high and low frequencies | $[\sin(2\pi \sigma_k B x), \cos(2\pi \sigma_k B x)]$ for scales $\sigma_k$ |
| **AdaptiveFourierFeatures** | Learn which scales matter for task | Attention-weighted sum of scale-specific features |
| **ProgressiveFourierFeatures** | Avoid spectral bias in early training | Gate high frequencies: $\alpha(t) \cdot \text{high\_freq} + \text{low\_freq}$ |
| **PositionalEncoding** | Standard transformer position embedding | $PE_{pos,2i} = \sin(pos/10000^{2i/d})$ |
| **SpatialPositionalEncoding** | 2D position embedding for grids | Separable encoding: $PE_x \oplus PE_y$ |

---

## Level 3: Component Diagram - Physics-Informed Loss

This diagram shows the physics-informed loss components for PDE solving.

```mermaid
C4Component
    title Component Diagram - Physics-Informed Loss

    Container_Boundary(physics_loss, "Physics-Informed Loss") {
        Component(residual_loss, "ResidualLoss", "PyTorch Module", "PDE residual minimization via collocation")

        Component(boundary_loss, "BoundaryLoss", "PyTorch Module", "Dirichlet, Neumann, Robin boundary enforcement")

        Component(ic_loss, "InitialConditionLoss", "PyTorch Module", "Time-dependent PDE initial state matching")

        Component(conservation_loss, "ConservationLoss", "PyTorch Module", "Global conservation law enforcement (mass, energy)")

        Component(pinn_loss, "PhysicsInformedLoss", "PyTorch Module", "Combined loss with adaptive balancing")

        Component(combined_loss, "CombinedAlphaGalerkinPhysicsLoss", "PyTorch Module", "Policy + Value + Physics for MCTS-guided PDE")
    }

    Component_Ext(pde_operators, "PDE Operators", "Computes residuals via autodiff")
    Component_Ext(loss_balancer, "Loss Balancer", "Weights physics terms")
    Component_Ext(neural_operator, "Neural Operator", "Predicts solution field")

    Rel(residual_loss, pde_operators, "Uses for residual computation")
    Rel(boundary_loss, pde_operators, "Uses for boundary values")
    Rel(ic_loss, pde_operators, "Uses for initial conditions")

    Rel(pinn_loss, residual_loss, "Combines")
    Rel(pinn_loss, boundary_loss, "Combines")
    Rel(pinn_loss, ic_loss, "Combines")
    Rel(pinn_loss, conservation_loss, "Combines")
    Rel(pinn_loss, loss_balancer, "Balanced by")

    Rel(combined_loss, pinn_loss, "Includes physics")
    Rel(neural_operator, pinn_loss, "Trained with")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Physics-Informed Loss Components

| Component | Loss Term | Mathematical Formulation |
|-----------|-----------|--------------------------|
| **ResidualLoss** | PDE residual | $\mathcal{L}_r = \frac{1}{N_r} \sum_i \|Lu(x_i) - f(x_i)\|^2$ |
| **BoundaryLoss** | Boundary conditions | $\mathcal{L}_b = \frac{1}{N_b} \sum_i \|u(x_i) - g(x_i)\|^2$ |
| **InitialConditionLoss** | Initial state | $\mathcal{L}_0 = \frac{1}{N_0} \sum_i \|u(x_i, 0) - u_0(x_i)\|^2$ |
| **ConservationLoss** | Global conservation | $\mathcal{L}_c = \|\int_\Omega u \, dx - C_0\|^2$ |
| **PhysicsInformedLoss** | Combined PINN loss | $\mathcal{L} = \lambda_r \mathcal{L}_r + \lambda_b \mathcal{L}_b + \lambda_0 \mathcal{L}_0 + \lambda_c \mathcal{L}_c$ |

---

## Level 4: Code Diagram - Galerkin Attention

This diagram shows the implementation details of the Galerkin Attention component.

```mermaid
classDiagram
    class GalerkinLinearAttention {
        +int d_model
        +int n_heads
        +int d_head
        +Linear query_proj
        +Linear key_proj
        +Linear value_proj
        +Linear output_proj
        +forward(x: Tensor) Tensor
        +_compute_context(K: Tensor, V: Tensor) Tensor
        +_check_lbb_stability(K: Tensor, Q: Tensor) float
    }

    class MultiHeadAttention {
        <<interface>>
        +forward(x: Tensor) Tensor
    }

    class FourierEncoding {
        +int n_features
        +Tensor freqs
        +encode(positions: Tensor) Tensor
    }

    class MonteCarloIntegral {
        +approximate(K: Tensor, V: Tensor) Tensor
        +_normalize(result: Tensor, n: int) Tensor
    }

    class LBBStabilityGuard {
        +float beta_threshold
        +check_stability(K: Tensor, Q: Tensor) bool
        +compute_singular_values(M: Tensor) Tensor
    }

    class GalerkinAttentionBlock {
        +GalerkinLinearAttention attention
        +LayerNorm norm1
        +MLP ffn
        +LayerNorm norm2
        +forward(x: Tensor) Tensor
    }

    MultiHeadAttention <|-- GalerkinLinearAttention : implements
    GalerkinLinearAttention --> MonteCarloIntegral : uses
    GalerkinLinearAttention --> LBBStabilityGuard : uses
    GalerkinLinearAttention --> FourierEncoding : uses
    GalerkinAttentionBlock *-- GalerkinLinearAttention : contains
```

### Key Implementation Details

**Galerkin Attention Algorithm:**

```python
# Step 1: Project to Query, Key, Value spaces
Q = query_proj(x)    # (batch, n, d_head)
K = key_proj(x)      # (batch, n, d_head)
V = value_proj(x)    # (batch, n, d_head)

# Step 2: Monte Carlo integral approximation
# Context = K^T V / n  (not K^T V / sqrt(d))
Context = einsum('bnd,bnm->bdm', K, V) / n

# Step 3: Reconstruct in Query basis
Output = einsum('bnd,bdm->bnm', Q, Context)

# Step 4: LBB stability check (training only)
if training:
    beta = compute_inf_sup_constant(K, Q)
    assert beta > beta_threshold
```

**Complexity Analysis:**

- Standard Attention: O(N² × d)
- Galerkin Attention: O(N × d²)
- For typical Go: N=361, d=32 → **10x speedup**

---

## Level 4: Code Diagram - PDE Game Framework

This diagram shows the implementation details of the PDE Game Framework.

```mermaid
classDiagram
    class PDEGame {
        <<abstract>>
        +PDEGameConfig config
        +PDEOperator operator
        +get_initial_state() PDEState
        +get_valid_actions(state: PDEState) list~int~
        +apply_action(state: PDEState, action: int) PDEState
        +get_reward(state: PDEState, prev_state: PDEState) float
        +is_terminal(state: PDEState) bool
        +to_tensor(state: PDEState) Tensor
    }

    class PDEState {
        +ndarray coords
        +ndarray solution
        +ndarray residuals
        +float error_estimate
        +int dof
        +int step
        +float budget_remaining
        +list~int~ history
    }

    class PDEOperator {
        <<abstract>>
        +str name
        +bool is_time_dependent
        +bool is_linear
        +int order
        +residual(u: Tensor, coords: Tensor) PDEResidual
        +source_term(coords: Tensor) Tensor
        +boundary_value(coords: Tensor) Tensor
        +compute_derivatives(u: Tensor, coords: Tensor) dict
    }

    class PDEResidual {
        +ndarray values
        +float l2_norm
        +float max_norm
        +dict derivatives
        +to_numpy() PDEResidual
    }

    class BasisSelectionGame {
        +list~BasisFunction~ active_bases
        +int n_candidate_bases
        +get_valid_actions(state) list~int~
        +apply_action(state, action) PDEState
        -_add_basis(state, basis_idx) PDEState
        -_remove_basis(state, basis_idx) PDEState
        -_solve_galerkin(state) ndarray
    }

    class MeshRefinementGame {
        +Mesh mesh
        +RefinementStrategy strategy
        +get_valid_actions(state) list~int~
        +apply_action(state, action) PDEState
        -_h_refine(element_idx) Mesh
        -_p_refine(element_idx) Mesh
        -_coarsen(element_idx) Mesh
    }

    class PoissonOperator {
        +residual(u, coords) PDEResidual
        +source_term(coords) Tensor
        -_laplacian(u, coords) Tensor
    }

    class BurgersOperator {
        +float viscosity
        +residual(u, coords) PDEResidual
        -_nonlinear_term(u, coords) Tensor
    }

    PDEGame <|-- BasisSelectionGame : implements
    PDEGame <|-- MeshRefinementGame : implements
    PDEGame --> PDEState : manages
    PDEGame --> PDEOperator : uses
    PDEOperator <|-- PoissonOperator : implements
    PDEOperator <|-- BurgersOperator : implements
    PDEOperator --> PDEResidual : produces
```

---

## Level 4: Code Diagram - Loss Balancing

This diagram shows the implementation of adaptive loss balancing.

```mermaid
classDiagram
    class LossBalancer {
        <<abstract>>
        +LossBalancingConfig config
        +list~str~ loss_names
        +dict~str,float~ weights
        +update(losses: dict) dict~str,float~
        +compute_weighted_loss(losses: dict) LossTerms
        +reset() void
    }

    class LossTerms {
        +dict~str,Tensor~ losses
        +dict~str,float~ weights
        +Tensor weighted_sum
        +to_dict() dict
    }

    class ReLoBRaLo {
        +float beta
        +float tau
        +int warmup_steps
        +dict _running_losses
        +dict _loss_history
        +update(losses) dict
        -_compute_relative_losses() dict
        -_apply_random_lookback() dict
    }

    class GradNorm {
        +float alpha
        +dict _log_weights
        +dict _initial_losses
        +compute_gradnorm_loss(losses, shared_layer) Tensor
        -_compute_grad_norms(losses, shared_layer) dict
    }

    class UncertaintyWeighting {
        +dict _log_vars
        +compute_regularized_loss(losses) Tensor
        -_precision_from_log_var(log_var) float
    }

    class SoftAdapt {
        +int window_size
        +list _history
        +update(losses) dict
        -_compute_improvement_rates() dict
    }

    class StaticWeighting {
        +update(losses) dict
    }

    LossBalancer <|-- ReLoBRaLo : implements
    LossBalancer <|-- GradNorm : implements
    LossBalancer <|-- UncertaintyWeighting : implements
    LossBalancer <|-- SoftAdapt : implements
    LossBalancer <|-- StaticWeighting : implements
    LossBalancer --> LossTerms : produces
```

### ReLoBRaLo Algorithm

```python
# Relative Loss Balancing with Random Lookback
def update(self, losses: dict) -> dict:
    # Step 1: Update running averages
    for name, loss in losses.items():
        self._running_losses[name] = (
            self.beta * self._running_losses.get(name, loss.item())
            + (1 - self.beta) * loss.item()
        )
        self._loss_history[name].append(loss.item())

    # Step 2: Random lookback (sample historical point)
    lookback = random.randint(0, len(self._loss_history[name]) - 1)

    # Step 3: Compute relative improvements
    for name in self.loss_names:
        current = self._running_losses[name]
        historical = self._loss_history[name][lookback]
        relative[name] = current / (historical + eps)

    # Step 4: Softmax normalization with temperature
    weights = softmax([relative[n] / self.tau for n in self.loss_names])

    return dict(zip(self.loss_names, weights))
```

---

## Data Flow Diagrams

### Training Data Flow

```mermaid
flowchart TB
    subgraph SelfPlay["Self-Play Generation"]
        SP1[Start Position] --> SP2[MCTS Search]
        SP2 --> SP3[Select Move]
        SP3 --> SP4[Execute Move]
        SP4 --> SP5{Game Over?}
        SP5 -->|No| SP2
        SP5 -->|Yes| SP6[Record Game]
    end

    subgraph ReplayBuffer["Replay Buffer"]
        RB1[Store Experience]
        RB2[Sample Batch]
        RB3[Priority Update]
    end

    subgraph Training["Training Loop"]
        T1[Forward Pass]
        T2[Compute Loss]
        T3[Backpropagation]
        T4[Update Weights]
    end

    SP6 --> RB1
    RB1 --> RB2
    RB2 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> RB3
    RB3 --> RB1
    T4 --> SP1

    style SelfPlay fill:#e1f5ff
    style ReplayBuffer fill:#fff5e1
    style Training fill:#ffe1f5
```

### Inference Data Flow

```mermaid
flowchart LR
    Input[Board State<br/>17×H×W] --> Embed[Continuous<br/>Embedding]
    Embed --> Galerkin[Galerkin<br/>Attention<br/>6 layers]
    Galerkin --> FNet[FNet<br/>Mixing]
    FNet --> Softmax[Softmax<br/>Attention<br/>2 layers]

    Softmax --> Policy[Policy Head<br/>361+1 moves]
    Softmax --> Value[Value Head<br/>[-1, 1]]

    Policy --> MCTS[MCTS<br/>Search]
    Value --> MCTS

    MCTS --> Move[Best Move]

    style Input fill:#e1f5ff
    style Policy fill:#ffe1e1
    style Value fill:#e1ffe1
    style Move fill:#ffe1f5
```

### Resolution Transfer Flow

```mermaid
flowchart TB
    subgraph Train["Training on 9×9"]
        T1[9×9 Board] --> T2[Fourier Encoding]
        T2 --> T3[Galerkin Model]
        T3 --> T4[Train Weights]
    end

    subgraph Adapt["Resolution Adaptation"]
        A1[Save Frequency<br/>Representations]
        A2[Apply Spectral<br/>Filter]
        A3[Adjust Monte Carlo<br/>Normalization]
    end

    subgraph Infer["Inference on 19×19"]
        I1[19×19 Board] --> I2[Fourier Encoding]
        I2 --> I3[Galerkin Model<br/>Same Weights]
        I3 --> I4[Policy/Value]
    end

    T4 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> I3

    style Train fill:#e1f5ff
    style Adapt fill:#ffe1e1
    style Infer fill:#e1ffe1
```

### PDE Solving Data Flow (MCTS-Guided)

```mermaid
flowchart TB
    subgraph Init["Initialization"]
        P1[PDE Configuration] --> P2[Create PDEGame]
        P2 --> P3[Initial Mesh/Basis]
        P3 --> P4[Compute Initial<br/>Residual]
    end

    subgraph MCTS["MCTS Search"]
        M1[Encode State] --> M2[Neural Network<br/>Policy + Value]
        M2 --> M3[MCTS Simulation]
        M3 --> M4[Select Best Action]
    end

    subgraph Action["Action Execution"]
        A1{Action Type}
        A1 -->|Add Basis| A2[Expand Basis Set]
        A1 -->|Refine Mesh| A3[h/p Refinement]
        A1 -->|Adjust| A4[Modify Parameters]
        A2 --> A5[Solve Galerkin<br/>System]
        A3 --> A5
        A4 --> A5
        A5 --> A6[Update Solution]
    end

    subgraph Evaluate["Evaluation"]
        E1[Compute PDE<br/>Residual] --> E2[Estimate Error]
        E2 --> E3[Calculate Reward]
        E3 --> E4{Converged?}
        E4 -->|No| E5[Continue]
        E4 -->|Yes| E6[Return Solution]
    end

    P4 --> M1
    M4 --> A1
    A6 --> E1
    E5 --> M1

    style Init fill:#e1f5ff
    style MCTS fill:#ffe1f5
    style Action fill:#fff5e1
    style Evaluate fill:#e1ffe1
```

### Physics-Informed Training Flow

```mermaid
flowchart LR
    subgraph Sampling["Collocation Sampling"]
        S1[Domain Points] --> S2[Boundary Points]
        S2 --> S3[Initial Points<br/>if time-dependent]
    end

    subgraph Forward["Forward Pass"]
        F1[Multi-Scale<br/>Fourier Features] --> F2[Neural Operator]
        F2 --> F3[Solution Field u]
    end

    subgraph Loss["Loss Computation"]
        L1[PDE Residual<br/>Lu - f] --> L4[ReLoBRaLo<br/>Balancing]
        L2[Boundary Loss<br/>u - g] --> L4
        L3[IC Loss<br/>u₀ match] --> L4
        L4 --> L5[Weighted Sum]
    end

    subgraph Update["Weight Update"]
        U1[Backprop] --> U2[Optimizer Step]
        U2 --> U3[Update Balancer<br/>Weights]
    end

    S1 --> F1
    S2 --> F1
    S3 --> F1
    F3 --> L1
    F3 --> L2
    F3 --> L3
    L5 --> U1
    U3 --> L4

    style Sampling fill:#e1f5ff
    style Forward fill:#ffe1e1
    style Loss fill:#fff5e1
    style Update fill:#e1ffe1
```

---

## Deployment Diagram

```mermaid
C4Deployment
    title Deployment Diagram - AlphaGalerkin

    Deployment_Node(dev_machine, "Developer Machine", "Linux/Mac/Windows") {
        Container(cli_dev, "CLI Tools", "Python 3.10+")
        Container(jupyter, "Jupyter Notebooks", "Interactive Analysis")
    }

    Deployment_Node(training_cluster, "Training Cluster", "GPU Cluster") {
        Deployment_Node(gpu_node1, "GPU Node 1", "8× A100 GPUs") {
            Container(trainer1, "Trainer Instance 1", "PyTorch Distributed")
        }
        Deployment_Node(gpu_node2, "GPU Node 2", "8× A100 GPUs") {
            Container(trainer2, "Trainer Instance 2", "PyTorch Distributed")
        }
        ContainerDb(shared_storage, "Shared Storage", "NFS/S3", "Model checkpoints, experiment results")
    }

    Deployment_Node(inference_server, "Inference Server", "Cloud/Edge Device") {
        Container(gtp_server_deploy, "GTP Server", "Python FastAPI")
        Container(model_inference, "Model Inference", "PyTorch + ONNX")
    }

    Deployment_Node(client_machine, "Client Machine", "Desktop/Laptop") {
        Container(go_gui_deploy, "Go GUI", "Sabaki/GoGui")
    }

    Rel(cli_dev, trainer1, "Submits training jobs", "SSH/SLURM")
    Rel(trainer1, shared_storage, "Saves checkpoints")
    Rel(trainer2, shared_storage, "Saves checkpoints")
    Rel(trainer1, trainer2, "Synchronizes gradients", "NCCL/Gloo")

    Rel(gtp_server_deploy, model_inference, "Gets predictions")
    Rel(model_inference, shared_storage, "Loads models")
    Rel(go_gui_deploy, gtp_server_deploy, "Sends GTP commands", "TCP")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Deep Learning** | PyTorch 2.0+ | Neural network implementation |
| **Numerical Computing** | NumPy, SciPy | Mathematical operations |
| **Configuration** | Hydra, Pydantic | Configuration management and validation |
| **Testing** | pytest, hypothesis | Unit and property-based testing |
| **Logging** | structlog | Structured logging |
| **Type Checking** | mypy, jaxtyping | Static type analysis |
| **Code Quality** | ruff | Linting and formatting |

### Key Libraries

```python
# Core dependencies
torch >= 2.0.0          # Deep learning framework
einops >= 0.7.0         # Tensor operations
jaxtyping >= 0.2.25     # Type annotations for arrays
pydantic >= 2.0.0       # Data validation
hydra-core >= 1.3.0     # Configuration management
structlog >= 23.0.0     # Structured logging
numpy >= 1.24.0         # Numerical computing
```

---

## Architecture Principles

### 1. Resolution Independence

- **Continuous Domain**: Treat board as Ω = [0,1]² rather than discrete grid
- **Fourier Encoding**: Position-independent frequency representation
- **Spectral Methods**: Proper anti-aliasing and frequency filtering

### 2. Mathematical Rigor

- **Galerkin Projection**: Well-founded operator approximation theory
- **LBB Stability**: Monitored inf-sup condition ensures convergence
- **Fredholm Operators**: Integral equation formulation for influence

### 3. Performance Optimization

- **O(N) Attention**: Linear complexity via Petrov-Galerkin projection
- **FFT Mixing**: O(N log N) spectral mixing for fast rollouts
- **CUDA Acceleration**: Full GPU utilization for training and inference

### 4. Testability

- **Property-Based Tests**: Mathematical properties verified with Hypothesis
- **PoC Framework**: Reproducible validation of core claims
- **Modular Design**: Independent testing of components

### 5. Configurability

- **Hydra Integration**: Hierarchical configuration management
- **Pydantic Schemas**: Runtime validation of parameters
- **Environment Variables**: Deployment-specific overrides

---

## Key Architectural Decisions

### Decision 1: Galerkin vs Standard Attention

- **Context**: Need O(N) complexity for large board sizes
- **Decision**: Use Petrov-Galerkin projection instead of softmax
- **Rationale**: Reduces complexity from O(N²d) to O(Nd²)
- **Trade-offs**: Requires careful normalization (1/n, not 1/√d)

### Decision 2: Hybrid Architecture (Galerkin + Softmax)

- **Context**: Balance global strategy and local tactics
- **Decision**: Galerkin layers for strategy, softmax for tactics
- **Rationale**: Galerkin captures long-range influence, softmax preserves injectivity for life/death
- **Trade-offs**: More complex than uniform architecture

### Decision 3: FNet for Fast Rollouts

- **Context**: MCTS requires thousands of neural evaluations
- **Decision**: FFT-based mixing as alternative to attention
- **Rationale**: 5× speedup for leaf evaluation
- **Trade-offs**: Slightly lower accuracy vs full attention

### Decision 4: PoC Framework for Validation

- **Context**: Need reproducible validation of mathematical claims
- **Decision**: Config-driven scenario framework
- **Rationale**: Ensures claims are testable and reproducible
- **Trade-offs**: Additional infrastructure complexity

### Decision 5: Pydantic for Configuration

- **Context**: Complex hyperparameter space with mathematical constraints
- **Decision**: Pydantic schemas with validators
- **Rationale**: Runtime validation, type safety, IDE support
- **Trade-offs**: More verbose than plain dicts

### Decision 6: PDE Solving as Sequential Decision-Making

- **Context**: Adaptive basis selection and mesh refinement require intelligent choices
- **Decision**: Model PDE solving as a game with MCTS search
- **Rationale**: Leverages AlphaZero infrastructure, learns optimal refinement strategies
- **Trade-offs**: Training overhead, requires careful reward design

### Decision 7: ReLoBRaLo for Multi-Objective Loss Balancing

- **Context**: Physics-informed losses have vastly different scales (residual vs boundary)
- **Decision**: Use Relative Loss Balancing with Random Lookback
- **Rationale**: Stable training, handles scale differences, minimal hyperparameters
- **Trade-offs**: Randomness in lookback, warmup period needed

### Decision 8: Multi-Scale Fourier Features for Spectral Bias

- **Context**: Neural networks learn low frequencies first (spectral bias)
- **Decision**: Parallel Fourier feature banks at multiple scales
- **Rationale**: Captures both fine and coarse solution features from start
- **Trade-offs**: Increased feature dimension, more parameters

### Decision 9: Autodiff for PDE Residuals

- **Context**: Need derivatives for PDE residual computation
- **Decision**: Use PyTorch autograd for all derivative computations
- **Rationale**: Exact gradients, GPU-accelerated, composable with neural networks
- **Trade-offs**: Memory overhead for computation graph, requires careful batching

---

## Future Architecture Enhancements

### Implemented (v2.0)

1. **PDE Game Framework** ✓
   - PDEGame abstraction for basis selection and mesh refinement
   - PDE operators (Poisson, Burgers, Heat, Advection-Diffusion)
   - MCTS-guided adaptive solving

2. **Physics-Informed Training** ✓
   - Multi-objective loss with residual, boundary, conservation terms
   - ReLoBRaLo, GradNorm, Uncertainty weighting
   - Combined AlphaGalerkin + Physics loss

3. **Multi-Scale Fourier Features** ✓
   - Spectral bias mitigation
   - Adaptive and progressive variants
   - Spatial positional encoding for 2D grids

### Planned Improvements

1. **Distributed Training**
   - Multi-node self-play generation
   - Gradient aggregation via NCCL
   - Model zoo for curriculum learning

2. **ONNX Export**
   - Convert PyTorch models to ONNX
   - Deploy on edge devices (Raspberry Pi, Jetson)
   - Quantization for int8 inference

3. **Multi-Game Support**
   - Abstract game interface
   - Support for Chess, Shogi, etc.
   - Shared continuous operator core

4. **Advanced MCTS**
   - Gumbel AlphaZero search
   - Value-based exploration
   - Policy improvement operators

5. **Enhanced PoC Framework**
   - Automated hyperparameter tuning
   - Statistical significance testing
   - Comparative visualizations

6. **PDE Extensions**
   - 3D domain support
   - Multi-physics coupling
   - Uncertainty quantification

---

## Level 3: Component Diagram - SBIR Research Infrastructure

This diagram shows the benchmarking and baseline comparison components for SBIR proposals.

```mermaid
C4Component
    title Component Diagram - SBIR Research Infrastructure

    Container_Boundary(research, "Research & Benchmarking") {
        Component(benchmark_runner, "PDEBenchmarkRunner", "Python Class", "Runs AlphaGalerkin vs baselines on benchmark suites; heavy=True opt-in extends refinement_levels with heavy_refinement_levels (e.g. 65 536-DOF Poisson for the P40 24 GiB advantage)")

        Component(fdm_solver, "UniformFDMSolver", "BaseSolver", "2nd-order finite differences on uniform grid via scipy.sparse")

        Component(amr_solver, "DorflerAMRSolver", "BaseSolver", "Dorfler bulk-chasing adaptive mesh refinement on 1D/2D grids; raised defaults (theta=0.5, max_refinements=30, max_initial_points_1d=256, divisor=2) escape the 18-DOF Burgers ceiling")

        Component(ns_fdm_solver, "NavierStokesFDMSolver", "BaseSolver", "Chorin projection FDM for incompressible 2D NS; consumes corrected Taylor-Green numpy reference")

        Component(pinn_solver, "SimplePINNSolver", "BaseSolver", "Physics-Informed Neural Network baseline; honours PINNConfig.device (auto/cpu/cuda/cuda:N), auto-detects vector PDEs (NS=>2-channel network), embeds GpuUtilizationProfiler around the training loop")

        Component(gpu_profiler, "GpuUtilizationProfiler", "Context Manager", "Wraps `nvidia-smi dmon` as a subprocess; returns mean SM-util %, mean memory-util %, peak FB-MiB; no-ops cleanly when nvidia-smi is missing")

        Component(device_helper, "resolve_device", "Helper", "Resolves auto/cpu/cuda/cuda:N preference strings to torch.device with fail-loud bounds checking")

        Component(solver_registry, "SOLVER_REGISTRY", "Dict Registry", "Maps solver names to classes: get_solver(), list_solvers(); pinn_p40 / pinn_cpu rows registered by run_sbir_p40.py from YAML profiles")

        Component(benchmark_config, "Benchmark Configs", "YAML", "sbir_suite.yaml (with heavy_refinement_levels), sbir_p40.yaml (PINN profiles for p40/cpu rows), navy_n252_088.yaml, doe_ascr_c59.yaml, nsf_sbir.yaml")

        Component(p40_driver, "scripts/run_sbir_p40.py", "argparse CLI", "Config-driven driver: loads sbir_p40.yaml, applies CLI overrides (--device, --n-epochs, --n-collocation, --refinement-levels, --skip-cpu), registers PINN profiles via _make_pinn_class")
    }

    Component_Ext(pde_operators, "PDE Operators", "Provides exact solutions, residuals; numpy/torch parity guarded by tests/pde/test_taylor_green_invariants.py")
    Component_Ext(alphagalerkin, "AlphaGalerkin Engine", "MCTS-guided solver under comparison")
    Component_Ext(nvidia_smi, "nvidia-smi dmon", "External GPU telemetry binary (optional)")

    Rel(benchmark_runner, fdm_solver, "Runs baseline")
    Rel(benchmark_runner, amr_solver, "Runs baseline")
    Rel(benchmark_runner, ns_fdm_solver, "Runs baseline")
    Rel(benchmark_runner, pinn_solver, "Runs baseline")
    Rel(benchmark_runner, solver_registry, "Discovers solvers")
    Rel(benchmark_runner, benchmark_config, "Loads config")

    Rel(fdm_solver, pde_operators, "Uses source_term, boundary_value")
    Rel(amr_solver, pde_operators, "Uses source_term, exact_solution")
    Rel(ns_fdm_solver, pde_operators, "Uses initial_condition, exact_solution (numpy)")
    Rel(pinn_solver, pde_operators, "Uses residual via autograd")
    Rel(pinn_solver, device_helper, "Resolves device preference")
    Rel(pinn_solver, gpu_profiler, "Wraps training loop")

    Rel(p40_driver, benchmark_runner, "Drives")
    Rel(p40_driver, solver_registry, "Registers pinn_p40, pinn_cpu")
    Rel(p40_driver, benchmark_config, "Loads sbir_p40.yaml")

    Rel(gpu_profiler, nvidia_smi, "Spawns + parses dmon output")

    Rel(benchmark_runner, alphagalerkin, "Compares against baselines")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### SBIR Research Components

| Component | Responsibility | Key Interface |
|-----------|----------------|---------------|
| **PDEBenchmarkRunner** | Load YAML config, run all solver x problem combos, compute convergence rates; `--heavy` opt-in extends `refinement_levels` with `heavy_refinement_levels` | `run_all()`, `generate_report()`, `__init__(..., heavy=False)` |
| **UniformFDMSolver** | Reference FDM solution on 1D/2D uniform grids | `solve(operator, n_dof)` |
| **DorflerAMRSolver** | Adaptive refinement with residual error indicators; target-aware n_start | `solve(operator, n_dof)` |
| **NavierStokesFDMSolver** | Chorin-projection FDM for 2D incompressible NS; Taylor-Green initial condition + L2 metric corrected by the numpy/torch parity fix | `solve(operator, n_dof)` |
| **SimplePINNSolver** | PINN baseline with configurable MLP + autograd; honours `PINNConfig.device`, auto-detects vector PDEs, embeds GPU-utilisation telemetry | `solve(operator, n_dof)` |
| **GpuUtilizationProfiler** | Wraps `nvidia-smi dmon` as a context manager; returns mean SM-util %, mean memory-util %, peak FB-MiB | `__enter__()` / `__exit__()` |
| **resolve_device** | Resolves `auto`/`cpu`/`cuda`/`cuda:N` strings to `torch.device` with fail-loud bounds checking | `resolve_device(preference, *, context)` |
| **SOLVER_REGISTRY** | Plugin system for baseline solvers; populated by `run_sbir_p40.py` with `pinn_p40` / `pinn_cpu` rows | `get_solver(name)`, `list_solvers()` |
| **scripts/run_sbir_p40.py** | Config-driven CLI driver: argparse + YAML profile loader, applies CLI overrides, registers PINN profiles | `parse_args`, `load_config`, `apply_overrides`, `register_pinn_profiles` |

---

## Level 3: Component Diagram - Domain Geometry & Time-Stepping

This diagram shows the domain geometry and time-integration components.

```mermaid
C4Component
    title Component Diagram - Domain Geometry & Time-Stepping

    Container_Boundary(geometry_time, "Domain Geometry & Time-Stepping") {
        Component(domain_abc, "DomainGeometry", "Abstract Base Class", "contains_point, is_boundary, sample_interior, sample_boundary, bounding_box")

        Component(rect_domain, "RectangularDomain", "DomainGeometry", "Standard [x_min, x_max] x [y_min, y_max] domain")

        Component(lshaped_domain, "LShapedDomain", "DomainGeometry", "[-1,1]^2 minus [0,1]x[-1,0], reentrant corner singularity")

        Component(cylinder_domain, "CylinderFlowDomain", "DomainGeometry", "DFG benchmark: rectangle with circular obstacle, rejection sampling")

        Component(geom_config, "GeometryConfig", "Pydantic", "geometry_type enum, scale, cylinder params, create_geometry() factory")

        Component(time_euler, "ForwardEuler", "TimeStepper", "1st-order explicit Euler method")

        Component(time_rk4, "RK4", "TimeStepper", "4th-order Runge-Kutta with 4 stages")

        Component(time_cn, "CrankNicolson", "TimeStepper", "2nd-order implicit with fixed-point iteration")

        Component(time_config, "TimeSteppingConfig", "Pydantic", "method enum, dt, t_start/t_end, adaptive_dt, create_time_stepper() factory")
    }

    Component_Ext(pde_operators, "PDE Operators", "NavierStokes, Burgers, Heat (time-dependent)")
    Component_Ext(mesh_game, "MeshRefinementGame", "Uses geometry for adaptive refinement")

    Rel(rect_domain, domain_abc, "Implements")
    Rel(lshaped_domain, domain_abc, "Implements")
    Rel(cylinder_domain, domain_abc, "Implements")

    Rel(time_euler, time_config, "Created by factory")
    Rel(time_rk4, time_config, "Created by factory")
    Rel(time_cn, time_config, "Created by factory")

    Rel(pde_operators, time_rk4, "Integrates in time")
    Rel(mesh_game, domain_abc, "Samples collocation points")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Domain & Time-Stepping Components

| Component | Responsibility | Key Feature |
|-----------|----------------|-------------|
| **RectangularDomain** | Standard box domains | Proportional boundary sampling per edge |
| **LShapedDomain** | AMR benchmark domain | Rejection sampling (75% acceptance), 6-segment boundary |
| **CylinderFlowDomain** | DFG CFD benchmark | Cylinder exclusion zone, combined wall + surface sampling |
| **ForwardEuler** | Explicit 1st-order | Simple, O(dt) error |
| **RK4** | Explicit 4th-order | O(dt^4) error, 4 function evaluations per step |
| **CrankNicolson** | Implicit 2nd-order | A-stable, fixed-point iteration with convergence warning |

---

## Level 3: Component Diagram - Swarm Planning Game

This diagram shows the multi-agent swarm planning game for DARPA/AFWERX positioning.

```mermaid
C4Component
    title Component Diagram - Swarm Planning Game

    Container_Boundary(swarm, "Swarm Planning (S500)") {
        Component(swarm_game, "SwarmPlanningGame", "PDEGame", "Multi-agent swarm control as sequential decision-making with MCTS")

        Component(swarm_state, "SwarmState", "Dataclass", "Agent positions, velocities, coverage map, communication graph, obstacles")

        Component(swarm_config, "SwarmPlanningConfig", "Pydantic", "n_agents, communication_range, collision_radius, domain_size, reward weights")

        Component(potential_field, "Potential Field", "Algorithm", "Inverse-distance obstacle avoidance via Laplace equation connection")

        Component(coverage_map, "Coverage Map", "NumPy Array", "Binary grid tracking explored regions, sensor radius coverage")

        Component(comm_graph, "Communication Graph", "Adjacency Matrix", "Agent connectivity based on communication_range threshold")
    }

    Component_Ext(mcts_engine, "MCTS Engine", "Searches 7-action space per agent (round-robin)")
    Component_Ext(pettingzoo, "PettingZoo Adapter", "Optional multi-agent RL interface")

    Rel(swarm_game, swarm_state, "Manages state transitions")
    Rel(swarm_game, swarm_config, "Configured by")
    Rel(swarm_game, potential_field, "Computes obstacle avoidance")
    Rel(swarm_game, coverage_map, "Updates explored regions")
    Rel(swarm_game, comm_graph, "Computes connectivity")

    Rel(mcts_engine, swarm_game, "Selects agent actions")
    Rel(pettingzoo, swarm_game, "Wraps as ParallelEnv")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Level 3: Component Diagram - SBIR Proposal Infrastructure

This diagram shows the proposal submission infrastructure supporting 5 target solicitations.

```mermaid
C4Component
    title Component Diagram - SBIR Proposal Infrastructure

    Container_Boundary(sbir, "SBIR Proposal Infrastructure") {
        Component(sam_guide, "SAM Registration Guide", "Markdown", "Step-by-step SAM.gov checklist: EIN, UEI, CAGE, NAICS 541715/541511")

        Component(timeline, "Submission Timeline", "Markdown", "Gantt-style calendar for 5 solicitations with prep windows and dependencies")

        Component(program_offices, "Program Offices", "Markdown", "Tier 1/2 contacts: DARPA STO, DOE ASCR, NAVAIR, AFWERX, AFRL, ONR")

        Component(budget_templates, "Budget Templates", "Markdown", "4 budget tables: DoD $250K, NSF $305K, AFWERX $75K, DARPA $1.5M")

        Component(proposal_configs, "Solicitation Configs", "YAML", "navy_n252_088.yaml, doe_ascr_c59.yaml, nsf_sbir.yaml, afwerx_open.yaml, darpa_d2p2.yaml")

        Component(ip_strategy, "IP Strategy", "Markdown", "3 provisional patent claims, trade secret inventory, publication plan, licensing strategy")

        Component(competitive, "Competitive Landscape", "Markdown", "PhysicsX, BeyondMath, Pasteur Labs, Godela, PhysicsNeMo comparison matrix")

        Component(valuation, "Valuation & M&A", "Markdown", "Stage-based valuation ($1M-$200M+), top 5 acquirers, revenue multiples")

        Component(differentiation, "Differentiation Matrix", "Markdown", "AlphaGalerkin vs PINNs vs FNO vs AMR vs PhysicsNeMo across 10 criteria")

        Component(sbir_demo, "SBIR Demo Script", "Python", "End-to-end benchmark runner with convergence plots and comparison reports")
    }

    Component_Ext(benchmark_runner, "PDEBenchmarkRunner", "Runs baselines and generates reports")
    Component_Ext(baseline_solvers, "Baseline Solvers", "FDM, Dorfler AMR, PINN")

    Rel(sbir_demo, benchmark_runner, "Executes benchmarks")
    Rel(sbir_demo, baseline_solvers, "Compares against")
    Rel(sbir_demo, proposal_configs, "Loads config")
    Rel(timeline, proposal_configs, "References deadlines from")
    Rel(budget_templates, proposal_configs, "Matches funding ranges")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### SBIR Infrastructure Components

| Component | Content | Key Files |
|-----------|---------|-----------|
| **SAM Guide** | Registration checklist with 7 steps and timeline | `docs/proposals/SAM_REGISTRATION_GUIDE.md` |
| **Submission Timeline** | Gantt chart for AFWERX, NSF, Navy, DOE, DARPA | `docs/proposals/SUBMISSION_TIMELINE.md` |
| **Program Offices** | Tier 1/2 POCs with engagement templates | `docs/proposals/PROGRAM_OFFICES.md` |
| **Budget Templates** | 4 agency-specific budget breakdowns | `docs/proposals/BUDGET_TEMPLATES.md` |
| **Solicitation Configs** | YAML configs with benchmark suites per solicitation | `config/proposals/*.yaml` |
| **IP Strategy** | 3 patent claims + trade secrets + publication plan | `docs/proposals/IP_STRATEGY.md` |
| **Competitive Landscape** | 6 competitors with strengths/weaknesses matrix | `docs/proposals/COMPETITIVE_LANDSCAPE.md` |
| **Valuation & M&A** | Stage-based valuation + top 5 acquirers | `docs/proposals/VALUATION_FRAMEWORK.md`, `MA_LANDSCAPE.md` |
| **Differentiation Matrix** | Technical comparison across 10 criteria | `docs/proposals/DIFFERENTIATION_MATRIX.md` |
| **SBIR Demo** | CLI script generating convergence plots + reports | `scripts/run_sbir_demo.py` |

---

## References

- **C4 Model**: [c4model.com](https://c4model.com)
- **Galerkin Transformers**: Cao et al. (2021)
- **FNet**: Lee-Thorp et al. (2021)
- **AlphaZero**: Silver et al. (2017)
- **Fredholm Theory**: Classical operator theory
- **Physics-Informed Neural Networks**: Raissi et al. (2019)
- **ReLoBRaLo**: Bischof & Kraus (2021)
- **GradNorm**: Chen et al. (2018)
- **Fourier Features**: Tancik et al. (2020)

---

## Related Documentation

- **PDE Game Framework C4**: [pde_game_c4.md](pde_game_c4.md) - Detailed C4 architecture for PDE solving
- **C4 Template**: [../templates/C4_TEMPLATE.md](../templates/C4_TEMPLATE.md) - Template for new modules

---

## Document Metadata

- **Version**: 3.0.0
- **Created**: 2026-01-26
- **Updated**: 2026-03-31
- **Format**: Mermaid C4 Diagrams
- **Status**: Complete
- **Audience**: Developers, Researchers, Computational Scientists, Technical Stakeholders
