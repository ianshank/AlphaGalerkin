# AlphaGalerkin C4 Architecture (Mermaid Format)

This document provides a comprehensive C4 architecture model for the AlphaGalerkin system using Mermaid diagrams.
The C4 model consists of four levels: System Context, Containers, Components, and Code.

The system supports two primary use cases:

1. **Go AI**: Resolution-independent game playing with zero-shot transfer between board sizes
2. **PDE Solving**: AlphaZero-style MCTS for adaptive basis selection and mesh refinement

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
        Container(dashboard, "E2E Dashboard", "Gradio / Python", "Interactive tabbed web UI exposing all AlphaGalerkin capabilities: Go AI, PDE Solver, PoC Scenarios, Training visualiser, About. Launches at http://localhost:7860. Config-driven via Pydantic DashboardConfig.")
        Container(cli, "CLI Entrypoints", "Python Scripts", "Command-line interface for training, benchmarking, PDE solving, and experiments")
        Container(gtp_server, "GTP Server", "Python", "Go Text Protocol server for game playing and analysis")

        Container(neural_operator, "Neural Operator Model", "PyTorch", "Core continuous operator learning model with Galerkin attention and FNet mixing")

        Container(mcts_engine, "MCTS Search Engine", "Python", "Monte Carlo Tree Search with neural network guidance for move/action selection")

        Container(pde_framework, "PDE Game Framework", "Python/PyTorch", "Treats PDE solving as sequential decision-making with MCTS-guided basis selection and mesh refinement")

        Container(training_pipeline, "Training Pipeline", "PyTorch", "Self-play, replay buffer, physics-informed loss, adaptive loss balancing, and checkpoint management")

        Container(math_kernel, "Math Kernel", "NumPy/PyTorch", "Mathematical primitives: multi-scale Fourier features, Galerkin projection, spectral filtering")

        Container(poc_framework, "PoC Framework", "Python/Pydantic", "Scenario-based validation of mathematical claims with reproducible experiments")

        Container(data_layer, "Data Layer", "PyTorch Dataset", "Board state preprocessing, variable-size batching, physics data generation")

        Container(research, "Research & Benchmarking", "Python/SciPy", "SBIR baselines (FDM, AMR, PINN), benchmark runner, convergence reports")

        Container(geometry, "Domain Geometry & Time-Stepping", "Python/PyTorch", "Rectangular, L-shaped, cylinder domains; Forward Euler, RK4, Crank-Nicolson")

        Container(swarm, "Swarm Planning", "Python/NumPy", "Multi-agent swarm game with potential field avoidance and coverage optimization")

        Container(agents, "Agent Orchestration", "Python/Pydantic", "Multi-physics PDE solving via specialized sub-agents (Orchestrator, Collocation, Decomposition, Coupling, Meta)")
        Container(engines, "Chess Engine Integration", "Python/UCI", "Stockfish evaluation, Elo calculation, UCI protocol adapter, tournament management")
        Container(curriculum, "Curriculum Learning", "Python", "Progressive difficulty scheduling and model zoo management for multi-game training")
        Container(demos, "SBIR Demo Suite", "Python", "End-to-end benchmark demonstrations for Navy/DOE/NSF/AFWERX proposals")

        ContainerDb(checkpoint_store, "Model Checkpoints", "File System", "Stores trained model weights and training state")
        ContainerDb(results_store, "Experiment Results", "JSON/YAML", "Stores PoC scenario results and metrics")
    }

    System_Ext(go_gui, "Go GUI", "GTP Client")
    System_Ext(compute, "GPU Cluster", "CUDA Infrastructure")

    Rel(user, dashboard, "Explores capabilities via browser", "HTTP / Gradio")
    Rel(user, cli, "Runs commands", "CLI")
    Rel(go_gui, gtp_server, "Sends GTP commands", "GTP/TCP")

    Rel(dashboard, mcts_engine, "Drives Go AI game moves")
    Rel(dashboard, pde_framework, "Triggers Poisson solve / comparison")
    Rel(dashboard, poc_framework, "Runs complexity & stability scenarios")
    Rel(dashboard, neural_operator, "Queries model for game tab")

    Rel(cli, training_pipeline, "Initiates training")
    Rel(cli, poc_framework, "Runs experiments")
    Rel(cli, pde_framework, "Solves PDEs")
    Rel(gtp_server, neural_operator, "Gets policy/value")
    Rel(gtp_server, mcts_engine, "Performs search")

    Rel(mcts_engine, neural_operator, "Evaluates positions")
    Rel(mcts_engine, pde_framework, "Guides basis/mesh selection")
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

    Rel(cli, agents, "Orchestrates multi-physics solving")
    Rel(agents, pde_framework, "Delegates PDE sub-problems")
    Rel(agents, neural_operator, "Uses for solution evaluation")
    Rel(training_pipeline, curriculum, "Stages training progression")
    Rel(training_pipeline, engines, "Evaluates vs Stockfish")
    Rel(cli, demos, "Generates SBIR benchmark reports")

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
| **Agent Orchestration** | Multi-physics PDE decomposition and coupling via specialized sub-agents | Python, Pydantic, structlog |
| **Chess Engine Integration** | UCI protocol, Elo evaluation, tournament management | Python, Stockfish, UCI |
| **Curriculum Learning** | Progressive board size and difficulty scheduling | Python, model zoo |
| **SBIR Demo Suite** | End-to-end benchmark demos for government proposal validation | Python, Matplotlib, JSON |

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
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Model under test")
    Component_Ext(physics_data, "Physics Data Generator", "Synthetic Poisson equation data")
    ComponentDb_Ext(results_store, "Results Store", "JSON/YAML files")

    Rel(cli_poc, registry, "Lists scenarios")
    Rel(cli_poc, runner, "Executes scenarios")
    Rel(cli_poc, config_mgr, "Loads configs")
    
    Rel(registry, scenario_transfer, "Registers")
    Rel(registry, scenario_complexity, "Registers")
    Rel(registry, scenario_stability, "Registers")
    
    Rel(runner, scenario_transfer, "Runs")
    Rel(runner, scenario_complexity, "Runs")
    Rel(runner, scenario_stability, "Runs")
    
    Rel(scenario_transfer, neural_operator, "Tests transfer")
    Rel(scenario_complexity, neural_operator, "Benchmarks complexity")
    Rel(scenario_stability, neural_operator, "Monitors stability")
    
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
        Component(benchmark_runner, "PDEBenchmarkRunner", "Python Class", "Runs AlphaGalerkin vs baselines on benchmark suites, generates JSON/Markdown reports")

        Component(fdm_solver, "UniformFDMSolver", "BaseSolver", "2nd-order finite differences on uniform grid via scipy.sparse")

        Component(amr_solver, "DorflerAMRSolver", "BaseSolver", "Dorfler bulk-chasing adaptive mesh refinement on 1D grids")

        Component(pinn_solver, "SimplePINNSolver", "BaseSolver", "Physics-Informed Neural Network baseline with autograd Laplacian")

        Component(solver_registry, "SOLVER_REGISTRY", "Dict Registry", "Maps solver names to classes: get_solver(), list_solvers()")

        Component(benchmark_config, "Benchmark Configs", "YAML", "sbir_suite.yaml, navy_n252_088.yaml, doe_ascr_c59.yaml, nsf_sbir.yaml")
    }

    Component_Ext(pde_operators, "PDE Operators", "Provides exact solutions, residuals")
    Component_Ext(alphagalerkin, "AlphaGalerkin Engine", "MCTS-guided solver under comparison")

    Rel(benchmark_runner, fdm_solver, "Runs baseline")
    Rel(benchmark_runner, amr_solver, "Runs baseline")
    Rel(benchmark_runner, pinn_solver, "Runs baseline")
    Rel(benchmark_runner, solver_registry, "Discovers solvers")
    Rel(benchmark_runner, benchmark_config, "Loads config")

    Rel(fdm_solver, pde_operators, "Uses source_term, boundary_value")
    Rel(amr_solver, pde_operators, "Uses source_term, exact_solution")
    Rel(pinn_solver, pde_operators, "Uses residual via autograd")

    Rel(benchmark_runner, alphagalerkin, "Compares against baselines")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### SBIR Research Components

| Component | Responsibility | Key Interface |
|-----------|----------------|---------------|
| **PDEBenchmarkRunner** | Load YAML config, run all solver x problem combos, compute convergence rates | `run_all()`, `generate_report()` |
| **UniformFDMSolver** | Reference FDM solution on 1D/2D uniform grids | `solve(operator, n_dof)` |
| **DorflerAMRSolver** | Adaptive refinement with residual error indicators | `solve(operator, n_dof)` |
| **SimplePINNSolver** | PINN baseline with configurable MLP + autograd | `solve(operator, n_dof)` |
| **SOLVER_REGISTRY** | Plugin system for baseline solvers | `get_solver(name)`, `list_solvers()` |

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

## Recent Architecture Changes (v0.3.0)

### New Modules Added
- **`src/agents/`** — Multi-physics agent orchestration (`OrchestratorAgent`, `CollocationAgent`, `DecompositionAgent`, `CouplingAgent`, `MetaAgent`)
- **`src/engines/`** — UCI chess engine integration (`UCIAdapter`, `EloCalculator`, `EngineMatch`, `TournamentManager`)
- **`src/curriculum/`** — Progressive training curriculum with model zoo integration
- **`src/tournament/`** — Chess tournament management and Elo rating system
- **`src/demos/`** — SBIR benchmark demonstration scripts
- **`src/research/`** — Classical solver baselines (FDM, AMR, PINN) and `PDEBenchmarkRunner`
- **`src/pde/geometry.py`** — Domain abstractions (Rectangular, L-shaped, Cylinder)
- **`src/pde/time_stepping.py`** — Time integrators (ForwardEuler, RK4, CrankNicolson)
- **`src/pde/games/swarm_planning.py`** — Multi-agent swarm with potential field avoidance
- **`src/games/pettingzoo_adapter.py`** — PettingZoo `ParallelEnv` wrapper
- **`src/training/base_trainer.py`** — Abstract `BaseTrainer[ConfigT]` with shared AMP/grad/LR
- **`src/training/losses/`** — Unified `LossRegistry` with `get_loss()` factory

### Key Architecture Decisions (v0.3.0)
- Game-agnostic self-play: `SelfPlayWorker` accepts any `GameInterface` (Go, Chess, PDE)
- Action mask derived from policy tensor size, not hardcoded `board_size²+1`
- DDP-safe: `_training_resolution` mutation moved out of `forward()`
- LBB fields exposed in config (Babuska-Brezzi condition): `lbb_loss_weight`, `lbb_target`, `lbb_eps`
- Coverage gates: 85% overall + per-module, nightly CI schedule

---

*Last Updated: 2026-04-01*
*Architecture Version: 3.0.0*
*Corresponds to AlphaGalerkin v0.3.0*

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
