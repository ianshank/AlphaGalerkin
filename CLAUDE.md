# CLAUDE.md - AlphaGalerkin Context

## Project Overview
AlphaGalerkin is a resolution-independent Go AI that uses Continuous Operator Learning
(Galerkin Transformers & FNet) instead of discrete CNNs, enabling zero-shot transfer
between board sizes (e.g., 9x9 to 19x19) and accelerating MCTS rollouts via FFT mixing.

## Mathematical Decisions
- [2026-01-26]: Chosen Kernel: Fredholm integral equation with Green's function formulation.
- [2026-01-26]: Basis function selection: Fourier Features for positional encoding.
- [2026-01-26]: Normalization scheme: Monte Carlo integral normalization (1/n) for Galerkin attention.
- [2026-01-26]: LBB Stability: dim(Key) >= dim(Query) to satisfy inf-sup condition.

## Architecture Decisions
- [2026-01-26]: Strategy Body uses GalerkinLinearAttention for O(N) global influence modeling.
- [2026-01-26]: Tactical Head uses SoftmaxAttention to preserve injectivity for local reading.
- [2026-01-26]: FNet mixing uses real-valued FFT (torch.fft.rfft2) for efficiency.
- [2026-01-26]: All tensor operations use einops for dimension clarity.

## Key Mathematical Operators

### GalerkinAttention
Implements Petrov-Galerkin projection with O(N) complexity:
- Projects values onto Key basis: K^T V (Monte Carlo integral)
- Reconstructs in Query basis: Q * Context
- Normalization: 1/n (not 1/sqrt(d))

### FNetBlock
FFT-based mixing for high-speed rollouts:
- FFT2D -> Spectral Mixing -> iFFT2D
- Enables batch MCTS leaf evaluation

### StabilityGuard
Monitors LBB condition during training:
- Computes singular values of Key-to-Value projection
- Ensures sigma_min > beta > 0

## Training Infrastructure
- [2026-01-26]: Added complete training pipeline with self-play, replay buffer, and trainer.
- [2026-01-26]: Loss = policy_CE + value_MSE + lbb_regularization for Galerkin stability.
- [2026-01-26]: Replay buffer supports uniform and prioritized experience replay.
- [2026-01-26]: Variable board size batching via padding and masking.
- [2026-01-26]: Checkpoint manager with best model tracking and rotation.

## Physics PoC (Supervised Learning Validation)
- [2026-01-26]: Added Poisson equation solver for synthetic data generation.
- [2026-01-26]: PhysicsOperator neural network for influence field prediction.
- [2026-01-26]: Zero-shot transfer validation: Train on 9x9 → Evaluate on 19x19.
- [2026-01-26]: Success criterion: MSE < 0.05 on 19x19 without retraining.
- [2026-01-26]: **MILESTONE ACHIEVED**: Zero-shot transfer MSE = 0.000209 (240x better than threshold)
- [2026-01-26]: Added W&B integration for experiment tracking (--wandb flag).

## PoC Scenario Framework
- [2026-01-26]: Added configuration-driven PoC scenario framework (src/poc/).
- [2026-01-26]: Three built-in scenarios: transfer, complexity, stability.
- [2026-01-26]: Pydantic-validated configs with no hardcoded values.
- [2026-01-26]: Structured logging via structlog throughout.
- [2026-01-26]: C4 architecture documentation in docs/architecture/c4_model.md.
- [2026-01-26]: Added comprehensive C4 architecture in Mermaid format (docs/architecture/c4_mermaid.md).

## Milestones
- [2026-01-26]: **Zero-Shot Transfer Validated** - Physics PoC achieved MSE 0.000209 on 19x19 (trained on 9x9)
- [2026-01-26]: **Training Pipeline Operational** - End-to-end GPU training with MCTS self-play working
- [2026-02-01]: **CI/CD Pipeline Added** - GitHub Actions workflow with lint, type check, tests, coverage
- [2026-02-01]: **Video Compression Hyperprior Fixed** - Proper z_bitstream encoding/decoding for entropy model
- [2026-02-01]: **Chess Game Implementation** - Full Chess rules with AlphaZero-style encoding (119 planes)
- [2026-02-04]: **P0 Critical Fixes** - Emergency checkpoint, GTP player assignment, parallel self-play
- [2026-02-04]: **PDE-MCTS Integration** - PDEGameAdapter bridges PDE games to MCTS search engine
- [2026-02-04]: **Physics Loss Wired** - Laplacian regularization via autodiff in PhysicsLoss
- [2026-02-04]: **Curriculum Config Schema** - curriculum_schedule field on TrainingConfig with transition logging
- [2026-03-30]: **CI Hardening** - MyPy strict enforcement, coverage gates raised to 75%/80%, nightly schedule
- [2026-03-30]: **Config-Driven LBB Loss** - Magic numbers surfaced as Pydantic fields with mathematical docs
- [2026-03-30]: **Race Condition Fixed** - _training_resolution mutation removed from forward(), explicit setter added
- [2026-03-30]: **Checkpoint Migration** - Version-aware migration system (0.0.0→1.0.0→1.1.0) with registry pattern
- [2026-03-30]: **Loss Package Unification** - src/training/losses/ with registry, backwards-compatible imports
- [2026-03-30]: **NavierStokesOperator** - Taylor-Green vortex benchmark with exact analytical solution
- [2026-03-30]: **BurgersOperator Enhanced** - Cole-Hopf exact solution, configurable shock params, convergence rates
- [2026-03-30]: **Domain Geometry Abstractions** - RectangularDomain, LShapedDomain, CylinderFlowDomain
- [2026-03-30]: **L-Shaped Poisson Operator** - r^(2/3)*sin(2θ/3) singularity for AMR benchmarking
- [2026-03-30]: **Time-Stepping Module** - Forward Euler, RK4, Crank-Nicolson with factory pattern
- [2026-03-30]: **SBIR Proposal Infrastructure** - Navy N252-088, AFWERX, DOE ASCR, NSF configs + benchmark suite
- [2026-03-30]: **IP Strategy Documented** - 3 provisional patent claims, publication plan, dual-licensing
- [2026-03-30]: **Property-Based Tests** - Hypothesis tests for loss, PDE operators, attention mechanisms
- [2026-03-30]: **Numerical Stability Tests** - Edge cases, extreme values, mixed precision, NaN propagation
- [2026-04-02]: **Physics Loss Fully Wired** - CombinedAlphaGalerkinPhysicsLoss passes lbb_constant, action_mask, model to trainer
- [2026-04-02]: **2D AMR Baseline** - DorflerAMRSolver extended to 2D with element-wise refinement and Dorfler marking
- [2026-04-02]: **Navier-Stokes FDM Solver** - Chorin projection method baseline for Taylor-Green vortex benchmark
- [2026-04-02]: **PDE GameInterface Bridge** - PDEGameInterface wraps PDEGame for GameRegistry registration
- [2026-04-02]: **PDE Games Registered** - pde_basis and pde_mesh registered in GameRegistry via src/pde/register_games.py
- [2026-04-02]: **PDE Training Config** - config/train_pde.yaml for MCTS-guided basis selection training
- [2026-04-02]: **ROI Implementation Plan** - Tiered next-steps plan in docs/ROI_IMPLEMENTATION_PLAN.md
- [2026-04-07]: **Physics Loss Tests** - 52 comprehensive tests for physics-informed training (config toggle, gradient flow, property-based)
- [2026-04-07]: **SBIR Benchmark Demo** - End-to-end sbir_demo.py with HTML/JSON/Markdown report generation
- [2026-04-07]: **Loss Balancing Audit** - Fixed NaN/Inf propagation bugs in ReLoBRaLo/SoftAdapt, 96 property-based tests
- [2026-04-07]: **PDE-MCTS Self-Play Wired** - PDE games auto-register, create_trainer() accepts game parameter, 40 tests
- [2026-04-07]: **Visualization Module** - PlotRegistry with 5 plot types, HTMLReportGenerator with themed templates
- [2026-04-07]: **Coverage Expansion** - 390+ new tests across training, PDE, games, curriculum, modeling modules
- [2026-04-07]: **BaseTrainer Refactor** - Extracted shared AMP, grad clip, LR scheduling into BaseTrainer base class
- [2026-04-07]: **Distributed Trainer Tests** - 35 new tests for DistributedTrainer (metrics, checkpoints, multi-process)
- [2026-04-07]: **Test Speed Fixes** - Mocked MCTS self-play in all trainer tests to prevent hanging (chess, physics, pipeline)
- [2026-04-07]: **Coverage Sprint** - 115 new tests: statistics significance (52), tuning sampler/tuner (33), ONNX integration (30)
- [2026-04-07]: **GPU Skip Hook** - Root conftest.py auto-skips gpu_required tests when CUDA unavailable; 0 spurious failures
- [2026-04-07]: **Gumbel MCTS Search Tests** - 38 integration tests for search(), _sequential_halving(), _simulate(), get_improved_policy(), factory
- [2026-04-11]: **Defense Domain Dashboards** - Reentry TPS, Wildfire Spread, Missile Defense tabs with PDE solvers and resolution comparison
- [2026-04-11]: **Dashboard Testing Infrastructure** - BE handler tests (19), FE rendering tests (17), E2E Playwright browser tests (18)
- [2026-04-11]: **Gradio 6 Compatibility Fixes** - CSS moved to Blocks(), PILImage runtime import for type-hint resolution

## SBIR Positioning
- **Verified Novelty Gap**: No published papers combine MCTS with Galerkin methods for PDE/mesh refinement
- **Target Solicitations**: Navy N252-088, DOE ASCR C59-01, NSF SBIR, AFWERX Open Topic
- **TRL Level**: 3-4 (advancing to 5-6 with benchmark demonstrations)
- **Key Differentiators**: Multi-step look-ahead (vs myopic RL), provable convergence (vs PINNs/FNO), no training data needed

## Next-Phase Infrastructure (v2.0)

### Distributed Training (src/distributed/)
- [2026-01-26]: Multi-node training via PyTorch DDP with NCCL backend.
- [2026-01-26]: Gradient synchronization with accumulation and compression support.
- [2026-01-26]: Distributed self-play coordination across nodes.
- [2026-01-26]: Model zoo for checkpoint management and curriculum learning.
- [2026-01-26]: Support for torchrun, SLURM, and custom launchers.

### ONNX Export (src/deployment/)
- [2026-01-26]: PyTorch to ONNX conversion with dynamic shape support.
- [2026-01-26]: Quantization support (dynamic/static) for edge deployment.
- [2026-01-26]: ONNX Runtime inference wrapper with multi-provider support.
- [2026-01-26]: Model validation against PyTorch outputs.

### Multi-Game Support (src/games/)
- [2026-01-26]: Abstract GameInterface for game-agnostic architecture.
- [2026-01-26]: Game registry with decorator-based registration.
- [2026-01-26]: Go implementation with full rules (Chinese scoring, superko).
- [2026-01-26]: 8-fold symmetry support for data augmentation.
- [2026-02-01]: Chess implementation with full rules (castling, en passant, promotion).
- [2026-02-01]: Chess uses 119-plane AlphaZero encoding with horizontal symmetry.

### Advanced MCTS (src/mcts/)
- [2026-01-26]: Gumbel AlphaZero implementation with sequential halving.
- [2026-01-26]: Improved policy targets via completed Q-values.
- [2026-01-26]: Gumbel-Top-k sampling for exploration.

### Enhanced PoC Framework (src/poc/tuning/, src/poc/statistics/)
- [2026-01-26]: Hyperparameter tuning with TPE, grid, and random samplers.
- [2026-01-26]: Statistical significance testing (t-test, Mann-Whitney, bootstrap).
- [2026-01-26]: Effect size calculations (Cohen's d, Hedges' g, Cliff's delta).
- [2026-01-26]: Multiple comparison corrections (Bonferroni, Holm, FDR).

### Google Vertex AI Training (src/vertex/)
- [2026-01-31]: Complete Vertex AI training integration for cloud-based training.
- [2026-01-31]: Pydantic configuration schemas for machine types, regions, and accelerators.
- [2026-01-31]: GCS checkpoint manager with local caching and atomic operations.
- [2026-01-31]: Multi-node distributed training setup with automatic environment detection.
- [2026-01-31]: Spot instance preemption handling with signal-based detection.
- [2026-01-31]: Cost tracking and estimation with GCP pricing data.
- [2026-01-31]: Vertex-aware trainer wrapper integrating all Vertex AI features.
- [2026-01-31]: Docker container infrastructure for training jobs.
- [2026-01-31]: CLI tools for job launching, monitoring, and management.
- [2026-01-31]: Full test suite with 124 tests (100% passing).

## Module Development Templates (src/templates/)
- [2026-01-27]: Reusable infrastructure for building new AlphaGalerkin modules.
- [2026-01-27]: Pydantic-based configuration with validation and no hardcoded values.
- [2026-01-27]: Thread-safe singleton registries with decorator-based registration.
- [2026-01-27]: Structured logging with context binding and timing utilities.
- [2026-01-27]: Base executable classes with result tracking and error handling.
- [2026-01-27]: CLI utilities with common options and error handling.
- [2026-01-27]: C4 architecture template in Mermaid format.
- [2026-01-27]: Full test suite with 107 tests (100% passing).

## PDE Game Framework (src/pde/)
- [2026-01-27]: PDEGame abstraction for treating PDE solving as sequential decision-making.
- [2026-01-27]: PDEState and PDEResult dataclasses for state representation and metrics.
- [2026-01-27]: PDE operator definitions with automatic differentiation (Poisson, Burgers, Advection-Diffusion, Heat).
- [2026-01-27]: Basis selection game for MCTS-guided Galerkin approximation.
- [2026-01-27]: Mesh refinement game for adaptive h/p-refinement strategies.
- [2026-01-27]: PDEOperatorRegistry with decorator-based registration.
- [2026-01-27]: Comprehensive test suite for all PDE components.
- [2026-02-04]: PDEGameAdapter bridging PDE games to MCTS GameInterface protocol.

## Adaptive Loss Balancing (src/training/loss_balancing.py)
- [2026-01-27]: ReLoBRaLo (Relative Loss Balancing with Random Lookback) for physics-informed training.
- [2026-01-27]: GradNorm gradient normalization for multi-task learning.
- [2026-01-27]: Uncertainty weighting with learnable log-variance parameters.
- [2026-01-27]: SoftAdapt rate-based adaptation for improving slower losses.
- [2026-01-27]: Factory function for creating balancers with Pydantic configuration.

## Physics-Informed Loss Components (src/training/physics_loss.py)
- [2026-01-27]: ResidualLoss for PDE residual minimization via autodiff.
- [2026-01-27]: BoundaryLoss for Dirichlet/Neumann/Robin BC enforcement.
- [2026-01-27]: InitialConditionLoss for time-dependent PDEs.
- [2026-01-27]: ConservationLoss for integral conservation properties.
- [2026-01-27]: PhysicsInformedLoss combining all physics terms with adaptive balancing.
- [2026-01-27]: CombinedAlphaGalerkinPhysicsLoss integrating policy/value with physics.

## Multi-Scale Fourier Features (src/modeling/multiscale_fourier.py)
- [2026-01-27]: MultiScaleFourierFeatures with multiple frequency bands to overcome spectral bias.
- [2026-01-27]: AdaptiveFourierFeatures with attention-based frequency selection.
- [2026-01-27]: ProgressiveFourierFeatures for curriculum-based frequency introduction.
- [2026-01-27]: SpatialPositionalEncoding for 2D grid data.
- [2026-01-27]: Configurable via Pydantic FourierFeaturesConfig.

## Neural Video Compression (src/video_compression/)
- [2026-01-30]: Resolution-independent neural video codec using Galerkin attention and FNet mixing.
- [2026-01-30]: Analysis transform (encoder) with O(N) Galerkin attention and O(N log N) FFT mixing.
- [2026-01-30]: Synthesis transform (decoder) with temporal cross-attention for P/B frames.
- [2026-01-30]: Scale hyperprior entropy model (Ballé et al.) for learned compression.
- [2026-01-30]: Differentiable quantization: noise, STE, and soft quantization modes.
- [2026-01-30]: MCTS-based rate control for GOP-level bit allocation with MuZero-style learned models.
- [2026-01-30]: Quality metrics: PSNR, SSIM, MS-SSIM, BD-rate computation.
- [2026-01-30]: R-D training with MSE, MS-SSIM, and perceptual (VGG) losses.
- [2026-01-30]: GOP manager for I/P/B frame scheduling and reference management.
- [2026-01-30]: Range encoder/decoder for lossless entropy coding.

### Key Architecture Decisions
- **Resolution Independence**: All encoder/decoder layers accept arbitrary (H, W) divisible by downsample factor.
- **Galerkin Attention**: Q(K^T V) formula with O(N) complexity, no softmax.
- **FNet Mixing**: torch.fft.fft2() for O(N log N) spatial mixing, no learnable parameters.
- **GDN/IGDN**: Generalized Divisive Normalization for density modeling.

## Known Issues
- [None yet]

## Verification Commands
```bash
# Linting and type checking
ruff check src/
mypy src/ --strict

# Unit tests
pytest tests/math_kernel/ -v
pytest tests/training/ -v

# Integration tests
pytest tests/integration/ -v

# Full test suite
pytest tests/ -v

# Verify resolution independence
python -m src.tools.verify_invariance --train-size 9 --infer-size 19
```

## Training Commands
```bash
# Default training (full config)
python -m scripts.train

# Fast test training (small model, few steps)
python -m scripts.train --config-name=train_fast

# Override parameters
python -m scripts.train training.batch_size=64 training.total_steps=10000

# Resume from checkpoint
python -m scripts.train +resume=checkpoints/alphagalerkin/checkpoint_00010000.pt

# Train on GPU with custom experiment name
python -m scripts.train device=cuda experiment_name=my_experiment
```

## Physics PoC Commands
```bash
# Train physics operator on Poisson data (supervised learning)
python -m src.experiments.train_physics

# Train with W&B logging
python -m src.experiments.train_physics --wandb

# Custom training configuration
python -m src.experiments.train_physics --train-size 9 --eval-size 19 --n-epochs 100

# Verify zero-shot transfer (train 9x9 → eval 9,13,19)
python -m src.experiments.verify_transfer

# Verify with existing model
python -m src.experiments.verify_transfer --model-path outputs/physics_poc/best_model.pt

# Run FNet vs Softmax speed benchmark
python -m src.experiments.benchmark_fnet

# Benchmark with custom sizes
python -m src.experiments.benchmark_fnet --sizes 81,169,361,625 --batch-size 64

# Run Fredholm integral property tests
pytest tests/math_kernel/test_fredholm.py -v
```

## PoC Scenario Framework Commands
```bash
# List available scenarios
python -m src.poc.cli list

# Show scenario details
python -m src.poc.cli info transfer

# Run all scenarios
python -m src.poc.cli run

# Run specific scenario
python -m src.poc.cli run --scenario transfer

# Run from config file (full suite)
python -m src.poc.cli run --config config/scenarios/poc_full.yaml

# Run quick validation suite
python -m src.poc.cli run --config config/scenarios/poc_quick.yaml

# Run with parallel workers
python -m src.poc.cli run --parallel 4

# Compare two runs
python -m src.poc.cli compare run_a run_b

# PoC framework unit tests
pytest tests/poc/ -v
```

## Distributed Training Commands
```bash
# Launch distributed training with torchrun (4 GPUs)
torchrun --nproc_per_node=4 scripts/train_distributed.py

# Multi-node training (2 nodes, 4 GPUs each)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=<MASTER_IP> scripts/train_distributed.py

# Unit tests for distributed module
pytest tests/distributed/ -v
```

## Vertex AI Training Commands
```bash
# Build and push training container to Artifact Registry
./scripts/build_vertex_container.sh my-project us-central1

# Launch training job on Vertex AI
python -m scripts.train_vertex \
    --project my-project \
    --region us-central1 \
    --bucket gs://my-training-bucket \
    --machine-type a2-highgpu-1g \
    --accelerator-type NVIDIA_TESLA_A100 \
    --accelerator-count 1 \
    --container-uri us-central1-docker.pkg.dev/my-project/alphagalerkin/trainer:latest

# Launch spot (preemptible) training for cost savings
python -m scripts.train_vertex \
    --project my-project \
    --bucket gs://my-training-bucket \
    --spot

# Launch multi-node distributed training
python -m scripts.train_vertex \
    --project my-project \
    --bucket gs://my-training-bucket \
    --machine-type a2-highgpu-8g \
    --accelerator-type NVIDIA_TESLA_A100 \
    --accelerator-count 8 \
    --replica-count 4

# List running Vertex AI jobs
python -m scripts.vertex_jobs list --project my-project

# Show job status
python -m scripts.vertex_jobs show JOB_ID --project my-project

# Wait for job completion
python -m scripts.vertex_jobs wait JOB_ID --project my-project

# Cancel a running job
python -m scripts.vertex_jobs cancel JOB_ID --project my-project

# View job logs
python -m scripts.vertex_jobs logs JOB_ID --project my-project

# Unit tests for Vertex AI module
pytest tests/vertex/ -v
```

## ONNX Export Commands
```bash
# Export model to ONNX
python -m src.deployment.export_onnx \
    --checkpoint path/to/model.pt \
    --output model.onnx

# Export with quantization
python -m src.deployment.export_onnx \
    --checkpoint path/to/model.pt \
    --output model_int8.onnx \
    --quantize dynamic

# Validate exported model
python -m src.deployment.validate \
    --pytorch path/to/model.pt \
    --onnx model.onnx

# Unit tests for deployment module
pytest tests/deployment/ -v
```

## Multi-Game Commands
```bash
# Train on Go (default)
python -m scripts.train game=go

# List registered games
python -c "from src.games import GameRegistry; print(GameRegistry().list_games())"

# Unit tests for games module
pytest tests/games/ -v
```

## Module Development Template Commands
```bash
# Run template tests
pytest tests/templates/ -v

# Example: Create a new module configuration
python -c "
from src.templates.config import BaseModuleConfig, create_config_class
from pydantic import Field

# Method 1: Subclass directly
class MyModuleConfig(BaseModuleConfig):
    my_param: int = Field(default=100, ge=1, description='My parameter')

config = MyModuleConfig(name='test')
print(f'Config hash: {config.compute_hash()}')

# Method 2: Use factory function
QuickConfig = create_config_class(
    'QuickConfig',
    my_float=(float, Field(default=0.5, gt=0, lt=1)),
)
quick = QuickConfig(name='quick')
print(f'Quick config: {quick.my_float}')
"

# Example: Create and use a registry
python -c "
from src.templates.registry import create_registry

class BaseProcessor:
    def process(self, data): raise NotImplementedError

ProcessorRegistry, register_processor = create_registry('Processor', BaseProcessor)

@register_processor('upper')
class UpperProcessor(BaseProcessor):
    def process(self, data): return data.upper()

# Use the registry
proc_cls = ProcessorRegistry().get('upper')
processor = proc_cls()
print(processor.process('hello'))  # HELLO
"

# Example: Use structured logging
python -c "
from src.templates.logging import create_logger_class, configure_module_logging

configure_module_logging(level='DEBUG')
MyLogger = create_logger_class('MyModule')
logger = MyLogger('component', run_id='test123')

with logger.timed('operation'):
    logger.metric('accuracy', 0.95, epoch=1)
"
```

## Hyperparameter Tuning Commands
```bash
# Run hyperparameter tuning for transfer scenario
python -c "
from src.poc.tuning import HyperparameterTuner, TuningConfig
from src.poc.scenarios.transfer import TransferScenario

config = TuningConfig(
    n_trials=50,
    sampler='tpe',
    search_space={
        'd_model': {'type': 'int', 'low': 64, 'high': 256, 'log_scale': True},
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log_scale': True},
    }
)
tuner = HyperparameterTuner(config, TransferScenario)
result = tuner.tune()
print(f'Best params: {result.best_params}')
"

# Statistical comparison of two runs
python -c "
from src.poc.statistics import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
result = analyzer.compare_runs([0.05, 0.04, 0.06], [0.03, 0.02, 0.04])
print(f'p-value: {result.p_value}, significant: {result.is_significant}')
"
```

## Video Compression Commands
```bash
# Train compression model
python scripts/train_compression.py --data-dir data/images --epochs 100

# Train with specific lambda
python scripts/train_compression.py --data-dir data/images --lambda-rd 0.01

# Encode video
python scripts/encode_video.py input.mp4 output.agk --qp 32

# Encode with custom model
python scripts/encode_video.py input.mp4 output.agk --model checkpoints/codec.pt

# Run video compression tests
pytest tests/video_compression/ -v

# Test configuration validation
pytest tests/video_compression/unit/test_config.py -v

# Test encoder/decoder
pytest tests/video_compression/unit/test_encoder.py tests/video_compression/unit/test_decoder.py -v
```

## PDE Game Commands
```bash
# Run PDE game tests
pytest tests/pde/ -v

# Example: Create and use PDE operators
python -c "
from src.pde.operators import PoissonOperator
from src.pde.config import PDEConfig, PDEType
import numpy as np

config = PDEConfig(name='test', pde_type=PDEType.POISSON)
operator = PoissonOperator(config)
points = operator.generate_collocation_points(100)
source = operator.source_term(points)
print(f'Generated {len(points)} collocation points')
"

# Example: Create basis selection game
python -c "
from src.pde.games import BasisSelectionGame
from src.pde.operators import PoissonOperator
from src.pde.config import PDEConfig, PDEGameConfig, PDEType

pde_config = PDEConfig(name='poisson', pde_type=PDEType.POISSON)
game_config = PDEGameConfig(name='basis_game', pde_config=pde_config, game_mode='basis_selection')
operator = PoissonOperator(pde_config)
game = BasisSelectionGame(operator, game_config)
state = game.get_initial_state()
print(f'Initial error: {state.error_estimate:.6f}')
"

# Example: Use adaptive loss balancing
python -c "
from src.training.loss_balancing import create_loss_balancer, LossBalancingConfig, BalancingStrategy
import torch

config = LossBalancingConfig(name='test', strategy=BalancingStrategy.RELOBRALO)
balancer = create_loss_balancer(config, ['policy', 'value', 'physics'])
losses = {'policy': torch.tensor(1.0), 'value': torch.tensor(0.5), 'physics': torch.tensor(2.0)}
result = balancer.compute_weighted_loss(losses)
print(f'Weights: {result.weights}')
"
```

## Directory Structure
```
src/
  modeling/     - Neural architectures and layers
    multiscale_fourier.py - Multi-scale Fourier features for spectral bias mitigation
  math_kernel/  - Basis functions, integral approximations
  mcts/         - Monte Carlo Tree Search logic
    gumbel.py         - Gumbel AlphaZero MCTS implementation
  tools/        - Verification and utility scripts
  training/     - Training infrastructure
    loss.py           - AlphaGalerkinLoss (policy + value + LBB)
    loss_balancing.py - ReLoBRaLo and other adaptive loss balancing
    physics_loss.py   - Physics-informed loss components
    replay_buffer.py  - Uniform and prioritized replay buffers
    self_play.py      - MCTS-based self-play game generation
    trainer.py        - Main Trainer class
    checkpoint.py     - Checkpoint save/load management
    evaluation.py     - Win rate and policy agreement metrics
  data/         - Data loading and preprocessing
    dataset.py        - PyTorch Dataset classes
    collate.py        - Variable board size collation
  physics/      - Synthetic physics data generation
    poisson.py        - Poisson equation solver (DST-based)
  experiments/  - Physics PoC experiments
    physics_model.py  - PhysicsOperator neural network
    train_physics.py  - Supervised learning on Poisson data
    verify_transfer.py - Zero-shot transfer verification
    benchmark_fnet.py - FNet O(N log N) speed benchmark
  distributed/  - Distributed training infrastructure
    config.py         - Pydantic distributed config schemas
    trainer.py        - DistributedTrainer with DDP
    gradient_sync.py  - NCCL gradient synchronization
    launcher.py       - torchrun/SLURM launcher utilities
    worker.py         - Distributed self-play workers
    model_zoo.py      - Model checkpoint management
  deployment/   - Model export and deployment
    config.py         - Export/quantization config schemas
    export_onnx.py    - PyTorch to ONNX conversion
    quantize.py       - Model quantization utilities
    runtime.py        - ONNX Runtime inference wrapper
    validate.py       - Export validation tools
  games/        - Multi-game support
    interface.py      - Abstract GameInterface base class
    registry.py       - Game registration and discovery
    state.py          - Generic game state representation
    go.py             - Go game implementation
  pde/          - PDE Game Framework
    config.py         - Pydantic PDE configuration schemas
    game.py           - Abstract PDEGame base class
    operators.py      - PDE operator definitions (Poisson, Burgers, etc.)
    registry.py       - PDE operator registration
    mcts_adapter.py   - Adapter bridging PDE games to MCTS GameInterface
    games/            - Concrete PDE game implementations
      basis_selection.py  - Galerkin basis selection game
      mesh_refinement.py  - Adaptive mesh refinement game
  poc/          - PoC scenario framework
    config.py         - Pydantic configuration schemas
    registry.py       - Scenario registration and discovery
    runner.py         - Scenario execution engine
    results.py        - Result collection and persistence
    logging.py        - Structured logging utilities
    cli.py            - CLI entry point
    scenarios/        - Built-in scenario implementations
      transfer.py     - Zero-shot transfer scenario
      complexity.py   - O(N) complexity benchmark
      stability.py    - LBB stability monitoring
    tuning/           - Hyperparameter tuning
      config.py       - Tuning configuration schemas
      sampler.py      - Parameter samplers (TPE, grid, random)
      tuner.py        - HyperparameterTuner orchestrator
    statistics/       - Statistical analysis
      significance.py - Significance testing & effect sizes
  templates/    - Reusable module development infrastructure
    config.py         - Base Pydantic configuration classes
    registry.py       - Thread-safe singleton registry pattern
    logging.py        - Structured logging with context binding
    base.py           - Base executable classes with result tracking
    cli.py            - CLI utilities with common options
  video_compression/  - Neural video compression system
    config.py         - Pydantic configuration schemas
    models/           - Neural network models
      encoder.py      - Analysis transform with FNet + Galerkin
      decoder.py      - Synthesis transform
      hyperprior.py   - Scale hyperprior entropy model
      quantizer.py    - Differentiable quantization
    codec/            - Codec implementation
      codec.py        - Complete encode/decode pipeline
      entropy_coder.py - Range encoder/decoder
      gop_manager.py  - GOP and reference management
    mcts/             - MCTS rate control
      networks.py     - Policy, value, dynamics networks
      rate_control.py - MCTS-based rate controller
    metrics/          - Quality metrics
      quality.py      - PSNR, SSIM, MS-SSIM
      rd_curves.py    - BD-rate computation
    training/         - Training utilities
      loss.py         - R-D loss functions
      trainer.py      - Compression trainer
tests/
  math_kernel/  - Property-based tests for mathematical operators
    test_fredholm.py  - Fredholm integral equation tests
  training/     - Tests for training infrastructure
  integration/  - End-to-end integration tests
  poc/          - PoC framework tests
    test_config.py    - Configuration validation tests
    test_registry.py  - Scenario registration tests
    test_runner.py    - Runner execution tests
    test_results.py   - Result collection tests
  distributed/  - Distributed training tests
    test_config.py    - Config validation tests
  deployment/   - Deployment tests
    test_config.py    - Export/quantization config tests
  games/        - Multi-game tests
    test_go.py        - Go implementation tests
  pde/          - PDE framework tests
    test_config.py    - PDE configuration validation tests
    test_operators.py - PDE operator tests
  modeling/     - Neural network modeling tests
    test_multiscale_fourier.py - Multi-scale Fourier feature tests
  templates/    - Module development template tests
    test_config.py    - Configuration validation tests
    test_registry.py  - Registry pattern tests
    test_logging.py   - Logging utilities tests
    test_base.py      - Base executable tests
  video_compression/  - Video compression tests
    unit/             - Unit tests
      test_config.py    - Configuration validation
      test_encoder.py   - Encoder tests
      test_decoder.py   - Decoder tests
      test_quantizer.py - Quantizer tests
      test_metrics.py   - Quality metrics tests
config/         - Hydra/Pydantic configuration schemas
  train.yaml          - Default training config
  train_fast.yaml     - Fast test config
  scenarios/          - PoC scenario configurations
    poc_full.yaml     - Full PoC suite
    poc_quick.yaml    - Quick validation suite
    transfer_ablation.yaml - Transfer ablation study
docs/           - Documentation
  architecture/       - C4 architecture diagrams
    c4_model.md       - C4 model documentation
  templates/          - Module development templates
    IMPLEMENTATION_TEMPLATE.md - Agentic coding system prompt template
    C4_TEMPLATE.md    - C4 architecture template in Mermaid format
  IMPLEMENTATION_PLAN.md - Next-phase implementation plan
  PROMPT_TEMPLATE.md  - Agentic coding prompt template
scripts/        - CLI entry points
  train.py            - Training CLI with Hydra
  train_vertex.py     - Vertex AI training job launcher
  vertex_jobs.py      - Vertex AI job management CLI
  build_vertex_container.sh - Build and push training container
docker/         - Container definitions
  Dockerfile.vertex   - Vertex AI training container
```
