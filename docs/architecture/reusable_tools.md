# AlphaGalerkin Reusable Tools Reference

> **Last Updated:** 2026-04-10

The codebase contains modular tools and patterns designed for reuse across modules and in external projects.

---

## 1. Math Kernel Tools (`src/math_kernel/`)

### 1.1 Basis Functions (`basis.py`)

- **`FourierBasis(n_features, scale)`**: Drop-in positional encoding for any 2D vision transformer or NeRF-like model. Supports PyTorch and JAX backends.
  ```python
  basis = FourierBasis(n_features=64, scale=1.0)
  features = basis(coords)  # Shape: (batch, n, 128)
  ```

- **`ChebyshevBasis(max_degree)`**: Alternative basis using Chebyshev polynomials with optimal approximation properties.

- **`create_grid_coordinates(board_size)`**: Generates normalized, cell-centered coordinates for any grid-based problem.

### 1.2 Spectral Utilities (`spectral.py`)

- **`ResolutionAdapter`**: Transfers weights/signals across resolutions with anti-aliasing filtering.
- **`SpectralFilter`**: Frequency-domain filtering for resolution transfer.

---

## 2. Registry Pattern (`src/templates/registry.py`)

Thread-safe singleton registries used across the entire codebase.

- **`create_registry(name, base_class)`**: Factory function creating a `Registry` class + `@register` decorator pair. Used by `GameRegistry`, `PDEOperatorRegistry`, `LossRegistry`, `AgentRegistry`, `EngineRegistry`, `ScenarioRegistry`.

```python
from src.templates.registry import create_registry

ProcessorRegistry, register_processor = create_registry('Processor', BaseProcessor)

@register_processor('upper')
class UpperProcessor(BaseProcessor):
    def process(self, data): return data.upper()

proc = ProcessorRegistry().get('upper')()
```

---

## 3. Pydantic Config Pattern (`src/templates/config.py`)

Base configuration classes with validation, hashing, and factory creation.

- **`BaseModuleConfig`**: Base class with `compute_hash()`, `to_dict()`, built-in `name` and `description` fields.
- **`create_config_class(name, **fields)`**: Factory for quick Pydantic config creation.

```python
from src.templates.config import BaseModuleConfig, create_config_class
from pydantic import Field

class MyConfig(BaseModuleConfig):
    my_param: int = Field(default=100, ge=1, description='My parameter')

# Or use the factory:
QuickConfig = create_config_class('QuickConfig', my_float=(float, Field(default=0.5, gt=0, lt=1)))
```

---

## 4. Structured Logging (`src/templates/logging.py`)

Context-aware structured logging with timing and metric support.

- **`create_logger_class(module_name)`**: Creates a logger class with context binding, timed operations, and metric logging.

```python
from src.templates.logging import create_logger_class, configure_module_logging

configure_module_logging(level='DEBUG')
MyLogger = create_logger_class('MyModule')
logger = MyLogger('component', run_id='test123')

with logger.timed('operation'):
    logger.metric('accuracy', 0.95, epoch=1)
```

---

## 5. Loss Balancing (`src/training/loss_balancing.py`)

Adaptive multi-objective loss balancing — reusable for any multi-task training setup.

- **`ReLoBRaLo`**: Relative Loss Balancing with Random Lookback.
- **`GradNorm`**: Gradient normalization for multi-task learning.
- **`UncertaintyWeighting`**: Learnable log-variance parameters.
- **`SoftAdapt`**: Rate-based adaptation for improving slower losses.
- **`create_loss_balancer(config, loss_names)`**: Factory function.

```python
from src.training.loss_balancing import create_loss_balancer, LossBalancingConfig, BalancingStrategy

config = LossBalancingConfig(name='test', strategy=BalancingStrategy.RELOBRALO)
balancer = create_loss_balancer(config, ['policy', 'value', 'physics'])
result = balancer.compute_weighted_loss(losses)
```

---

## 6. PDE Operators (`src/pde/operators.py`)

Reusable PDE operator definitions with automatic differentiation — applicable to any PDE benchmark.

- **`PoissonOperator`**: Standard Poisson equation with configurable source terms.
- **`BurgersOperator`**: 1D Burgers equation with Cole-Hopf exact solution and configurable shock parameters.
- **`NavierStokesOperator`**: Taylor-Green vortex benchmark with analytical solution.
- **`AdvectionDiffusionOperator`**: Advection-diffusion with configurable coefficients.
- **`LShapedPoissonOperator`**: r^(2/3)*sin(2theta/3) singularity for AMR benchmarking.

All operators provide `generate_collocation_points()`, `source_term()`, and `exact_solution()` methods.

---

## 7. Domain Geometry (`src/pde/geometry.py`)

Reusable domain abstractions for PDE solving.

- **`RectangularDomain`**: Standard rectangular domain.
- **`LShapedDomain`**: Non-convex L-shaped domain with rejection sampling.
- **`CylinderFlowDomain`**: DFG benchmark cylinder flow domain.
- **`create_geometry(config)`**: Factory function from `GeometryConfig`.

---

## 8. Time Stepping (`src/pde/time_stepping.py`)

Numerical time integration methods with factory pattern.

- **`ForwardEuler`**: First-order explicit method.
- **`RK4`**: Fourth-order Runge-Kutta.
- **`CrankNicolson`**: Second-order implicit (fixed-point iteration).
- **`integrate(method, f, u0, t_span, dt)`**: Unified interface with snapshot saving.

---

## 9. CLI Tools (`src/tools/`)

### 9.1 Verify Invariance (`verify_invariance.py`)

- **Command**: `alphagalerkin verify --train-size 9 --infer-size 19`
- **Purpose**: Mathematically verify resolution-independent model output consistency.
- **Reusability**: Verification logic adaptable for any operator learning model.

### 9.2 GTP Engine (`gtp.py`)

- **Command**: `alphagalerkin gtp`
- **Purpose**: Go Text Protocol implementation compatible with Sabaki/GoGui/KaTrain.
- **Reusability**: Wraps any `f(board) -> move` function into a GTP-compatible engine.

---

## 10. Modeling Modules (`src/modeling/`)

### 10.1 StabilityGuard (`stability.py`)

- **Purpose**: Monitors LBB inf-sup condition for stability in Galerkin Transformers.
- **Usage**: Attach to any attention mechanism to log minimum singular value.
- **Why**: Essential for debugging mode collapse or instability.

### 10.2 FNetBlock (`fnet.py`)

- **Purpose**: O(N log N) mixing using `torch.fft.rfft2`.
- **Usage**: Drop-in replacement for self-attention when speed is critical and global mixing is sufficient.

### 10.3 Multi-Scale Fourier Features (`multiscale_fourier.py`)

- **`MultiScaleFourierFeatures`**: Multiple frequency bands to overcome spectral bias.
- **`AdaptiveFourierFeatures`**: Attention-based frequency selection.
- **`ProgressiveFourierFeatures`**: Curriculum-based frequency introduction.

---

## 11. Configuration (`config/`)

- **Pydantic schemas** in `config/schemas.py` are templates for any research project requiring strict experiment configuration.
- **Hydra YAML** configs in `config/` demonstrate override patterns for CLI-driven experiments.
