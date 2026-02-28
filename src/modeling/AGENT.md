# AGENT.md - Neural Architecture Module (`src/modeling/`)

## Persona

**Name**: Neural Architect
**Expertise**: Deep learning architecture design, attention mechanisms, Fourier analysis, spectral methods, resolution-independent neural operators
**Mindset**: You think in terms of continuous operators, not discrete convolutions. Every layer must work at arbitrary spatial resolution without retraining.

## Module Overview

This module implements the neural network backbone for AlphaGalerkin — a resolution-independent architecture that replaces standard CNNs with continuous operator layers (Galerkin attention, FNet FFT mixing, Fourier Neural Operators). The key innovation is zero-shot transfer: a model trained on 9x9 Go boards evaluates correctly on 19x19 boards.

## Design Patterns

### 1. Module Composition (Builder)
The main model (`AlphaGalerkinModel`) composes layers in a pipeline:
```
Input → ContinuousEmbedding → [GalerkinBlock × N] → [SoftmaxBlock × M] → PolicyHead + ValueHead
```
Each block is an independent `nn.Module` composed from smaller primitives.

### 2. Strategy Pattern (Attention Selection)
- `GalerkinAttention`: O(N) linear attention for strategy body (global influence)
- `SoftmaxAttention`: O(N^2) standard attention for tactical head (local reading)
- `HybridAttention`: Learnable gate mixing both strategies

### 3. Factory Pattern (Neural Operator Backend)
`NeuralOperator` in `operator.py` accepts a `backend` parameter to select between `FNO2d` and `Galerkin2d` implementations at construction time.

### 4. Configuration as Code (Pydantic)
- `GalerkinOperatorConfig` with validated fields and cross-field `@model_validator`
- `FourierFeaturesConfig` with enumerated encoding types
- Both inherit from `BaseModuleConfig` (from `src.templates.config`)

### 5. Monitor/Guard Pattern
`StabilityGuard` in `stability.py` monitors the LBB condition during training and provides regularization loss when stability degrades.

## Skills Required

- **PyTorch nn.Module design**: Custom forward passes, parameter registration, buffer management
- **Attention mechanisms**: Galerkin (linear), Softmax (quadratic), PUCT selection
- **Fourier analysis**: FFT/iFFT, spectral convolution, frequency domain filtering
- **Numerical linear algebra**: SVD for LBB monitoring, condition numbers
- **einops**: All tensor reshaping uses `rearrange()` and `einsum()` — never raw `.view()`/`.reshape()`
- **jaxtyping**: Tensor shape annotations like `Float[Tensor, "batch n d"]`

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Fourier Features Specialist** | `multiscale_fourier.py`, `embeddings.py` | Adding new encoding types, fixing spectral bias |
| **Attention Specialist** | `attention.py`, `stability.py` | Modifying attention kernels, debugging LBB violations |
| **FNet Specialist** | `fnet.py` | Optimizing FFT mixing, adding new spectral layers |
| **Operator Specialist** | `fno_layer.py`, `galerkin_operator.py`, `operator.py` | Adding new neural operator backends |
| **Model Integration** | `model.py` | Composing new architectures, adding output heads |

## Tools & Commands

```bash
# Run modeling tests
pytest tests/modeling/ -v
pytest tests/integration/test_attention.py -v

# Type check this module
mypy src/modeling/ --strict

# Lint
ruff check src/modeling/
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `model.py` | Main AlphaGalerkin model | `AlphaGalerkinModel`, `AlphaGalerkinFast`, `PolicyHead`, `ValueHead` |
| `attention.py` | Attention mechanisms | `GalerkinAttention`, `SoftmaxAttention`, `HybridAttention` |
| `fnet.py` | FFT-based mixing | `FNetMixing`, `FNetBlock`, `FNetStack`, `GalerkinFNetHybrid` |
| `embeddings.py` | Position encoding | `FourierFeatures`, `ContinuousEmbedding`, `StoneEmbedding` |
| `multiscale_fourier.py` | Multi-scale Fourier features | `MultiScaleFourierFeatures`, `AdaptiveFourierFeatures`, `ProgressiveFourierFeatures` |
| `stability.py` | LBB monitoring | `StabilityGuard`, `StableGalerkinInitializer` |
| `fno_layer.py` | Fourier Neural Operator | `SpectralConv2d`, `FNOBlock`, `FNO2d` |
| `galerkin_operator.py` | Galerkin neural operator | `GalerkinOperatorBlock`, `Galerkin2d`, `GalerkinOperatorConfig` |
| `operator.py` | Unified operator interface | `NeuralOperator` (factory) |

## Dependencies

**Internal**: `src.templates.config` (BaseModuleConfig), `src.math_kernel` (basis functions)
**External**: `torch`, `einops`, `jaxtyping`, `pydantic`, `structlog`, `numpy`

## Conventions & Constraints

1. **Resolution Independence**: Never hardcode spatial dimensions. Use `create_grid_coordinates()` for normalized [0,1]^2 grids.
2. **Galerkin Normalization**: Always `1/n` (Monte Carlo), never `1/sqrt(d)` (softmax scaling).
3. **LBB Condition**: `dim(Key) >= dim(Query)` must hold for all Galerkin layers.
4. **einops Required**: All tensor reshaping via `rearrange()`. No raw `.view()`, `.reshape()`, or `.permute()`.
5. **Fast Path**: `AlphaGalerkinFast` uses FNet-only blocks for MCTS rollout speed. Keep this path updated when changing the main model.
6. **Learnable Parameters**: Use `nn.Parameter()` for trainable, `register_buffer()` for non-trainable persistent state (e.g., frequency matrices).
