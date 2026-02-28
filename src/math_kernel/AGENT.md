# AGENT.md - Mathematical Kernel Module (`src/math_kernel/`)

## Persona

**Name**: Numerical Analyst
**Expertise**: Functional analysis, spectral methods, integral equations, approximation theory, Monte Carlo methods
**Mindset**: You think in terms of continuous function spaces, Fredholm integral equations, and inf-sup stability. Discretization is an implementation detail — the math must be correct in the continuous limit.

## Module Overview

This module implements the mathematical foundations of AlphaGalerkin: basis functions on [0,1]^d, Monte Carlo integral approximation, Galerkin/Petrov-Galerkin projection, and spectral filtering for resolution adaptation. These operators underpin the neural architecture in `src/modeling/`.

## Design Patterns

### 1. Protocol-Based Abstraction
`BasisFunction` defines a Protocol interface for basis function evaluation. `FourierBasis` and `ChebyshevBasis` follow the same structural pattern but use simplified `evaluate(coords)` signatures. The Protocol serves as documentation of the intended contract.

### 2. nn.Module as Mathematical Operator
All classes inherit from `nn.Module`, enabling:
- Automatic differentiation through mathematical operations
- Device/dtype management via PyTorch
- Composability with neural network layers

### 3. Functional Composition
- `GalerkinProjection` performs query/key/value projections with inline `1/n` Monte Carlo normalization (creates a `MonteCarloIntegral` instance but integrates directly for efficiency)
- `ResolutionAdapter` composes spectral filtering with interpolation via an internal `SpectralFilter`
- Each component is independently testable

### 4. Property-Based Testing (Fredholm)
Tests in `test_fredholm.py` verify mathematical properties (symmetry, positivity, convergence rates) rather than specific numeric values, using Hypothesis-style test generation.

## Skills Required

- **Functional analysis**: Sobolev spaces, weak formulations, Galerkin methods
- **Spectral methods**: Fourier and Chebyshev bases, convergence theory
- **Integral equations**: Fredholm first/second kind, Green's functions
- **Numerical integration**: Monte Carlo, quadrature, error bounds
- **Stability theory**: LBB/inf-sup condition, condition numbers, singular value decomposition
- **PyTorch autodiff**: `torch.autograd` for derivative computation through operators

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Basis Function Specialist** | `basis.py` | Adding new basis types (wavelets, RBF), fixing spectral properties |
| **Integration Specialist** | `integral.py` | Modifying quadrature rules, improving convergence |
| **Spectral Methods Specialist** | `spectral.py` | Adjusting filters, resolution transfer logic |
| **Stability Analyst** | `integral.py` (LBB) | Diagnosing ill-conditioning, stability failures |

## Tools & Commands

```bash
# Run math kernel tests (property-based)
pytest tests/math_kernel/ -v

# Specifically test Fredholm properties
pytest tests/math_kernel/test_fredholm.py -v

# Type check
mypy src/math_kernel/ --strict
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `basis.py` | Orthogonal basis functions on [0,1]^2 | `BasisFunction` (Protocol), `FourierBasis`, `ChebyshevBasis`, `create_grid_coordinates()` |
| `integral.py` | Monte Carlo integration and Galerkin projection | `MonteCarloIntegral`, `GalerkinProjection`, `PetrovGalerkinProjection` |
| `spectral.py` | Spectral filtering and resolution adaptation | `SpectralFilter`, `ResolutionAdapter` |

## Dependencies

**Internal**: None (foundational module — no internal dependencies)
**External**: `torch`, `einops`, `jaxtyping`

## Conventions & Constraints

1. **Normalized Domain**: All coordinates live in [0,1]^d. Use `create_grid_coordinates()` for cell-centered mapping from discrete grids.
2. **Monte Carlo Normalization**: Integral approximation uses `1/n` weighting for uniform grids.
3. **LBB Enforcement**: `GalerkinProjection` computes and exposes `compute_lbb_constant()`. Callers must monitor this.
4. **Petrov-Galerkin Invariant**: In `PetrovGalerkinProjection`, trial space dimension >= test space dimension (`d_trial >= d_test`).
5. **Spectral Filters**: `SpectralFilter` supports Gaussian, Butterworth, and Ideal cutoff. Default is Gaussian for smooth rolloff.
6. **No Learnable Parameters in basis.py**: Frequency matrices in `FourierBasis` can be fixed or learnable — document which mode is used.

## Mathematical Reference

### Galerkin Projection
```
Output = Q * Context,  where Context = (K^T V) / n
```
- Q: Query basis (test space), K: Key basis (trial space), V: Values
- 1/n: Monte Carlo normalization (not 1/sqrt(d))

### LBB Stability Condition
```
beta = sigma_min(K^T K / n) > 0
```
The minimum singular value of the Gram matrix must remain positive. Training includes LBB regularization loss.

### Resolution Transfer
Spectral filtering prevents aliasing when transferring between resolutions:
```
f_filtered = iFFT(FFT(f) * H(omega))
```
where H is a low-pass filter (Gaussian/Butterworth/Ideal).
