# AlphaGalerkin Components Reference

This document provides a detailed reference for the core components of the AlphaGalerkin system.

## 1. Math Kernel Components (`src/math_kernel/`)

These components handle the continuous domain mathematics that enable resolution independence.

### 1.1 `FourierBasis`

- **Path**: `src/math_kernel/basis.py`
- **Purpose**: Maps continuous coordinates $(x, y) \in [0, 1]^2$ to high-dimensional features.
- **Method**: Random Fourier Features (Tancik et al., 2020).
- **Key Args**:
  - `n_features`: Number of frequency pairs.
  - `scale`: Standard deviation of the random frequency matrix (controls spectral bias).
  - `learnable`: If `True`, frequencies can be tuned during training.

### 1.2 `ChebyshevBasis`

- **Path**: `src/math_kernel/basis.py`
- **Purpose**: Alternative basis using Chebyshev polynomials (optimal approximation theory properties).
- **Method**: Tensor product of 1D Chebyshev polynomials.
- **Key Args**:
  - `max_degree`: Maximum polynomial degree.

### 1.3 `grid_coordinates`

- **Path**: `src/math_kernel/basis.py`
- **Function**: `create_grid_coordinates`
- **Purpose**: Generates normalized, cell-centered coordinates for any board size.
- **Behavior**: Maps discrete integer indices to $[0, 1]^2$.

## 2. Modeling Components (`src/modeling/`)

These components build the neural network architecture.

### 2.1 `GalerkinAttention` (The Core Innovation)

- **Path**: `src/modeling/attention.py`
- **Purpose**: Global influence modeling with $O(N)$ complexity.
- **Mechanism**: Interprets attention as a Petrov-Galerkin projection.
  - Projects values onto a global "Context" using the Key basis.
  - Reconstructs locally using the Query basis.
- **Stability**: Monitors the LBB condition (inf-sup) via `_compute_lbb_constant`.
- **Normalization**: Uses $1/n$ (Monte Carlo) instead of $1/\sqrt{d}$.

### 2.2 `SoftmaxAttention`

- **Path**: `src/modeling/attention.py`
- **Purpose**: Precision tactical reading (life & death).
- **Mechanism**: Standard $O(N^2)$ scaled dot-product attention.
- **Use Case**: Used in the "Tactical Head" of the model.

### 2.3 `HybridAttention`

- **Path**: `src/modeling/attention.py`
- **Purpose**: Combines global and local specificities.
- **Mechanism**: Weighted sum of Galerkin and Softmax outputs, controlled by a learnable gate.

### 2.4 `AlphaGalerkinModel`

- **Path**: `src/modeling/model.py`
- **Purpose**: The main architecture container.
- **Flow**:
  1. **Embedding**: `ContinuousEmbedding` + Fourier Features.
  2. **Strategy Body**: Stack of `GalerkinBlock` (Global context).
  3. **Mixing**: Optional `FNetBlock` layers (Speed).
  4. **Tactical Head**: Stack of `SoftmaxBlock` (Local precision).
  5. **Heads**: Policy and Value outputs.

## 3. Configuration Components (`config/`)

### 3.1 `OperatorConfig`

- **Path**: `config/schemas.py`
- **Purpose**: Defines model topology without hardcoded sizes.
- **Key Params**: `d_model`, `n_galerkin_layers`, `fourier_scale`.

### 3.2 `ResolutionAdapter`

- **Path**: `src/math_kernel/spectral.py`
- **Purpose**: Handles the transfer of weights/state when moving between resolutions (e.g., 9x9 -> 19x19).
