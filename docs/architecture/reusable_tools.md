# AlphaGalerkin Reusable Tools Reference

The codebase contains several modular tools designed for research and experimentation beyond the core Go logic.

## 1. Math Kernel Tools

These tools in `src/math_kernel/` can be used for any problem involving continuous operator learning or spectral methods.

### 1.1 Basis Functions (`basis.py`)

- **`FourierBasis(n_features, scale)`**:
  - *Usage*: Can be used as a drop-in positional encoding for any 2D vision transformer or NeRF-like model.
  - *Example*:

    ```python
    basis = FourierBasis(n_features=64, scale=1.0)
    features = basis(coords)  # Shape: (batch, n, 128)
    ```

- **`create_grid_coordinates(board_size)`**:
  - *Usage*: Generates normalized coordinates for any grid-based problem (e.g., fluid dynamics, heat equation on a square).

### 1.2 Spectral Utilities (`spectral.py`)

- **`ResolutionAdapter`**:
  - *Usage*: Provides logic for transferring weights or interpreting signals across different resolutions.
  - *Key Feature*: Anti-aliasing filtering when downscaling.

## 2. CLI Tools (`src/tools/`)

### 2.1 Verify Invariance (`verify_invariance.py`)

- **Command**: `alphagalerkin verify --train-size 9 --infer-size 19`
- **Purpose**: A standalone script to mathematically verify that the model's output is consistent across resolutions.
- **Reusability**: The verification logic (comparing properties of integrals) can be adapted for any operator learning model.

### 2.2 GTP Engine (`gtp.py`)

- **Command**: `alphagalerkin gtp`
- **Purpose**: Implements the Go Text Protocol (GTP).
- **Reusability**: Can be wrapped around *any* function `f(board) -> move` to create a Go engine compatible with Sabaki/Lizzie.

## 3. Modeling Modules (`src/modeling/`)

### 3.1 `StabilityGuard` (`stability.py`)

- **Purpose**: Monitors the LBB (Ladyzhenskaya-Babuska-Brezzi) condition for stability in mixed finite element methods.
- **Usage**: Can be attached to any attention mechanism to log the minimum singular value of the projection matrix.
- **Why use it?**: Essential for debugging "mode collapse" or instability in Galerkin Transformers.

### 3.2 `FNetBlock` (`fnet.py`)

- **Purpose**: Extremely fast mixing using FFTs ($O(N \log N)$).
- **Usage**: A drop-in replacement for Self-Attention when speed is critical and global mixing is sufficient (no local pairwise interactions needed).

## 4. Configuration (`config/`)

- **Schemas**: The Pydantic models in `config/schemas.py` are excellent templates for any research project requiring strict experiment configuration.
