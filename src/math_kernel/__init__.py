"""Mathematical kernels for Galerkin projection and integral approximation.

Supports both PyTorch and JAX backends. The original PyTorch classes are
always available. JAX-backed equivalents are conditionally available when
JAX/Flax are installed (see :data:`HAS_JAX`). Factory functions provide a
convenient way to select the right implementation based on a backend name.

Backward compatibility: all original imports continue to work unchanged.
"""

from src.math_kernel.basis import (
    BasisFunction,
    ChebyshevBasis,
    FourierBasis,
    HAS_JAX,
    create_chebyshev_basis,
    create_fourier_basis,
    create_grid_coordinates,
)
from src.math_kernel.integral import (
    GalerkinProjection,
    MonteCarloIntegral,
    PetrovGalerkinProjection,
    create_galerkin_projection,
    create_monte_carlo_integral,
    create_petrov_galerkin_projection,
)
from src.math_kernel.spectral import (
    ResolutionAdapter,
    SpectralFilter,
    create_resolution_adapter,
    create_spectral_filter,
)

__all__ = [
    # Protocols
    "BasisFunction",
    # PyTorch classes (always available)
    "FourierBasis",
    "ChebyshevBasis",
    "GalerkinProjection",
    "MonteCarloIntegral",
    "PetrovGalerkinProjection",
    "SpectralFilter",
    "ResolutionAdapter",
    # Utility functions
    "create_grid_coordinates",
    # Factory functions
    "create_fourier_basis",
    "create_chebyshev_basis",
    "create_monte_carlo_integral",
    "create_galerkin_projection",
    "create_petrov_galerkin_projection",
    "create_spectral_filter",
    "create_resolution_adapter",
    # Feature flag
    "HAS_JAX",
]

# Conditionally export JAX classes when JAX/Flax are installed
if HAS_JAX:
    from src.math_kernel.basis import (  # type: ignore[attr-defined]
        JaxChebyshevBasis,
        JaxFourierBasis,
        create_grid_coordinates_jax,
    )
    from src.math_kernel.integral import (  # type: ignore[attr-defined]
        JaxGalerkinProjection,
        JaxMonteCarloIntegral,
        JaxPetrovGalerkinProjection,
    )
    from src.math_kernel.spectral import (  # type: ignore[attr-defined]
        JaxResolutionAdapter,
        JaxSpectralFilter,
    )

    __all__ += [
        # JAX classes (only when JAX/Flax are installed)
        "JaxFourierBasis",
        "JaxChebyshevBasis",
        "JaxMonteCarloIntegral",
        "JaxGalerkinProjection",
        "JaxPetrovGalerkinProjection",
        "JaxSpectralFilter",
        "JaxResolutionAdapter",
        "create_grid_coordinates_jax",
    ]
