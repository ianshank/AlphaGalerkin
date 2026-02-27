"""Cross-backend equivalence tests for math_kernel.

These tests verify that PyTorch and JAX implementations produce
numerically equivalent results for the same inputs. They are skipped
when either backend is unavailable.

Marked with ``cross_backend`` pytest marker.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

both_backends = pytest.mark.skipif(
    not (HAS_TORCH and HAS_JAX), reason="both torch and jax required"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    """Convert any tensor to numpy array."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if HAS_JAX:
        return np.asarray(jax.device_get(x))
    return np.asarray(x)


# ------------------------------------------------------------------
# Monte Carlo Integral
# ------------------------------------------------------------------


@both_backends
@pytest.mark.cross_backend
class TestMonteCarloIntegralEquivalence:
    """Verify torch and JAX MonteCarloIntegral produce same results."""

    def test_uniform_integration_equivalence(self):
        from src.math_kernel.integral import (
            create_monte_carlo_integral,
        )

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((2, 8, 4)).astype(np.float32)

        # Torch path
        torch_mc = create_monte_carlo_integral(backend="torch")
        torch_input = torch.from_numpy(data)
        torch_out = torch_mc(torch_input)

        # JAX path
        jax_mc = create_monte_carlo_integral(backend="jax")
        jax_input = jnp.array(data)
        params = jax_mc.init(jax.random.PRNGKey(0), jax_input)
        jax_out = jax_mc.apply(params, jax_input)

        np.testing.assert_allclose(
            _to_numpy(torch_out),
            _to_numpy(jax_out),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_weighted_integration_equivalence(self):
        from src.math_kernel.integral import (
            create_monte_carlo_integral,
        )

        np_rng = np.random.default_rng(123)
        data = np_rng.standard_normal((2, 8, 4)).astype(np.float32)
        weights = np_rng.uniform(0.1, 1.0, (2, 8)).astype(np.float32)

        # Torch path
        torch_mc = create_monte_carlo_integral(backend="torch")
        torch_out = torch_mc(torch.from_numpy(data), torch.from_numpy(weights))

        # JAX path
        jax_mc = create_monte_carlo_integral(backend="jax")
        jax_input = jnp.array(data)
        jax_weights = jnp.array(weights)
        params = jax_mc.init(jax.random.PRNGKey(0), jax_input, jax_weights)
        jax_out = jax_mc.apply(params, jax_input, jax_weights)

        np.testing.assert_allclose(
            _to_numpy(torch_out),
            _to_numpy(jax_out),
            atol=1e-5,
            rtol=1e-5,
        )


# ------------------------------------------------------------------
# Galerkin Projection
# ------------------------------------------------------------------


@both_backends
@pytest.mark.cross_backend
class TestGalerkinProjectionEquivalence:
    """Verify torch and JAX GalerkinProjection produce same output shapes."""

    def test_output_shape_matches(self):
        from src.math_kernel.integral import (
            create_galerkin_projection,
        )

        d_model, d_key, d_value = 16, 8, 8
        batch, n = 2, 10

        torch_proj = create_galerkin_projection(d_model, d_key, d_value, backend="torch")
        jax_proj = create_galerkin_projection(d_model, d_key, d_value, backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, n, d_model)).astype(np.float32)

        # Torch
        torch_out = torch_proj(torch.from_numpy(data))
        assert torch_out.shape == (batch, n, d_model)

        # JAX
        jax_input = jnp.array(data)
        params = jax_proj.init(jax.random.PRNGKey(0), jax_input)
        jax_out = jax_proj.apply(params, jax_input)
        assert jax_out.shape == (batch, n, d_model)

    def test_lbb_constant_positive_both_backends(self):
        from src.math_kernel.integral import (
            create_galerkin_projection,
        )

        d_model, d_key, d_value = 16, 8, 8
        batch, n = 2, 10

        torch_proj = create_galerkin_projection(d_model, d_key, d_value, backend="torch")
        jax_proj = create_galerkin_projection(d_model, d_key, d_value, backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, n, d_model)).astype(np.float32)

        # Torch LBB
        torch_lbb = torch_proj.compute_lbb_constant(torch.from_numpy(data))
        assert (_to_numpy(torch_lbb) > 0).all()

        # JAX LBB
        jax_input = jnp.array(data)
        variables = jax_proj.init(jax.random.PRNGKey(0), jax_input)
        jax_lbb = jax_proj.compute_lbb_constant(variables, jax_input)
        assert (_to_numpy(jax_lbb) > 0).all()


# ------------------------------------------------------------------
# Petrov-Galerkin Projection
# ------------------------------------------------------------------


@both_backends
@pytest.mark.cross_backend
class TestPetrovGalerkinProjectionEquivalence:
    """Verify torch and JAX PetrovGalerkinProjection consistency."""

    def test_lbb_dimension_check_both_backends(self):
        from src.math_kernel.integral import (
            create_petrov_galerkin_projection,
        )

        # Both backends should reject d_trial < d_test
        with pytest.raises(ValueError, match="LBB"):
            create_petrov_galerkin_projection(16, d_trial=4, d_test=8, d_value=8, backend="torch")

        # JAX raises on init (setup), so we need to trigger it
        jax_pg = create_petrov_galerkin_projection(
            16, d_trial=4, d_test=8, d_value=8, backend="jax"
        )
        with pytest.raises(ValueError, match="LBB"):
            dummy = jnp.ones((1, 5, 16))
            jax_pg.init(jax.random.PRNGKey(0), dummy)

    def test_valid_dimensions_both_backends(self):
        from src.math_kernel.integral import (
            create_petrov_galerkin_projection,
        )

        d_model, d_trial, d_test, d_value = 16, 8, 4, 8
        batch, n = 2, 10

        torch_pg = create_petrov_galerkin_projection(
            d_model, d_trial, d_test, d_value, backend="torch"
        )
        jax_pg = create_petrov_galerkin_projection(d_model, d_trial, d_test, d_value, backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, n, d_model)).astype(np.float32)

        # Torch
        torch_out = torch_pg(torch.from_numpy(data))
        assert torch_out.shape == (batch, n, d_model)

        # JAX
        jax_input = jnp.array(data)
        params = jax_pg.init(jax.random.PRNGKey(0), jax_input)
        jax_out = jax_pg.apply(params, jax_input)
        assert jax_out.shape == (batch, n, d_model)


# ------------------------------------------------------------------
# Spectral Filter
# ------------------------------------------------------------------


@both_backends
@pytest.mark.cross_backend
class TestSpectralFilterEquivalence:
    """Verify torch and JAX SpectralFilter produce equivalent results."""

    @pytest.mark.parametrize("filter_type", ["gaussian", "butterworth", "ideal"])
    def test_filter_output_shape(self, filter_type):
        from src.math_kernel.spectral import create_spectral_filter

        batch, channels, h, w = 2, 3, 8, 8

        torch_filt = create_spectral_filter(
            cutoff_ratio=0.5, filter_type=filter_type, backend="torch"
        )
        jax_filt = create_spectral_filter(cutoff_ratio=0.5, filter_type=filter_type, backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, channels, h, w)).astype(np.float32)

        # Torch
        torch_out = torch_filt(torch.from_numpy(data))
        assert torch_out.shape == (batch, channels, h, w)

        # JAX
        jax_input = jnp.array(data)
        params = jax_filt.init(jax.random.PRNGKey(0), jax_input)
        jax_out = jax_filt.apply(params, jax_input)
        assert jax_out.shape == (batch, channels, h, w)

    @pytest.mark.parametrize("filter_type", ["gaussian", "butterworth", "ideal"])
    def test_filter_values_equivalent(self, filter_type):
        from src.math_kernel.spectral import create_spectral_filter

        batch, channels, h, w = 1, 1, 8, 8

        torch_filt = create_spectral_filter(
            cutoff_ratio=0.5, filter_type=filter_type, backend="torch"
        )
        jax_filt = create_spectral_filter(cutoff_ratio=0.5, filter_type=filter_type, backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, channels, h, w)).astype(np.float32)

        torch_out = _to_numpy(torch_filt(torch.from_numpy(data)))

        jax_input = jnp.array(data)
        params = jax_filt.init(jax.random.PRNGKey(0), jax_input)
        jax_out = _to_numpy(jax_filt.apply(params, jax_input))

        np.testing.assert_allclose(torch_out, jax_out, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------
# Resolution Adapter
# ------------------------------------------------------------------


@both_backends
@pytest.mark.cross_backend
class TestResolutionAdapterEquivalence:
    """Verify torch and JAX ResolutionAdapter produce equivalent results."""

    def test_same_resolution_identity(self):
        from src.math_kernel.spectral import create_resolution_adapter

        batch, d = 2, 8
        size = 4

        torch_adapter = create_resolution_adapter(backend="torch")
        jax_adapter = create_resolution_adapter(backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, size * size, d)).astype(np.float32)

        torch_out = _to_numpy(torch_adapter(torch.from_numpy(data), size, size))

        jax_input = jnp.array(data)
        params = jax_adapter.init(jax.random.PRNGKey(0), jax_input, size, size)
        jax_out = _to_numpy(jax_adapter.apply(params, jax_input, size, size))

        np.testing.assert_allclose(torch_out, jax_out, atol=1e-5, rtol=1e-5)

    def test_upsampling_shape_matches(self):
        from src.math_kernel.spectral import create_resolution_adapter

        batch, d = 2, 8
        source, target = 4, 8

        torch_adapter = create_resolution_adapter(backend="torch")
        jax_adapter = create_resolution_adapter(backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, source * source, d)).astype(np.float32)

        torch_out = torch_adapter(torch.from_numpy(data), source, target)
        assert torch_out.shape == (batch, target * target, d)

        jax_input = jnp.array(data)
        params = jax_adapter.init(jax.random.PRNGKey(0), jax_input, source, target)
        jax_out = jax_adapter.apply(params, jax_input, source, target)
        assert jax_out.shape == (batch, target * target, d)

    def test_downsampling_shape_matches(self):
        from src.math_kernel.spectral import create_resolution_adapter

        batch, d = 2, 8
        source, target = 8, 4

        torch_adapter = create_resolution_adapter(backend="torch")
        jax_adapter = create_resolution_adapter(backend="jax")

        np_rng = np.random.default_rng(42)
        data = np_rng.standard_normal((batch, source * source, d)).astype(np.float32)

        torch_out = torch_adapter(torch.from_numpy(data), source, target)
        assert torch_out.shape == (batch, target * target, d)

        jax_input = jnp.array(data)
        params = jax_adapter.init(jax.random.PRNGKey(0), jax_input, source, target)
        jax_out = jax_adapter.apply(params, jax_input, source, target)
        assert jax_out.shape == (batch, target * target, d)


# ------------------------------------------------------------------
# Factory function validation
# ------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestFactoryFunctions:
    """Verify factory functions reject invalid backends consistently.

    These tests only need one backend (torch) since they test error paths.
    """

    def test_monte_carlo_invalid_backend(self):
        from src.math_kernel.integral import create_monte_carlo_integral

        with pytest.raises(ValueError, match="Unknown backend"):
            create_monte_carlo_integral(backend="numpy")

    def test_galerkin_invalid_backend(self):
        from src.math_kernel.integral import create_galerkin_projection

        with pytest.raises(ValueError, match="Unknown backend"):
            create_galerkin_projection(16, 8, 8, backend="numpy")

    def test_spectral_filter_invalid_backend(self):
        from src.math_kernel.spectral import create_spectral_filter

        with pytest.raises(ValueError, match="Unknown backend"):
            create_spectral_filter(backend="numpy")

    def test_resolution_adapter_invalid_backend(self):
        from src.math_kernel.spectral import create_resolution_adapter

        with pytest.raises(ValueError, match="Unknown backend"):
            create_resolution_adapter(backend="numpy")

    def test_torch_factory_creates_torch_classes(self):
        from src.math_kernel.integral import (
            GalerkinProjection,
            MonteCarloIntegral,
            create_galerkin_projection,
            create_monte_carlo_integral,
        )
        from src.math_kernel.spectral import SpectralFilter, create_spectral_filter

        assert isinstance(create_monte_carlo_integral(backend="torch"), MonteCarloIntegral)
        assert isinstance(create_galerkin_projection(16, 8, 8, backend="torch"), GalerkinProjection)
        assert isinstance(create_spectral_filter(backend="torch"), SpectralFilter)
