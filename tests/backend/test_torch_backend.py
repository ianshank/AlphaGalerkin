"""PyTorch-specific backend tests.

Tests autograd wrappers (grad, value_and_grad, has_aux), device
placement, cuDNN configuration propagation, and seed reproducibility.
"""

from __future__ import annotations

import numpy.testing as npt
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


@pytest.fixture
def tb():
    """Provide a TorchBackend instance for testing."""
    from src.backend import get_backend

    return get_backend("torch")


# ------------------------------------------------------------------
# Autograd: grad
# ------------------------------------------------------------------


class TestTorchGrad:
    """Test functional gradient computation via backend.grad."""

    def test_grad_simple(self, tb) -> None:
        """grad(f)(x) should compute df/dx for f(x) = sum(x^2)."""

        def f(x):
            return (x**2).sum()

        x = tb.tensor([3.0, 4.0])
        grad_fn = tb.grad(f)
        g = grad_fn(x)
        expected = torch.tensor([6.0, 8.0])
        torch.testing.assert_close(g, expected)

    def test_grad_with_aux(self, tb) -> None:
        """Grad with has_aux=True should return (grads, aux)."""

        def f(x):
            loss = (x**2).sum()
            aux = {"intermediate": x.mean()}
            return loss, aux

        x = tb.tensor([3.0, 4.0])
        grad_fn = tb.grad(f, has_aux=True)
        g, aux = grad_fn(x)
        expected = torch.tensor([6.0, 8.0])
        torch.testing.assert_close(g, expected)
        assert "intermediate" in aux

    def test_grad_does_not_mutate_input(self, tb) -> None:
        """Grad should not modify the original tensor."""
        x = tb.tensor([1.0, 2.0])
        original = x.clone()

        def f(t):
            return (t**2).sum()

        grad_fn = tb.grad(f)
        grad_fn(x)
        torch.testing.assert_close(x, original)

    def test_grad_multiarg(self, tb) -> None:
        """Grad with argnums tuple should return multiple gradients."""

        def f(x, y):
            return (x * y).sum()

        x = tb.tensor([2.0, 3.0])
        y = tb.tensor([4.0, 5.0])
        grad_fn = tb.grad(f, argnums=(0, 1))
        gx, gy = grad_fn(x, y)
        # d(x*y)/dx = y, d(x*y)/dy = x
        torch.testing.assert_close(gx, y)
        torch.testing.assert_close(gy, x)


# ------------------------------------------------------------------
# Autograd: value_and_grad
# ------------------------------------------------------------------


class TestTorchValueAndGrad:
    """Test functional value_and_grad computation."""

    def test_value_and_grad(self, tb) -> None:
        """value_and_grad should return (value, grads)."""

        def f(x):
            return (x**2).sum()

        x = tb.tensor([3.0, 4.0])
        vag_fn = tb.value_and_grad(f)
        val, g = vag_fn(x)
        npt.assert_allclose(val.item(), 25.0, rtol=1e-5)
        expected = torch.tensor([6.0, 8.0])
        torch.testing.assert_close(g, expected)

    def test_value_and_grad_with_aux(self, tb) -> None:
        """value_and_grad with has_aux should return ((value, aux), grads)."""

        def f(x):
            loss = (x**2).sum()
            return loss, {"tag": "test"}

        x = tb.tensor([3.0, 4.0])
        vag_fn = tb.value_and_grad(f, has_aux=True)
        (val, aux), g = vag_fn(x)
        npt.assert_allclose(val.item(), 25.0, rtol=1e-5)
        assert aux["tag"] == "test"
        expected = torch.tensor([6.0, 8.0])
        torch.testing.assert_close(g, expected)


# ------------------------------------------------------------------
# Device placement
# ------------------------------------------------------------------


class TestTorchDevice:
    """Test device management."""

    def test_default_device_is_string(self, tb) -> None:
        device = tb.get_default_device()
        assert device in ("cpu", "cuda")

    def test_tensors_on_default_device(self, tb) -> None:
        x = tb.zeros((2, 3))
        assert str(x.device).startswith(tb.get_default_device())

    def test_to_device_cpu(self, tb) -> None:
        x = tb.ones((3,))
        y = tb.to_device(x, "cpu")
        assert str(y.device) == "cpu"


# ------------------------------------------------------------------
# cuDNN config propagation
# ------------------------------------------------------------------


class TestCuDNNConfig:
    """Test that cuDNN settings from BackendConfig propagate."""

    @pytest.fixture(autouse=True)
    def _restore_cudnn_state(self):
        """Save and restore cuDNN global state around each test."""
        orig_benchmark = torch.backends.cudnn.benchmark
        orig_deterministic = torch.backends.cudnn.deterministic
        yield
        torch.backends.cudnn.benchmark = orig_benchmark
        torch.backends.cudnn.deterministic = orig_deterministic
        torch.use_deterministic_algorithms(False)

    def test_cudnn_benchmark_default(self) -> None:
        """Default config should enable cuDNN benchmark mode."""
        from src.backend.config import BackendConfig

        config = BackendConfig(torch_cudnn_benchmark=True, torch_deterministic=False)

        from src.backend.torch_backend import TorchBackend

        TorchBackend(config)
        assert torch.backends.cudnn.benchmark is True

    def test_cudnn_benchmark_disabled(self) -> None:
        """Config with benchmark=False should disable it."""
        from src.backend.config import BackendConfig

        config = BackendConfig(torch_cudnn_benchmark=False, torch_deterministic=False)

        from src.backend.torch_backend import TorchBackend

        TorchBackend(config)
        assert torch.backends.cudnn.benchmark is False

    def test_deterministic_mode(self) -> None:
        """Config with deterministic=True should enable deterministic mode."""
        from src.backend.config import BackendConfig

        config = BackendConfig(torch_cudnn_benchmark=False, torch_deterministic=True)

        from src.backend.torch_backend import TorchBackend

        TorchBackend(config)
        assert torch.backends.cudnn.deterministic is True


# ------------------------------------------------------------------
# Seed reproducibility
# ------------------------------------------------------------------


class TestTorchSeedReproducibility:
    """Test reproducibility with seeded generator."""

    def test_randn_reproducible(self, tb) -> None:
        tb.set_seed(42)
        a = tb.to_numpy(tb.randn((10,)))
        tb.set_seed(42)
        b = tb.to_numpy(tb.randn((10,)))
        npt.assert_allclose(a, b)

    def test_rand_reproducible(self, tb) -> None:
        tb.set_seed(99)
        a = tb.to_numpy(tb.rand((10,)))
        tb.set_seed(99)
        b = tb.to_numpy(tb.rand((10,)))
        npt.assert_allclose(a, b)
