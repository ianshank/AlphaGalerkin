"""Tests for BackendConfig Pydantic validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.backend.config import BackendConfig
from src.backend.types import BackendType, DeviceType, Precision

# ------------------------------------------------------------------
# Default values
# ------------------------------------------------------------------


class TestBackendConfigDefaults:
    """Verify that default values are set correctly."""

    def test_default_backend(self) -> None:
        config = BackendConfig()
        assert config.backend == BackendType.TORCH

    def test_default_device(self) -> None:
        config = BackendConfig()
        assert config.device == DeviceType.AUTO

    def test_default_precision(self) -> None:
        config = BackendConfig()
        assert config.precision == Precision.FLOAT32

    def test_default_rng_seed(self) -> None:
        config = BackendConfig()
        assert config.rng_seed == 42

    def test_default_name(self) -> None:
        config = BackendConfig()
        assert config.name == "backend"

    def test_default_jax_jit_enabled(self) -> None:
        config = BackendConfig()
        assert config.jax_jit_enabled is True

    def test_default_jax_debug_nans(self) -> None:
        config = BackendConfig()
        assert config.jax_debug_nans is False

    def test_default_jax_log_compiles(self) -> None:
        config = BackendConfig()
        assert config.jax_log_compiles is False

    def test_default_jax_platform(self) -> None:
        config = BackendConfig()
        assert config.jax_platform is None

    def test_default_torch_cudnn_benchmark(self) -> None:
        config = BackendConfig()
        assert config.torch_cudnn_benchmark is True

    def test_default_torch_deterministic(self) -> None:
        config = BackendConfig()
        assert config.torch_deterministic is False


# ------------------------------------------------------------------
# Valid configurations
# ------------------------------------------------------------------


class TestBackendConfigValid:
    """Verify that valid configurations are accepted."""

    @pytest.mark.parametrize("backend", [BackendType.TORCH, BackendType.JAX])
    def test_valid_backend_enum(self, backend: BackendType) -> None:
        config = BackendConfig(backend=backend)
        assert config.backend == backend

    @pytest.mark.parametrize("backend_str", ["torch", "jax"])
    def test_valid_backend_string(self, backend_str: str) -> None:
        config = BackendConfig(backend=backend_str)
        assert config.backend == BackendType(backend_str)

    @pytest.mark.parametrize(
        "device", [DeviceType.CPU, DeviceType.GPU, DeviceType.TPU, DeviceType.AUTO]
    )
    def test_valid_device(self, device: DeviceType) -> None:
        config = BackendConfig(device=device)
        assert config.device == device

    @pytest.mark.parametrize(
        "precision",
        [Precision.FLOAT16, Precision.BFLOAT16, Precision.FLOAT32, Precision.FLOAT64],
    )
    def test_valid_precision(self, precision: Precision) -> None:
        config = BackendConfig(precision=precision)
        assert config.precision == precision

    @pytest.mark.parametrize("seed", [0, 1, 100, 999999])
    def test_valid_rng_seed(self, seed: int) -> None:
        config = BackendConfig(rng_seed=seed)
        assert config.rng_seed == seed

    @pytest.mark.parametrize("platform", [None, "cpu", "gpu", "tpu"])
    def test_valid_jax_platform(self, platform: str | None) -> None:
        config = BackendConfig(jax_platform=platform)
        assert config.jax_platform == platform

    def test_full_custom_config(self) -> None:
        config = BackendConfig(
            name="custom",
            backend=BackendType.JAX,
            device=DeviceType.GPU,
            precision=Precision.FLOAT64,
            rng_seed=123,
            jax_jit_enabled=False,
            jax_debug_nans=True,
            jax_log_compiles=True,
            jax_platform="gpu",
            torch_cudnn_benchmark=False,
            torch_deterministic=True,
        )
        assert config.name == "custom"
        assert config.backend == BackendType.JAX
        assert config.device == DeviceType.GPU
        assert config.precision == Precision.FLOAT64
        assert config.rng_seed == 123
        assert config.jax_jit_enabled is False
        assert config.jax_debug_nans is True
        assert config.jax_log_compiles is True
        assert config.jax_platform == "gpu"
        assert config.torch_cudnn_benchmark is False
        assert config.torch_deterministic is True


# ------------------------------------------------------------------
# Invalid configurations
# ------------------------------------------------------------------


class TestBackendConfigInvalid:
    """Verify that invalid configurations are rejected."""

    def test_invalid_backend_string(self) -> None:
        with pytest.raises(ValidationError):
            BackendConfig(backend="numpy")

    def test_invalid_device_string(self) -> None:
        with pytest.raises(ValidationError):
            BackendConfig(device="mps")

    def test_invalid_precision_string(self) -> None:
        with pytest.raises(ValidationError):
            BackendConfig(precision="int8")

    def test_negative_rng_seed(self) -> None:
        with pytest.raises(ValidationError):
            BackendConfig(rng_seed=-1)

    def test_invalid_jax_platform(self) -> None:
        with pytest.raises(ValidationError):
            BackendConfig(jax_platform="metal")


# ------------------------------------------------------------------
# Config hashing
# ------------------------------------------------------------------


class TestBackendConfigHashing:
    """Verify deterministic hashing for cache keying."""

    def test_same_config_same_hash(self) -> None:
        config_a = BackendConfig(backend="torch", precision="float32")
        config_b = BackendConfig(backend="torch", precision="float32")
        assert config_a.compute_hash() == config_b.compute_hash()

    def test_different_backend_different_hash(self) -> None:
        config_a = BackendConfig(backend="torch")
        config_b = BackendConfig(backend="jax")
        assert config_a.compute_hash() != config_b.compute_hash()

    def test_different_precision_different_hash(self) -> None:
        config_a = BackendConfig(precision="float32")
        config_b = BackendConfig(precision="float64")
        assert config_a.compute_hash() != config_b.compute_hash()

    def test_different_seed_different_hash(self) -> None:
        config_a = BackendConfig(rng_seed=42)
        config_b = BackendConfig(rng_seed=0)
        assert config_a.compute_hash() != config_b.compute_hash()

    def test_hash_is_string(self) -> None:
        config = BackendConfig()
        h = config.compute_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_is_deterministic(self) -> None:
        config = BackendConfig(backend="torch", rng_seed=7)
        h1 = config.compute_hash()
        h2 = config.compute_hash()
        assert h1 == h2


# ------------------------------------------------------------------
# JAX platform validation
# ------------------------------------------------------------------


class TestJaxPlatformValidation:
    """Verify the jax_platform model validator."""

    def test_none_platform_accepted(self) -> None:
        config = BackendConfig(jax_platform=None)
        assert config.jax_platform is None

    @pytest.mark.parametrize("platform", ["cpu", "gpu", "tpu"])
    def test_valid_platform_strings(self, platform: str) -> None:
        config = BackendConfig(jax_platform=platform)
        assert config.jax_platform == platform

    @pytest.mark.parametrize("bad_platform", ["cuda", "metal", "rocm", ""])
    def test_invalid_platform_strings(self, bad_platform: str) -> None:
        with pytest.raises(ValidationError):
            BackendConfig(jax_platform=bad_platform)
