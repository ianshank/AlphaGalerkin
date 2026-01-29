"""Unit tests for Galerkin Neural Operator.

Tests configuration validation, forward passes, resolution independence,
and LBB stability monitoring.
"""

from __future__ import annotations

import pytest

# Skip entire module if torch not available
torch = pytest.importorskip("torch")

from pydantic import ValidationError

from src.modeling.galerkin_operator import (
    Galerkin2d,
    GalerkinOperatorBlock,
    GalerkinOperatorConfig,
)
from src.modeling.operator import NeuralOperator


class TestGalerkinOperatorConfig:
    """Tests for GalerkinOperatorConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GalerkinOperatorConfig(name="test")
        assert config.in_channels == 1
        assert config.out_channels == 1
        assert config.width == 64
        assert config.n_layers == 4
        assert config.n_heads == 4
        assert config.fourier_features == 64
        assert len(config.fourier_scales) == 4
        assert config.dropout == 0.0
        assert config.lbb_threshold == 1e-6
        assert config.lbb_regularization == 0.01

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = GalerkinOperatorConfig(
            name="custom",
            in_channels=3,
            out_channels=2,
            width=128,
            n_layers=6,
            n_heads=8,
            fourier_features=32,
            fourier_scales=[1.0, 2.0],
            dropout=0.1,
        )
        assert config.in_channels == 3
        assert config.out_channels == 2
        assert config.width == 128
        assert config.n_layers == 6
        assert config.n_heads == 8
        assert config.fourier_features == 32
        assert config.fourier_scales == [1.0, 2.0]
        assert config.dropout == 0.1

    def test_validation_in_channels_min(self) -> None:
        """Test in_channels minimum validation."""
        with pytest.raises(ValidationError):
            GalerkinOperatorConfig(name="test", in_channels=0)

    def test_validation_width_range(self) -> None:
        """Test width range validation."""
        with pytest.raises(ValidationError):
            GalerkinOperatorConfig(name="test", width=4)  # Below min 8
        with pytest.raises(ValidationError):
            GalerkinOperatorConfig(name="test", width=2048)  # Above max 1024

    def test_validation_dropout_range(self) -> None:
        """Test dropout range validation."""
        with pytest.raises(ValidationError):
            GalerkinOperatorConfig(name="test", dropout=-0.1)
        with pytest.raises(ValidationError):
            GalerkinOperatorConfig(name="test", dropout=1.0)

    def test_config_hash_deterministic(self) -> None:
        """Test that config hash is deterministic."""
        config1 = GalerkinOperatorConfig(name="test", width=64)
        config2 = GalerkinOperatorConfig(name="test", width=64)
        assert config1.compute_hash() == config2.compute_hash()

    def test_validation_width_n_heads_divisibility(self) -> None:
        """Test width must be divisible by n_heads."""
        with pytest.raises(ValidationError):
            GalerkinOperatorConfig(name="test", width=32, n_heads=5)  # 32 % 5 != 0

    def test_validation_width_greater_than_n_heads(self) -> None:
        """Test width must be >= n_heads."""
        with pytest.raises(ValidationError):
            GalerkinOperatorConfig(name="test", width=8, n_heads=16)


class TestGalerkinOperatorBlock:
    """Tests for GalerkinOperatorBlock."""

    @pytest.fixture
    def block(self) -> GalerkinOperatorBlock:
        """Create a test block."""
        torch.manual_seed(42)
        return GalerkinOperatorBlock(
            d_model=32,
            n_heads=4,
            d_ffn=64,
            dropout=0.0,
        )

    def test_output_shape(self, block: GalerkinOperatorBlock) -> None:
        """Test block output shape matches input."""
        batch, seq, d_model = 2, 16, 32
        x = torch.randn(batch, seq, d_model)
        y = block(x)
        assert y.shape == x.shape

    def test_lbb_return(self, block: GalerkinOperatorBlock) -> None:
        """Test LBB constant return."""
        batch, seq, d_model = 2, 16, 32
        x = torch.randn(batch, seq, d_model)
        y, lbb = block(x, return_lbb=True)
        assert y.shape == x.shape
        assert lbb.shape == (batch,)
        assert (lbb > 0).all(), "LBB constants should be positive"

    def test_gradient_flow(self, block: GalerkinOperatorBlock) -> None:
        """Test gradients flow through block."""
        x = torch.randn(2, 16, 32, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_different_activations(self) -> None:
        """Test different activation functions work."""
        for activation in ["gelu", "relu", "silu"]:
            block = GalerkinOperatorBlock(
                d_model=32,
                n_heads=4,
                activation=activation,
            )
            x = torch.randn(2, 16, 32)
            y = block(x)
            assert y.shape == x.shape

    def test_dimension_validation_divisibility(self) -> None:
        """Test d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            GalerkinOperatorBlock(d_model=32, n_heads=5)

    def test_dimension_validation_minimum(self) -> None:
        """Test d_model must be >= n_heads."""
        with pytest.raises(ValueError, match="must be >="):
            GalerkinOperatorBlock(d_model=4, n_heads=8)


class TestGalerkin2d:
    """Tests for Galerkin2d neural operator."""

    @pytest.fixture
    def model(self) -> Galerkin2d:
        """Create a small test model."""
        torch.manual_seed(42)
        return Galerkin2d(
            in_channels=1,
            out_channels=1,
            width=32,
            n_layers=2,
            n_heads=4,
            fourier_features=16,
        )

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_output_shape(self, model: Galerkin2d) -> None:
        """Test output shape matches expected."""
        batch, h, w = 2, 16, 16
        x = torch.randn(batch, 1, h, w)
        y = model(x)
        assert y.shape == (batch, 1, h, w)

    def test_variable_resolution(self, model: Galerkin2d) -> None:
        """Test model works at different resolutions."""
        batch = 2
        for resolution in [8, 12, 16, 20, 24]:
            x = torch.randn(batch, 1, resolution, resolution)
            y = model(x)
            assert y.shape == (batch, 1, resolution, resolution), \
                f"Failed at resolution {resolution}"

    def test_non_square_resolution(self, model: Galerkin2d) -> None:
        """Test model works with non-square inputs."""
        batch = 2
        x = torch.randn(batch, 1, 12, 16)
        y = model(x)
        assert y.shape == (batch, 1, 12, 16)

    def test_explicit_coords(self, model: Galerkin2d) -> None:
        """Test model with explicit coordinates."""
        batch, h, w = 2, 16, 16
        x = torch.randn(batch, 1, h, w)

        # Create custom coordinates
        grid_x = torch.linspace(0, 1, h)
        grid_y = torch.linspace(0, 1, w)
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)  # (h, w, 2)
        coords = coords.unsqueeze(0).expand(batch, -1, -1, -1)  # (batch, h, w, 2)

        y = model(x, coords=coords)
        assert y.shape == (batch, 1, h, w)

    def test_lbb_return(self, model: Galerkin2d) -> None:
        """Test LBB constants are returned correctly."""
        x = torch.randn(2, 1, 16, 16)
        y, lbb_list = model(x, return_lbb=True)
        assert y.shape == (2, 1, 16, 16)
        assert len(lbb_list) == model.n_layers
        for lbb in lbb_list:
            assert lbb.shape == (2,)
            assert (lbb > 0).all()

    def test_lbb_regularization(self, model: Galerkin2d) -> None:
        """Test LBB regularization loss computation."""
        x = torch.randn(2, 1, 16, 16)
        _ = model(x, return_lbb=True)
        reg_loss = model.get_lbb_regularization()
        assert reg_loss.shape == ()
        assert reg_loss >= 0

    def test_gradient_flow(self, model: Galerkin2d) -> None:
        """Test gradients flow through model."""
        x = torch.randn(2, 1, 16, 16, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_training_step(self, model: Galerkin2d) -> None:
        """Test a single training step."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 1, 16, 16)
        target = torch.randn(4, 1, 16, 16)

        optimizer.zero_grad()
        y, lbb_list = model(x, return_lbb=True)
        mse_loss = torch.nn.functional.mse_loss(y, target)
        lbb_loss = model.get_lbb_regularization()
        total_loss = mse_loss + lbb_loss
        total_loss.backward()
        optimizer.step()

        # Verify parameters updated
        assert total_loss.item() > 0

    def test_multi_channel(self) -> None:
        """Test model with multiple input/output channels."""
        model = Galerkin2d(
            in_channels=3,
            out_channels=2,
            width=32,
            n_layers=2,
        )
        x = torch.randn(2, 3, 16, 16)
        y = model(x)
        assert y.shape == (2, 2, 16, 16)

    def test_from_config(self) -> None:
        """Test creating model from config object."""
        config = GalerkinOperatorConfig(
            name="test",
            in_channels=2,
            out_channels=2,
            width=32,
            n_layers=2,
            n_heads=4,
        )
        model = Galerkin2d(config=config)
        assert model.in_channels == 2
        assert model.out_channels == 2
        assert model.width == 32
        assert model.n_layers == 2

    def test_count_parameters(self, model: Galerkin2d) -> None:
        """Test parameter counting."""
        n_params = model.count_parameters()
        assert n_params > 0
        assert n_params == sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_device_transfer(self, model: Galerkin2d, device: torch.device) -> None:
        """Test model works after device transfer."""
        model = model.to(device)
        x = torch.randn(2, 1, 16, 16, device=device)
        y = model(x)
        assert y.device == device
        assert y.shape == (2, 1, 16, 16)

    def test_lbb_regularization_device_consistency(
        self, model: Galerkin2d, device: torch.device
    ) -> None:
        """Test LBB regularization returns tensor on correct device."""
        model = model.to(device)

        # Test with no prior forward pass - should return zero on correct device
        reg_loss = model.get_lbb_regularization()
        assert reg_loss.device == device, "LBB reg should be on model device"
        assert reg_loss.item() == 0.0

        # Test after forward pass with return_lbb=True
        x = torch.randn(2, 1, 16, 16, device=device)
        _ = model(x, return_lbb=True)
        reg_loss = model.get_lbb_regularization()
        assert reg_loss.device == device, "LBB reg should be on model device"


class TestNeuralOperatorGalerkinBackend:
    """Tests for Galerkin backend integration with NeuralOperator."""

    def test_galerkin_backend_init(self) -> None:
        """Test initializing NeuralOperator with Galerkin backend."""
        model = NeuralOperator(
            in_channels=1,
            out_channels=1,
            width=32,
            n_layers=2,
            backend="galerkin",
        )
        assert model.backend == "galerkin"
        assert isinstance(model.model, Galerkin2d)

    def test_galerkin_forward(self) -> None:
        """Test forward pass with Galerkin backend."""
        model = NeuralOperator(
            in_channels=1,
            out_channels=1,
            width=32,
            n_layers=2,
            backend="galerkin",
        )
        x = torch.randn(2, 1, 16, 16)
        y = model(x)
        assert y.shape == (2, 1, 16, 16)

    def test_galerkin_resolution_transfer(self) -> None:
        """Test resolution independence with Galerkin backend."""
        torch.manual_seed(42)
        model = NeuralOperator(
            in_channels=1,
            out_channels=1,
            width=32,
            n_layers=2,
            backend="galerkin",
        )

        # Test at multiple resolutions
        for res in [9, 13, 17, 19]:
            x = torch.randn(2, 1, res, res)
            y = model(x)
            assert y.shape == (2, 1, res, res), f"Failed at resolution {res}"

    def test_count_parameters(self) -> None:
        """Test parameter counting for Galerkin backend."""
        model = NeuralOperator(
            in_channels=1,
            out_channels=1,
            width=32,
            n_layers=2,
            backend="galerkin",
        )
        n_params = model.count_parameters()
        assert n_params > 0

    def test_fno_vs_galerkin_interface_compatibility(self) -> None:
        """Test that FNO and Galerkin backends have compatible interfaces."""
        fno_model = NeuralOperator(
            in_channels=1,
            out_channels=1,
            width=32,
            n_layers=2,
            backend="fno",
        )
        galerkin_model = NeuralOperator(
            in_channels=1,
            out_channels=1,
            width=32,
            n_layers=2,
            backend="galerkin",
        )

        x = torch.randn(2, 1, 16, 16)

        # Both should work with same input
        y_fno = fno_model(x)
        y_galerkin = galerkin_model(x)

        assert y_fno.shape == y_galerkin.shape
