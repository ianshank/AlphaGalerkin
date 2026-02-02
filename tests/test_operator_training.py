"""Tests for neural operator training infrastructure.

Tests:
- Model forward pass at multiple resolutions
- Loss function correctness
- Training loop integration
- Resolution transfer
"""

import pytest
import torch

from src.data.physics_dataset import PhysicsDataset
from src.modeling.fno_layer import FNO2d, FNOBlock, SpectralConv2d
from src.modeling.operator import NeuralOperator
from src.physics.darcy import DarcyFlowSolver
from src.training.losses import H1Loss, L2RelativeLoss, get_loss
from src.training.operator_trainer import OperatorTrainer, TrainingConfig


class TestSpectralConv2d:
    """Test spectral convolution layer."""

    def test_output_shape(self) -> None:
        """Test output has same shape as input."""
        layer = SpectralConv2d(in_channels=32, out_channels=32, modes1=8, modes2=8)
        x = torch.randn(4, 32, 16, 16)
        y = layer(x)
        assert y.shape == x.shape

    def test_different_resolutions(self) -> None:
        """Test layer works at multiple resolutions."""
        layer = SpectralConv2d(in_channels=32, out_channels=32, modes1=8, modes2=8)

        for res in [16, 32, 64]:
            x = torch.randn(2, 32, res, res)
            y = layer(x)
            assert y.shape == x.shape

    def test_channel_change(self) -> None:
        """Test changing number of channels."""
        layer = SpectralConv2d(in_channels=32, out_channels=64, modes1=8, modes2=8)
        x = torch.randn(4, 32, 16, 16)
        y = layer(x)
        assert y.shape == (4, 64, 16, 16)


class TestFNOBlock:
    """Test FNO block."""

    def test_output_shape(self) -> None:
        """Test output shape matches input."""
        block = FNOBlock(width=32, modes1=8, modes2=8)
        x = torch.randn(4, 32, 16, 16)
        y = block(x)
        assert y.shape == x.shape

    def test_different_activations(self) -> None:
        """Test different activation functions."""
        for activation in ["gelu", "relu"]:
            block = FNOBlock(width=32, modes1=8, modes2=8, activation=activation)
            x = torch.randn(2, 32, 16, 16)
            y = block(x)
            assert y.shape == x.shape


class TestFNO2d:
    """Test full FNO model."""

    def test_output_shape(self) -> None:
        """Test correct output shape."""
        model = FNO2d(in_channels=1, out_channels=1, width=32, n_layers=2)
        x = torch.randn(4, 1, 16, 16)
        y = model(x)
        assert y.shape == (4, 1, 16, 16)

    def test_resolution_independence(self) -> None:
        """Test model works at different resolutions."""
        model = FNO2d(in_channels=1, out_channels=1, width=32, modes1=8, modes2=8, n_layers=2)

        for res in [16, 32, 64]:
            x = torch.randn(2, 1, res, res)
            y = model(x)
            assert y.shape == (2, 1, res, res)

    def test_multi_channel_output(self) -> None:
        """Test multiple output channels."""
        model = FNO2d(in_channels=1, out_channels=3, width=32, n_layers=2)
        x = torch.randn(4, 1, 16, 16)
        y = model(x)
        assert y.shape == (4, 3, 16, 16)


class TestNeuralOperator:
    """Test NeuralOperator wrapper."""

    def test_fno_backend(self) -> None:
        """Test FNO backend initialization."""
        model = NeuralOperator(in_channels=1, out_channels=1, backend="fno")
        x = torch.randn(4, 1, 16, 16)
        y = model(x)
        assert y.shape == x.shape

    def test_count_parameters(self) -> None:
        """Test parameter counting."""
        model = NeuralOperator(in_channels=1, out_channels=1, width=32, n_layers=2)
        n_params = model.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)


class TestLosses:
    """Test loss functions."""

    def test_l2_relative_loss(self) -> None:
        """Test L2 relative loss computation."""
        loss_fn = L2RelativeLoss()
        pred = torch.randn(4, 1, 16, 16)
        target = torch.randn(4, 1, 16, 16)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_l2_relative_zero_for_identical(self) -> None:
        """Test L2 relative is zero for identical tensors."""
        loss_fn = L2RelativeLoss()
        x = torch.randn(4, 1, 16, 16)

        loss = loss_fn(x, x)
        assert loss.item() < 1e-6

    def test_h1_loss(self) -> None:
        """Test H1 Sobolev loss."""
        loss_fn = H1Loss(lambda_grad=0.1)
        pred = torch.randn(4, 1, 16, 16)
        target = torch.randn(4, 1, 16, 16)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_get_loss_factory(self) -> None:
        """Test loss factory function."""
        l2_loss = get_loss("l2_relative")
        h1_loss = get_loss("h1", lambda_grad=0.2)

        assert isinstance(l2_loss, L2RelativeLoss)
        assert isinstance(h1_loss, H1Loss)


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.lr == 1e-3

    def test_auto_device(self) -> None:
        """Test auto device selection."""
        config = TrainingConfig(device="auto")
        assert config.device in ["cuda", "cpu"]


class TestOperatorTrainer:
    """Test training loop."""

    @pytest.fixture
    def small_model(self) -> NeuralOperator:
        return NeuralOperator(in_channels=1, out_channels=1, width=16, n_layers=1, modes=4)

    @pytest.fixture
    def small_dataset(self) -> PhysicsDataset:
        solver = DarcyFlowSolver(resolution=16)
        return PhysicsDataset(solver, n_samples=32, seed=42)

    def test_trainer_initialization(self, small_model: NeuralOperator) -> None:
        """Test trainer initializes correctly."""
        config = TrainingConfig(epochs=1, device="cpu")
        trainer = OperatorTrainer(small_model, config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.current_epoch == 0

    def test_single_epoch_training(
        self,
        small_model: NeuralOperator,
        small_dataset: PhysicsDataset,
    ) -> None:
        """Test single epoch runs without error."""
        from torch.utils.data import DataLoader

        config = TrainingConfig(epochs=1, batch_size=8, device="cpu")
        trainer = OperatorTrainer(small_model, config)

        train_loader = DataLoader(small_dataset, batch_size=8)
        loss = trainer.train_epoch(train_loader)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_validation(
        self,
        small_model: NeuralOperator,
        small_dataset: PhysicsDataset,
    ) -> None:
        """Test validation runs without error."""
        from torch.utils.data import DataLoader

        config = TrainingConfig(device="cpu")
        trainer = OperatorTrainer(small_model, config)

        val_loader = DataLoader(small_dataset, batch_size=8)
        loss = trainer.validate(val_loader)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_full_training_loop(
        self,
        small_model: NeuralOperator,
        small_dataset: PhysicsDataset,
        tmp_path,
    ) -> None:
        """Test full training loop for 2 epochs."""
        from torch.utils.data import DataLoader

        config = TrainingConfig(
            epochs=2,
            batch_size=8,
            device="cpu",
            patience=100,  # Disable early stopping
            checkpoint_dir=tmp_path,
        )
        trainer = OperatorTrainer(small_model, config)

        loader = DataLoader(small_dataset, batch_size=8)
        history = trainer.fit(loader, loader)

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2


class TestResolutionTransfer:
    """Test resolution transfer capability."""

    def test_train_low_infer_high(self) -> None:
        """Test training on 16x16, inference on 64x64."""
        model = NeuralOperator(in_channels=1, out_channels=1, width=32, modes=8, n_layers=2)

        # "Train" at 16x16
        x_train = torch.randn(4, 1, 16, 16)
        y_train = model(x_train)
        assert y_train.shape == (4, 1, 16, 16)

        # Infer at 64x64 without retraining
        x_test = torch.randn(2, 1, 64, 64)
        y_test = model(x_test)
        assert y_test.shape == (2, 1, 64, 64)

    def test_multiple_resolutions(self) -> None:
        """Test inference at multiple resolutions."""
        model = NeuralOperator(in_channels=1, out_channels=1, width=32, modes=8, n_layers=2)

        for res in [16, 32, 64, 128]:
            x = torch.randn(2, 1, res, res)
            y = model(x)
            assert y.shape == (2, 1, res, res), f"Failed at resolution {res}"
