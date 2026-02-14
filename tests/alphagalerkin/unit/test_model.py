"""Tests for AlphaGalerkinNetwork (nn/model.py)."""

from __future__ import annotations

from pathlib import Path

import torch

from src.alphagalerkin.core.config import NetworkConfig
from src.alphagalerkin.nn.model import AlphaGalerkinNetwork


# Use a small config to keep tests fast.
def _small_config() -> NetworkConfig:
    return NetworkConfig(
        input_features=8,
        gnn={"hidden_dim": 32, "num_layers": 2, "attention_heads": 4},
        policy_head={"hidden_dims": [32, 16]},
        value_head={"hidden_dims": [32, 16]},
    )


class TestAlphaGalerkinNetworkForward:
    """Forward pass shape and dtype checks."""

    def test_forward_batched_shape(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        batch, num_elements = 2, 6
        features = torch.randn(batch, num_elements, config.input_features)

        policy, value = model(features)

        assert policy.shape[0] == batch
        assert policy.shape[1] == num_elements
        assert value.shape == (batch, 1)

    def test_forward_single_element(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        features = torch.randn(1, 1, config.input_features)

        policy, value = model(features)

        assert policy.shape[0] == 1
        assert value.shape == (1, 1)

    def test_forward_dtype_float32(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        features = torch.randn(2, 4, config.input_features)

        policy, value = model(features)

        assert policy.dtype == torch.float32
        assert value.dtype == torch.float32

    def test_forward_with_action_mask(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        batch, num_elements = 2, 4
        features = torch.randn(batch, num_elements, config.input_features)
        # PolicyHead outputs num_actions based on its final layer;
        # run once without mask to discover output action dim.
        policy_unmasked, _ = model(features)
        num_actions = policy_unmasked.shape[-1]

        mask = torch.ones(batch, num_elements, num_actions)
        mask[:, :, 0] = 0  # mask out first action

        policy_masked, value = model(features, action_mask=mask)

        assert policy_masked.shape == policy_unmasked.shape
        assert value.shape == (batch, 1)


class TestAlphaGalerkinNetworkPredict:
    """Inference-mode predictions."""

    def test_predict_no_grad(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        features = torch.randn(1, 4, config.input_features)

        policy, value = model.predict(features)

        assert not policy.requires_grad
        assert not value.requires_grad

    def test_predict_sets_eval_mode(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        model.train()  # explicitly in training mode

        _ = model.predict(torch.randn(1, 4, config.input_features))

        assert not model.training


class TestAlphaGalerkinNetworkLBB:
    """LBB regularization loss computation."""

    def test_compute_lbb_loss_returns_scalar(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        features = torch.randn(1, 4, config.input_features)

        lbb = model.compute_lbb_loss(features)

        assert lbb.dim() == 0  # scalar
        assert lbb.dtype == torch.float32

    def test_compute_lbb_loss_non_negative(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        features = torch.randn(1, 4, config.input_features)

        lbb = model.compute_lbb_loss(features)

        assert float(lbb) >= 0.0

    def test_compute_lbb_loss_galerkin_backbone(self) -> None:
        """With galerkin backbone, LBB loss may be non-trivial."""
        config = NetworkConfig(
            input_features=8,
            gnn={
                "hidden_dim": 32,
                "num_layers": 2,
                "attention_heads": 4,
                "architecture": "galerkin",
            },
            policy_head={"hidden_dims": [32, 16]},
            value_head={"hidden_dims": [32, 16]},
        )
        model = AlphaGalerkinNetwork(config)
        features = torch.randn(1, 4, config.input_features)

        lbb = model.compute_lbb_loss(features)

        assert lbb.dim() == 0
        assert lbb.dtype == torch.float32


class TestAlphaGalerkinNetworkSaveLoad:
    """Save / load round-trip."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        save_path = tmp_path / "subdir" / "model.pt"

        model.save(save_path)

        assert save_path.exists()

    def test_load_restores_weights(self, tmp_path: Path) -> None:
        config = _small_config()
        model_a = AlphaGalerkinNetwork(config)
        save_path = tmp_path / "model.pt"
        model_a.save(save_path)

        model_b = AlphaGalerkinNetwork.load(save_path, config)

        # All parameters should be identical.
        for (name_a, p_a), (_, p_b) in zip(
            model_a.named_parameters(),
            model_b.named_parameters(),
        ):
            assert torch.equal(p_a, p_b), f"Mismatch in {name_a}"

    def test_load_produces_same_output(self, tmp_path: Path) -> None:
        config = _small_config()
        model_a = AlphaGalerkinNetwork(config)
        model_a.eval()
        features = torch.randn(1, 4, config.input_features)
        policy_a, value_a = model_a(features)

        save_path = tmp_path / "model.pt"
        model_a.save(save_path)
        model_b = AlphaGalerkinNetwork.load(save_path, config)
        model_b.eval()
        policy_b, value_b = model_b(features)

        assert torch.allclose(policy_a, policy_b, atol=1e-6)
        assert torch.allclose(value_a, value_b, atol=1e-6)

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)
        deep_path = tmp_path / "a" / "b" / "c" / "model.pt"

        model.save(deep_path)

        assert deep_path.exists()


class TestAlphaGalerkinNetworkConfig:
    """Config is stored on the model."""

    def test_config_attribute(self) -> None:
        config = _small_config()
        model = AlphaGalerkinNetwork(config)

        assert model.config is config
        assert model.config.input_features == 8
