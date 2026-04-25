"""Tests for loss registry, base types, and get_loss factory.

Covers:
- LossRegistry singleton pattern
- register_loss decorator
- get_loss factory with aliases and error handling
- LossOutput dataclass
- BaseLoss protocol
- All registered losses produce finite output
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.training.losses import get_loss
from src.training.losses.alphagalerkin import AlphaGalerkinLoss, EntropyRegularizer
from src.training.losses.base import BaseLoss, LossOutput, LossRegistry
from src.training.losses.operator import H1Loss, L2RelativeLoss, MSELoss

# ---------------------------------------------------------------------------
# LossRegistry singleton
# ---------------------------------------------------------------------------


class TestLossRegistrySingleton:
    """Test LossRegistry singleton pattern."""

    def test_singleton_identity(self) -> None:
        """Two instantiations return the same object."""
        r1 = LossRegistry()
        r2 = LossRegistry()
        assert r1 is r2

    def test_registry_has_items(self) -> None:
        """Registry is populated after import of loss modules."""
        registry = LossRegistry()
        items = registry.list_items()
        assert len(items) > 0

    def test_known_losses_registered(self) -> None:
        """All expected loss names are present in the registry."""
        registry = LossRegistry()
        items = registry.list_items()
        expected = {
            "alphagalerkin",
            "entropy_regularizer",
            "l2_relative",
            "h1",
            "mse",
            "residual",
            "boundary",
            "initial_condition",
            "conservation",
            "physics_informed",
            "combined_alphagalerkin_physics",
        }
        for name in expected:
            assert name in items, f"'{name}' not found in registry. Available: {items}"

    def test_get_returns_correct_class(self) -> None:
        """Registry.get() returns the class that was registered."""
        registry = LossRegistry()
        assert registry.get("alphagalerkin") is AlphaGalerkinLoss
        assert registry.get("l2_relative") is L2RelativeLoss
        assert registry.get("h1") is H1Loss
        assert registry.get("mse") is MSELoss
        assert registry.get("entropy_regularizer") is EntropyRegularizer

    def test_get_unknown_returns_none(self) -> None:
        """Registry.get() returns None for unknown names."""
        registry = LossRegistry()
        assert registry.get("nonexistent_loss_xyz") is None


# ---------------------------------------------------------------------------
# register_loss decorator
# ---------------------------------------------------------------------------


class TestRegisterLossDecorator:
    """Test the register_loss decorator."""

    def test_decorator_registers_class(self) -> None:
        """A class decorated with @register_loss appears in the registry."""
        # alphagalerkin was registered at import time
        registry = LossRegistry()
        assert registry.get("alphagalerkin") is not None

    def test_decorated_class_is_callable(self) -> None:
        """Registered classes can be instantiated."""
        registry = LossRegistry()
        cls = registry.get("mse")
        assert cls is not None
        instance = cls()
        assert isinstance(instance, nn.Module)


# ---------------------------------------------------------------------------
# get_loss factory
# ---------------------------------------------------------------------------


class TestGetLossFactory:
    """Test the get_loss() factory function."""

    def test_get_loss_by_name(self) -> None:
        """get_loss returns correctly typed instances."""
        assert isinstance(get_loss("l2_relative"), L2RelativeLoss)
        assert isinstance(get_loss("h1"), H1Loss)
        assert isinstance(get_loss("mse"), MSELoss)
        assert isinstance(get_loss("alphagalerkin"), AlphaGalerkinLoss)
        assert isinstance(get_loss("entropy_regularizer"), EntropyRegularizer)

    def test_alias_l2(self) -> None:
        """'l2' alias resolves to L2RelativeLoss."""
        loss = get_loss("l2")
        assert isinstance(loss, L2RelativeLoss)

    def test_alias_sobolev(self) -> None:
        """'sobolev' alias resolves to H1Loss."""
        loss = get_loss("sobolev")
        assert isinstance(loss, H1Loss)

    def test_unknown_name_raises_value_error(self) -> None:
        """Unknown name raises ValueError with available names."""
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("definitely_not_a_loss")

    def test_kwargs_forwarded(self) -> None:
        """Keyword arguments are forwarded to the constructor."""
        loss = get_loss("l2_relative", eps=1e-4)
        assert loss.eps == 1e-4

    def test_alphagalerkin_kwargs(self) -> None:
        """AlphaGalerkinLoss kwargs forwarded correctly."""
        loss = get_loss("alphagalerkin", policy_weight=2.0, value_weight=0.5)
        assert loss.policy_weight == 2.0
        assert loss.value_weight == 0.5

    def test_all_non_physics_losses_are_nn_module(self) -> None:
        """All non-physics losses retrieved via get_loss are nn.Module."""
        for name in ["l2_relative", "h1", "mse", "alphagalerkin", "entropy_regularizer"]:
            loss = get_loss(name)
            assert isinstance(loss, nn.Module), f"{name} is not an nn.Module"


# ---------------------------------------------------------------------------
# LossOutput dataclass
# ---------------------------------------------------------------------------


class TestLossOutput:
    """Test LossOutput dataclass."""

    def test_creation(self) -> None:
        """LossOutput can be created with tensor values."""
        output = LossOutput(
            total=torch.tensor(3.0),
            policy=torch.tensor(1.0),
            value=torch.tensor(1.5),
            lbb=torch.tensor(0.5),
        )
        assert output.total.item() == pytest.approx(3.0)
        assert output.policy.item() == pytest.approx(1.0)
        assert output.value.item() == pytest.approx(1.5)
        assert output.lbb.item() == pytest.approx(0.5)

    def test_to_dict(self) -> None:
        """to_dict returns Python floats."""
        output = LossOutput(
            total=torch.tensor(2.0),
            policy=torch.tensor(1.0),
            value=torch.tensor(0.8),
            lbb=torch.tensor(0.2),
        )
        d = output.to_dict()
        assert set(d.keys()) == {"total", "policy", "value", "lbb"}
        for v in d.values():
            assert isinstance(v, float)

    def test_to_dict_values(self) -> None:
        """to_dict values match tensor values."""
        output = LossOutput(
            total=torch.tensor(5.5),
            policy=torch.tensor(2.0),
            value=torch.tensor(3.0),
            lbb=torch.tensor(0.5),
        )
        d = output.to_dict()
        assert d["total"] == pytest.approx(5.5)
        assert d["policy"] == pytest.approx(2.0)
        assert d["value"] == pytest.approx(3.0)
        assert d["lbb"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# BaseLoss protocol
# ---------------------------------------------------------------------------


class TestBaseLossProtocol:
    """Test BaseLoss protocol compliance."""

    def test_alphagalerkin_satisfies_protocol(self) -> None:
        """AlphaGalerkinLoss satisfies BaseLoss protocol."""
        loss = AlphaGalerkinLoss()
        assert isinstance(loss, BaseLoss)

    def test_l2_satisfies_protocol(self) -> None:
        """L2RelativeLoss satisfies BaseLoss protocol."""
        loss = L2RelativeLoss()
        assert isinstance(loss, BaseLoss)

    def test_h1_satisfies_protocol(self) -> None:
        """H1Loss satisfies BaseLoss protocol."""
        loss = H1Loss()
        assert isinstance(loss, BaseLoss)

    def test_mse_satisfies_protocol(self) -> None:
        """MSELoss satisfies BaseLoss protocol."""
        loss = MSELoss()
        assert isinstance(loss, BaseLoss)

    def test_entropy_satisfies_protocol(self) -> None:
        """EntropyRegularizer satisfies BaseLoss protocol."""
        loss = EntropyRegularizer()
        assert isinstance(loss, BaseLoss)


# ---------------------------------------------------------------------------
# All registered losses produce finite output
# ---------------------------------------------------------------------------

BATCH = 4
SIZE = 5
ACTION_SIZE = SIZE * SIZE


class TestAllRegisteredLossesFiniteOutput:
    """Verify all non-physics registered losses produce finite output."""

    def test_l2_relative_finite(self) -> None:
        """L2RelativeLoss produces finite output."""
        loss = get_loss("l2_relative")
        pred = torch.randn(BATCH, SIZE, SIZE)
        target = torch.randn(BATCH, SIZE, SIZE)
        result = loss(pred, target)
        assert result.isfinite()

    def test_h1_finite(self) -> None:
        """H1Loss produces finite output."""
        loss = get_loss("h1")
        pred = torch.randn(BATCH, 1, SIZE, SIZE)
        target = torch.randn(BATCH, 1, SIZE, SIZE)
        result = loss(pred, target)
        assert result.isfinite()

    def test_mse_finite(self) -> None:
        """MSELoss produces finite output."""
        loss = get_loss("mse")
        pred = torch.randn(BATCH, SIZE)
        target = torch.randn(BATCH, SIZE)
        result = loss(pred, target)
        assert result.isfinite()

    def test_alphagalerkin_finite(self) -> None:
        """AlphaGalerkinLoss produces finite output."""
        loss = get_loss("alphagalerkin")
        policy_logits = torch.randn(BATCH, ACTION_SIZE)
        value = torch.randn(BATCH, 1)
        target_policy = torch.softmax(torch.randn(BATCH, ACTION_SIZE), dim=-1)
        target_value = torch.randn(BATCH, 1)
        result = loss(policy_logits, value, target_policy, target_value)
        assert result.total.isfinite()
        assert result.policy.isfinite()
        assert result.value.isfinite()
        assert result.lbb.isfinite()

    def test_entropy_regularizer_finite(self) -> None:
        """EntropyRegularizer produces finite output."""
        loss = get_loss("entropy_regularizer")
        logits = torch.randn(BATCH, ACTION_SIZE)
        result = loss(logits)
        assert result.isfinite()

    def test_alias_l2_finite(self) -> None:
        """Alias 'l2' produces finite output."""
        loss = get_loss("l2")
        pred = torch.randn(BATCH, SIZE)
        target = torch.randn(BATCH, SIZE)
        result = loss(pred, target)
        assert result.isfinite()

    def test_alias_sobolev_finite(self) -> None:
        """Alias 'sobolev' produces finite output."""
        loss = get_loss("sobolev")
        pred = torch.randn(BATCH, 1, SIZE, SIZE)
        target = torch.randn(BATCH, 1, SIZE, SIZE)
        result = loss(pred, target)
        assert result.isfinite()
