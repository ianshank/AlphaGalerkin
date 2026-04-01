"""Base loss types and registry for AlphaGalerkin.

Provides:
- BaseLoss: Protocol for all loss classes
- LossOutput: Dataclass for AlphaGalerkin loss outputs
- LossRegistry / register_loss: Registry for loss discovery
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

from src.templates.registry import create_typed_registry

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LossRegistry, register_loss = create_typed_registry("Loss")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseLoss(Protocol):
    """Protocol that all registered loss classes should satisfy.

    Loss classes must be ``nn.Module`` subclasses and implement a
    ``forward()`` method.  The exact signature varies per loss family,
    so this protocol only requires the common subset.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the loss value."""
        ...


# ---------------------------------------------------------------------------
# Common output dataclass
# ---------------------------------------------------------------------------


@dataclass
class LossOutput:
    """Output from AlphaGalerkin composite loss computation."""

    total: Float[Tensor, ""]
    policy: Float[Tensor, ""]
    value: Float[Tensor, ""]
    lbb: Float[Tensor, ""]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of scalar values."""
        return {
            "total": self.total.item(),
            "policy": self.policy.item(),
            "value": self.value.item(),
            "lbb": self.lbb.item(),
        }
