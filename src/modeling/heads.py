"""Output heads for AlphaGalerkin (Track B).

Provides a registry-driven :class:`VectorFieldHead` that emits a named
mapping of tensors compatible with
:attr:`src.modeling.model.ModelOutput.vector_fields`.  This is the
canonical way for multi-physics PDE operators (e.g. the Navier-Stokes
``(u, v, p)`` triple) to attach vector-valued outputs to the standard
scalar policy/value :class:`ModelOutput` without bespoke plumbing per
operator.

Design rules (carried over from Track A / B in the Production Hardening
Push plan):

* No hardcoded values — every shape parameter is a constructor argument.
* Backwards compatible — ``n_fields=1`` is the default; the head then
  emits a single-key dict whose value matches what the legacy
  :class:`src.modeling.model.DenseHead` would have produced.
* Modular & dynamic — instantiation is registry-driven via
  :func:`src.templates.registry.create_registry`, so configs select
  heads by string name rather than ``if/elif``.
* Reused infra — the underlying MLP is a thin wrapper, the registry
  pattern matches every other module in the codebase.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
from torch import Tensor, nn

from src.templates.registry import create_registry

logger = structlog.get_logger(__name__)


class HeadBase(nn.Module):
    """Base class for AlphaGalerkin output heads."""

    n_fields: int
    field_names: tuple[str, ...]


HeadRegistry, register_head = create_registry("Head", HeadBase)


@register_head("vector_field")
class VectorFieldHead(HeadBase):
    """MLP head that emits a named dict of vector-valued outputs.

    For each field name in ``field_names`` the head produces a tensor
    of shape ``[B, n, components_per_field]`` where ``components_per_field``
    defaults to 1.  The total output dimension of the underlying linear
    layer is therefore ``n_fields * components_per_field``.

    Output contract: a ``dict[str, Tensor]`` with one entry per field
    name.  This mirrors the ``ModelOutput.vector_fields`` shape exactly,
    so callers can do::

        out = base_output.with_vector_fields(head(features))

    without any reshaping.

    Backwards compatibility: with ``n_fields=1`` and the default field
    name ``"scalar"`` the head produces a single-key dict whose tensor
    has shape ``[B, n, 1]`` — equivalent to the legacy
    :class:`src.modeling.model.DenseHead(output_channels=1)` output up
    to a dictionary wrapper.
    """

    def __init__(
        self,
        d_model: int,
        n_fields: int = 1,
        field_names: tuple[str, ...] | list[str] | None = None,
        components_per_field: int = 1,
        d_hidden: int | None = None,
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_fields <= 0:
            raise ValueError(f"n_fields must be positive, got {n_fields}")
        if components_per_field <= 0:
            raise ValueError(f"components_per_field must be positive, got {components_per_field}")

        if field_names is None:
            field_names = (
                ("scalar",) if n_fields == 1 else tuple(f"field_{i}" for i in range(n_fields))
            )
        else:
            field_names = tuple(field_names)
        if len(field_names) != n_fields:
            raise ValueError(
                f"len(field_names)={len(field_names)} does not match n_fields={n_fields}"
            )
        if len(set(field_names)) != n_fields:
            raise ValueError(f"field_names must be unique; got {field_names!r}")

        super().__init__()
        self.n_fields = n_fields
        self.field_names = field_names
        self.components_per_field = components_per_field
        d_hidden_resolved = d_hidden if d_hidden is not None else d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden_resolved),
            nn.GELU(),
            nn.Linear(d_hidden_resolved, n_fields * components_per_field),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Map ``[B, n, d_model]`` to a per-field dict.

        Args:
            x: Input features of shape ``[B, n, d_model]``.

        Returns:
            ``{field_name: Tensor[B, n, components_per_field]}``.

        """
        if x.dim() != 3:
            raise ValueError(
                f"VectorFieldHead expects [B, n, d_model] input, got shape {tuple(x.shape)}"
            )
        out: Tensor = self.net(x)  # [B, n, n_fields * components_per_field]
        out = out.view(*out.shape[:-1], self.n_fields, self.components_per_field)
        # Split along the field axis.
        return {name: out[..., i, :] for i, name in enumerate(self.field_names)}


def make_vector_field_head(
    d_model: int,
    *,
    n_fields: int = 1,
    field_names: tuple[str, ...] | list[str] | None = None,
    components_per_field: int = 1,
    d_hidden: int | None = None,
    head_name: str = "vector_field",
) -> HeadBase:
    """Factory: create a head by registry name.

    Provided for symmetry with the rest of the codebase
    (``create_loss_balancer``, ``create_time_stepper``, …) so callers
    can drive head selection from config strings without importing the
    concrete class.

    Args:
        d_model: Input feature dimension.
        n_fields: Number of named output fields.
        field_names: Optional explicit names.  Required to be of length
            ``n_fields`` when supplied.
        components_per_field: Scalars per field (1 = scalar field;
            higher values useful for tensor-valued fields).
        d_hidden: Hidden width of the MLP; defaults to ``d_model``.
        head_name: Registry key.  Defaults to ``"vector_field"``.

    Returns:
        An instantiated :class:`HeadBase`.

    """
    head_cls: type[HeadBase] = HeadRegistry().get(head_name)
    instance = head_cls(
        d_model=d_model,
        n_fields=n_fields,
        field_names=field_names,
        components_per_field=components_per_field,
        d_hidden=d_hidden,
    )
    logger.debug(
        "head_created",
        head=head_name,
        n_fields=n_fields,
        field_names=instance.field_names,
        components_per_field=components_per_field,
    )
    return instance


__all__ = [
    "HeadBase",
    "HeadRegistry",
    "VectorFieldHead",
    "make_vector_field_head",
    "register_head",
]


# Local helper to silence the linter when the imports above are
# considered "unused" by tooling that does not see the public re-export.
def _typing_anchor() -> Any:  # pragma: no cover - typing helper
    return torch
