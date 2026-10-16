"""The warnings torch emits when an ONNX export takes its deprecated ``dynamic_axes`` route.

On torch >= 2.9 ``torch.onnx.export`` runs the dynamo exporter, which converts a
``dynamic_axes`` argument to ``dynamic_shapes`` through
``torch.onnx._internal.exporter._dynamic_shapes.from_dynamic_axes_to_dynamic_shapes``.
That conversion is deprecated and slated for removal, and each export through it emits
both messages below (measured in CI run 38667168220, torch 2.14.1; also on torch 2.11).
Once torch removes the function, every such export fails, so the exporter must pass
``dynamic_shapes`` itself (hygiene B50). Shared by the ONNX suites that assert an export
never takes the route.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final
from warnings import WarningMessage

#: A substring of each message torch emits on the deprecated route, one per warning.
DEPRECATED_DYNAMIC_AXES_MESSAGES: Final[tuple[str, ...]] = (
    # UserWarning from torch.onnx._internal.exporter._compat.export_compat
    "'dynamic_axes' is not recommended when dynamo=True",
    # DeprecationWarning from _dynamic_shapes.from_dynamic_axes_to_dynamic_shapes
    "from_dynamic_axes_to_dynamic_shapes is deprecated",
)


def deprecated_dynamic_axes_warnings(caught: Iterable[WarningMessage]) -> list[str]:
    """Return each caught warning that marks the deprecated route, as ``Category: message``."""
    return [
        f"{record.category.__name__}: {record.message}"
        for record in caught
        if any(marker in str(record.message) for marker in DEPRECATED_DYNAMIC_AXES_MESSAGES)
    ]


__all__ = ["DEPRECATED_DYNAMIC_AXES_MESSAGES", "deprecated_dynamic_axes_warnings"]
