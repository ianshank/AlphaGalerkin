"""Shared device-resolution helper for PoC scenarios.

The project default is GPU-preferred: scenarios pass ``device='cuda'`` and
expect a clean failure when CUDA is unavailable rather than a silent CPU
fallback that turns a 20-minute training run into a multi-hour one. This
module concentrates that policy in one place so every scenario behaves
consistently.

Three preferences are accepted:

- ``"cuda"`` (default for headline runs): require CUDA. Raise ``RuntimeError``
  if it is unavailable so the caller fails loud.
- ``"cpu"``: force CPU regardless of CUDA availability. Used by CI smoke
  tests.
- ``"auto"``: best-effort. Use CUDA if available, otherwise fall back to
  CPU silently.
"""

from __future__ import annotations

from typing import Literal

import structlog
import torch

logger = structlog.get_logger(__name__)

DevicePreference = Literal["cuda", "cpu", "auto"]


def resolve_device(
    preference: DevicePreference | str,
    *,
    context: str = "scenario",
) -> torch.device:
    """Resolve a device-preference string into a concrete ``torch.device``.

    Args:
        preference: One of ``"cuda"``, ``"cpu"``, or ``"auto"``.
        context: Human-readable name of the calling scenario, included in
            the ``RuntimeError`` message when CUDA is requested but
            unavailable. Lets the user see which scenario asked for GPU.

    Returns:
        The resolved ``torch.device``.

    Raises:
        RuntimeError: ``preference="cuda"`` was requested but CUDA is not
            available.
        ValueError: ``preference`` is not one of the supported values.

    """
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{context} requested device='cuda' but CUDA is not "
                f"available. Set device='cpu' to force CPU, or "
                f"device='auto' to fall back silently."
            )
        device = torch.device("cuda")
    elif preference == "cpu":
        device = torch.device("cpu")
    elif preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(
            f"Unknown device preference: {preference!r}. Expected one of "
            f"'cuda', 'cpu', 'auto'."
        )

    logger.debug(
        "device_resolved",
        context=context,
        preference=preference,
        device=str(device),
    )
    return device
