"""Shared device-resolution helper for PoC scenarios.

The project default is GPU-preferred: scenarios pass ``device='cuda'`` and
expect a clean failure when CUDA is unavailable rather than a silent CPU
fallback that turns a 20-minute training run into a multi-hour one. This
module concentrates that policy in one place so every scenario behaves
consistently.

Four preference forms are accepted:

- ``"cuda"`` (default for headline runs): require any CUDA device. Raise
  ``RuntimeError`` if CUDA is unavailable so the caller fails loud.
- ``"cuda:N"``: require a specific CUDA index. Raise ``RuntimeError`` if
  CUDA is unavailable or index N exceeds the device count. Used to pin a
  workload to a specific GPU on a multi-GPU rig (e.g. ``"cuda:0"`` for
  the 16 GiB primary, ``"cuda:1"`` for the 8 GiB secondary).
- ``"cpu"``: force CPU regardless of CUDA availability. Used by CI smoke
  tests.
- ``"auto"``: best-effort. Use CUDA if available, otherwise fall back to
  CPU silently.
"""

from __future__ import annotations

import re
from typing import Literal

import structlog
import torch

logger = structlog.get_logger(__name__)

# Bare-form device preferences. The runtime ``resolve_device`` also accepts
# ``cuda:N`` indexed strings (matched via ``_CUDA_INDEXED_RE`` below), but
# those don't fit a Literal alias and are passed through as plain ``str``.
DevicePreference = Literal["cuda", "cpu", "auto"]

_CUDA_INDEXED_RE = re.compile(r"^cuda:(\d+)$")


def resolve_device(
    preference: DevicePreference | str,
    *,
    context: str = "scenario",
) -> torch.device:
    """Resolve a device-preference string into a concrete ``torch.device``.

    Args:
    ----
        preference: One of ``"cuda"``, ``"cuda:N"``, ``"cpu"``, or
            ``"auto"``.
        context: Human-readable name of the calling scenario, included in
            the ``RuntimeError`` message when CUDA is requested but
            unavailable. Lets the user see which scenario asked for GPU.

    Returns:
    -------
        The resolved ``torch.device``.

    Raises:
    ------
        RuntimeError: ``preference="cuda"`` or ``"cuda:N"`` was requested
            but CUDA is not available, or N exceeds the device count.
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
    elif (match := _CUDA_INDEXED_RE.match(preference)) is not None:
        index = int(match.group(1))
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{context} requested device={preference!r} but CUDA is not "
                f"available. Set device='cpu' to force CPU, or "
                f"device='auto' to fall back silently."
            )
        n_devices = torch.cuda.device_count()
        if index >= n_devices:
            raise RuntimeError(
                f"{context} requested device={preference!r} but only "
                f"{n_devices} CUDA device(s) are available "
                f"(valid indices: 0..{n_devices - 1})."
            )
        device = torch.device(preference)
    else:
        raise ValueError(
            f"Unknown device preference: {preference!r}. Expected one of "
            f"'cuda', 'cuda:N', 'cpu', 'auto'."
        )

    logger.debug(
        "device_resolved",
        context=context,
        preference=preference,
        device=str(device),
    )
    return device
