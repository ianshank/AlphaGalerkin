"""GPU-primary device resolution for the perf benchmark.

The benchmark targets a multi-GPU workstation (e.g. RTX 5060 Ti 16 GB at
cuda:0 + RTX 5060 8 GB at cuda:1) and must measure both cards independently,
so it needs indexed ``"cuda:N"`` resolution. ``src.poc.device.resolve_device``
now accepts ``"cuda:N"`` directly (with identical RuntimeError/ValueError
semantics), so this module delegates to it — one source of truth for device
resolution and GPU-failure semantics across the PoC and perf paths.
"""

from __future__ import annotations

import structlog
import torch

from src.poc.device import resolve_device as _resolve_bare

logger = structlog.get_logger(__name__)


def list_cuda_devices() -> list[torch.device]:
    """Return all available CUDA devices in index order.

    Empty list if CUDA is not available — callers are expected to fall
    back gracefully (e.g. by skipping the GPU sweep) rather than abort.
    """
    if not torch.cuda.is_available():
        return []
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def resolve_device(
    preference: str,
    *,
    context: str = "perf_benchmark",
) -> torch.device:
    """Resolve a preference string to a concrete ``torch.device``.

    Accepts every form ``src.poc.device.resolve_device`` does, plus
    indexed CUDA strings::

        resolve_device("cuda")     # default device
        resolve_device("cuda:0")   # primary GPU
        resolve_device("cuda:1")   # secondary GPU
        resolve_device("cpu")
        resolve_device("auto")

    Raises
    ------
        RuntimeError: CUDA requested but unavailable, or an indexed CUDA
            device is out of range.
        ValueError: ``preference`` is malformed.

    """
    # The shared helper already validates bare and ``cuda:N`` forms with the
    # exact GPU-failure semantics we want; delegate to avoid drift.
    return _resolve_bare(preference, context=context)


def device_label(device: torch.device) -> str:
    """Stable, hardware-friendly label for log events and reports.

    For CUDA devices we include the GPU model name (e.g.
    ``"cuda:0:NVIDIA-GeForce-RTX-5060-Ti"``) so reports recorded on
    different hardware are visibly distinct without needing the hardware
    tag. CPU devices return ``"cpu"`` unchanged.
    """
    if device.type != "cuda":
        return device.type
    try:
        props = torch.cuda.get_device_properties(device)
        # Replace whitespace so the label is safe to use in cell keys.
        name = props.name.strip().replace(" ", "-")
    except (RuntimeError, AssertionError):  # pragma: no cover - defensive
        return f"cuda:{device.index if device.index is not None else 0}"
    return f"cuda:{device.index if device.index is not None else 0}:{name}"
