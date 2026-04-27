"""GPU-primary device resolution for the perf benchmark.

Wraps the project-level ``src.poc.device.resolve_device`` to additionally
accept indexed CUDA strings (``"cuda:0"``, ``"cuda:1"``, ...). The benchmark
needs this because a multi-GPU workstation (e.g. RTX 5060 Ti 16 GB at
cuda:0 + RTX 5060 8 GB at cuda:1) is the canonical home-lab setup we
target — Phase 0 must measure both cards independently so later phases
(daemon, multi-stream decode) can plan around their VRAM and throughput.

The shared helper is left untouched to avoid breaking PoC scenarios that
expect bare ``"cuda"``/``"cpu"``/``"auto"``.
"""

from __future__ import annotations

import re

import structlog
import torch

from src.poc.device import resolve_device as _resolve_bare

logger = structlog.get_logger(__name__)

# Pattern for ``cuda:N`` strings. We do not accept whitespace or trailing
# characters; the user must type the canonical form.
_CUDA_INDEX_RE = re.compile(r"^cuda:(\d+)$")


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

    Raises:
    ------
        RuntimeError: indexed CUDA requested but the index is out of range
            (or CUDA is unavailable).
        ValueError: ``preference`` is malformed.

    """
    match = _CUDA_INDEX_RE.match(preference)
    if match is None:
        # Delegate the bare cases to the shared helper — keeps GPU-failure
        # semantics identical across PoC and perf code.
        return _resolve_bare(preference, context=context)

    index = int(match.group(1))
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{context} requested device={preference!r} but CUDA is not "
            f"available. Set device='cpu' or run on a CUDA-capable host.",
        )
    n_devices = torch.cuda.device_count()
    if index >= n_devices:
        raise RuntimeError(
            f"{context} requested device={preference!r} but only "
            f"{n_devices} CUDA device(s) are present (indices 0.."
            f"{n_devices - 1}).",
        )
    device = torch.device(f"cuda:{index}")
    logger.debug(
        "device_resolved.indexed",
        context=context,
        preference=preference,
        device=str(device),
        n_devices=n_devices,
    )
    return device


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
