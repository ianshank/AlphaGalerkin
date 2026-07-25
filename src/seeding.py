"""Global RNG seeding for reproducible AlphaGalerkin runs.

A single home for the ``numpy`` + ``torch`` seeding idiom that was duplicated
verbatim across the PoC scenarios, the research harnesses, and the agent
research loop (each doing ``np.random.seed(seed); torch.manual_seed(seed)`` at
run/episode setup).

Kept as a plain module (a sibling to ``src/constants.py``) rather than a package
so it does not enlarge the drift-guarded package map enforced by
``tests/docs/test_architecture_map.py`` (which enumerates ``src/*/__init__.py``).

This is the reproducibility-seeding entry point for run/episode setup. It is
deliberately distinct from ``Backend.set_seed`` (``src/backend/``), which seeds a
specific backend's RNG through the backend abstraction; call sites that seed
``numpy`` and ``torch`` directly should use :func:`set_global_seeds`.
"""

from __future__ import annotations

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Seed the ``numpy`` and ``torch`` global RNGs for reproducibility.

    ``torch.manual_seed`` seeds both the CPU generator and all CUDA devices, so a
    single call makes ``numpy``- and ``torch``-based sampling deterministic on CPU
    and GPU alike. Seeding two independent generators is order-independent, so
    this is a byte-for-byte replacement for either ``np``-then-``torch`` or
    ``torch``-then-``np`` inline pairs.

    Args:
        seed: Seed applied to both RNGs. Callers that need to clamp the value to
            numpy's valid ``[0, 2**32 - 1]`` range (e.g. from a large or derived
            seed) should do so before calling and keep their bespoke seeding.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)
