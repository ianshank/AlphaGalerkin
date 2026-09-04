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


def derive_seeds(base_seed: int, n_seeds: int, stride: int) -> list[int]:
    """Deterministic, decorrelated per-seed values for a multi-seed sweep.

    Multiple PoC scenario configs and the agents research-loop config each
    derive per-cell RNG seeds from a master seed via ``seed + i * stride``
    for a scenario-local prime ``stride`` (values currently differ across
    modules — 1009 in most PoC scenarios, 7919 in ``noyron_basis`` and
    ``src/research/seed_sweep.py`` — so callers keep their own stride rather
    than this helper choosing one). This function centralises the arithmetic
    those call sites duplicated verbatim; it does not unify the stride
    values, since doing so would change which seeds are derived and
    invalidate results already committed to ``config/baselines/*.json``.

    Args:
        base_seed: The first seed; subsequent seeds are strided from it.
        n_seeds: Number of seeds to derive (``>= 0``).
        stride: Per-caller prime stride decorrelating successive seeds.

    Returns:
        ``[base_seed + i * stride for i in range(n_seeds)]``.

    """
    return [base_seed + i * stride for i in range(n_seeds)]


def set_global_seeds(seed: int) -> None:
    """Seed the ``numpy`` and ``torch`` global RNGs for reproducibility.

    ``torch.manual_seed`` seeds both the CPU generator and all CUDA devices, so a
    single call makes ``numpy``- and ``torch``-based *sampling* reproducible.
    Seeding two independent generators is order-independent, so this is a
    byte-for-byte replacement for either ``np``-then-``torch`` or
    ``torch``-then-``np`` inline pairs.

    Scope of the guarantee -- read this before pinning a float in a test:
        Seeding fixes the RNG streams, **not** the arithmetic. This function
        deliberately does *not* set ``torch.backends.cudnn.deterministic`` or
        ``torch.use_deterministic_algorithms``, so GPU kernels (and multi-threaded
        CPU matmul) may reassociate reductions and give run-to-run differences
        within floating-point tolerance. ``src/backend/torch_backend.py`` sets the
        cuDNN flags for callers that need bitwise reproducibility; nothing on the
        scenario/harness paths uses it. Assert on tolerances, not on exact floats.

        (An earlier version of this docstring claimed sampling was "deterministic
        on CPU and GPU alike", which overstated what the two ``manual_seed`` calls
        provide and is the kind of claim a test would then be written against.)

    Args:
        seed: Seed applied to both RNGs. Callers that need to clamp the value to
            numpy's valid ``[0, 2**32 - 1]`` range (e.g. from a large or derived
            seed) should do so before calling and keep their bespoke seeding.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)
