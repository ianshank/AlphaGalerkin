"""Entry-point shim for the ``eval_harness.plugins`` group.

The harness discovers third-party plugins by calling ``ep.load()`` on each entry
point — which imports the referenced object but does not call it. This module is
the entry-point target: importing it (the side effect of ``ep.load()``) registers
the AlphaGalerkin adapters. Kept separate from :mod:`plugins` so that *importing*
``plugins`` stays side-effect free; only this shim registers on import.

Registration is torch-free, so the global entry-point path that fires on any
``eval_harness.bootstrap()`` never pulls AlphaGalerkin's heavy import graph.
"""

from __future__ import annotations

from src.integrations.eval_harness.plugins import register_all

register_all()
