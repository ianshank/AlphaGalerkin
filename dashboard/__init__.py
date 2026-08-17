"""AlphaGalerkin E2E Dashboard.

A comprehensive Gradio application exposing all AlphaGalerkin functionality
through an 8-tab interactive interface.

Quick start::

    python dashboard/app.py

Public API::

    from dashboard.app import build_app
    demo = build_app()
    demo.launch()

"""

from __future__ import annotations

try:
    from dashboard.app import build_app, main
except ImportError:
    build_app = None  # type: ignore[assignment]
    main = None  # type: ignore[assignment]

from dashboard.config import DEFAULT_CONFIG, DashboardConfig

__all__ = [
    "DEFAULT_CONFIG",
    "DashboardConfig",
    "build_app",
    "main",
]
