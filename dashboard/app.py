"""AlphaGalerkin E2E Dashboard — main entry point.

A comprehensive Gradio application that exposes all AlphaGalerkin
functionality in a single tabbed interface:

1. **Go AI**        — Human vs AI and AI vs AI with zero-shot board-size transfer
2. **Physics Demo** — Interactive Poisson zero-shot transfer (hf_space demo)
3. **Benchmark**    — FNet O(N log N) vs Softmax O(N²) speed comparison
4. **Architecture** — Galerkin attention, Fourier features, LBB stability visuals
5. **PDE Solver**   — Interactive Poisson equation solver with resolution comparison
6. **PoC Scenarios**— Complexity, LBB stability, and zero-shot transfer validation
7. **Training**     — Architecture summary, loss breakdown, and example training curves
8. **About**        — Project overview and quick-start commands

Usage::

    python dashboard/app.py
    python dashboard/app.py --port 8080 --share

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

# ── Path setup ──────────────────────────────────────────────────────────────
# hf_space/ is inserted FIRST so `import src.X` resolves to hf_space/src/,
# which provides game_manager, rendering, demos, and all shared modules.
# ROOT is inserted SECOND to enable `from dashboard.tabs.X import …`.
ROOT: Final[Path] = Path(__file__).parent.parent
HF_SPACE: Final[Path] = ROOT / "hf_space"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HF_SPACE) not in sys.path:
    sys.path.insert(0, str(HF_SPACE))

# ── Core imports ─────────────────────────────────────────────────────────────
import gradio as gr  # noqa: E402
import structlog  # noqa: E402

from dashboard.config import DEFAULT_CONFIG, AppConfig, DashboardConfig  # noqa: E402
from dashboard.tabs.game_tab import create_game_tab  # noqa: E402
from dashboard.tabs.pde_tab import create_pde_tab  # noqa: E402
from dashboard.tabs.poc_tab import create_poc_tab  # noqa: E402
from dashboard.tabs.training_tab import create_training_tab  # noqa: E402
from dashboard.utils import configure_structlog  # noqa: E402

configure_structlog()
logger = structlog.get_logger(__name__)

# ── Optional: existing hf_space demo tab creators ────────────────────────────
_HF_DEMOS_AVAILABLE = False
try:
    from src.demos.architecture_demo import create_architecture_demo_tab  # type: ignore[import]
    from src.demos.benchmark_demo import create_benchmark_demo_tab  # type: ignore[import]
    from src.demos.physics_demo import create_physics_demo_tab  # type: ignore[import]

    _HF_DEMOS_AVAILABLE = True
    logger.debug("hf_demos_available")
except Exception as _exc:
    logger.warning("hf_demos_unavailable", error=str(_exc))

# ── About page content ────────────────────────────────────────────────────────
_ABOUT_MARKDOWN: Final[str] = """
## About AlphaGalerkin

AlphaGalerkin is a **resolution-independent Go AI** using Continuous Operator
Learning — Galerkin Transformers and FNet mixing — instead of discrete CNNs.

### Key innovations

| Property | Detail |
|---|---|
| Zero-shot transfer | Trained on 9×9, plays 13×13 and 19×19 without retraining |
| Galerkin attention | O(N) via Petrov-Galerkin projection (vs O(N²) softmax) |
| FNet mixing | O(N log N) FFT token mixing for fast MCTS rollouts |
| LBB stability | dim(Key) ≥ dim(Query) guarantees inf-sup condition β > 0 |
| Physics PoC | Poisson MSE = 0.000209 on 19×19 (trained on 9×9) — 240× below threshold |

### Architecture

```
Input positions → Fourier positional encoding (resolution-independent)
              → Strategy body: N × GalerkinLinearAttention  O(N)
              → Tactical head: M × SoftmaxAttention  (exact)
              → FNet mixing  O(N log N) rollout speed
              → Policy head + Value head
```

### Quick-start commands

```bash
# Run this dashboard
python dashboard/app.py

# Self-play training
python -m scripts.train

# Physics PoC (zero-shot transfer)
python -m src.experiments.train_physics

# PoC scenario framework
python -m src.poc.cli run --scenario transfer
```

---
*Built with PyTorch, Gradio, and the AlphaGalerkin framework.*
"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _build_css(app_cfg: AppConfig) -> str:
    """Generate custom CSS from AppConfig values.

    Args:
        app_cfg: Application-level configuration.

    Returns:
        CSS string to inject into the Gradio Blocks.

    """
    return (
        f".tab-nav button {{ "
        f"font-size: {app_cfg.css_tab_font_size}; "
        f"padding: {app_cfg.css_tab_padding}; }}\n"
        "footer { display: none !important; }"
    )


def build_app(cfg: DashboardConfig | None = None) -> gr.Blocks:
    """Construct and return the full Gradio Blocks application.

    Args:
        cfg: Optional DashboardConfig override; uses ``DEFAULT_CONFIG`` when *None*.

    Returns:
        A fully-wired ``gr.Blocks`` instance, ready to ``.launch()``.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    logger.info("dashboard_building")

    with gr.Blocks(title="AlphaGalerkin Dashboard") as demo:
        gr.Markdown(
            "# AlphaGalerkin E2E Dashboard\n"
            "Resolution-independent Go AI · Galerkin Transformers · FNet · Physics PoC"
        )

        create_game_tab(cfg.game)

        if _HF_DEMOS_AVAILABLE:
            create_physics_demo_tab(model=None, device="cpu")
            create_benchmark_demo_tab()
            create_architecture_demo_tab(model=None, device="cpu")
        else:
            with gr.Tab("Demos (unavailable)"):
                gr.Markdown(
                    "Physics, Benchmark, and Architecture demo tabs could not be loaded.\n"
                    "Ensure `hf_space/` is present and all dependencies are installed."
                )

        create_pde_tab(cfg.pde)
        create_poc_tab(cfg.poc)
        create_training_tab(cfg.training)

        with gr.Tab("About"):
            gr.Markdown(_ABOUT_MARKDOWN)

    logger.info("dashboard_built")
    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; uses ``sys.argv[1:]`` when *None*.

    Returns:
        Parsed ``argparse.Namespace``.

    """
    parser = argparse.ArgumentParser(
        description="AlphaGalerkin E2E Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_CONFIG.app.host, help="Bind host")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_CONFIG.app.port, help="Bind port"
    )
    parser.add_argument(
        "--share", action="store_true", default=DEFAULT_CONFIG.app.share,
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--debug", action="store_true", default=DEFAULT_CONFIG.app.debug,
        help="Enable Gradio debug mode",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch the dashboard.

    Args:
        argv: Argument list; uses ``sys.argv[1:]`` when *None*.

    """
    args = _parse_args(argv)
    logger.info(
        "dashboard_starting",
        host=args.host,
        port=args.port,
        share=args.share,
        debug=args.debug,
    )

    app_cfg = AppConfig(
        host=args.host,
        port=args.port,
        share=args.share,
        debug=args.debug,
    )
    cfg = DashboardConfig(app=app_cfg)
    app = build_app(cfg)
    css = _build_css(cfg.app)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        css=css,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "build_app",
    "main",
]
