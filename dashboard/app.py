"""AlphaGalerkin E2E Dashboard.

A comprehensive Gradio application that exposes all AlphaGalerkin functionality:

  1. Go AI           — Human vs AI and AI vs AI with zero-shot board-size transfer
  2. Physics Demo    — Interactive Poisson zero-shot transfer (hf_space demo)
  3. Benchmark       — FNet O(N log N) vs Softmax O(N²) speed comparison
  4. Architecture    — Galerkin attention, Fourier features, LBB stability visuals
  5. PDE Solver      — Interactive Poisson equation solver with resolution comparison
  6. PoC Scenarios   — Complexity, LBB stability, and zero-shot transfer validation
  7. Training        — Architecture summary, loss breakdown, and example training curves
  8. About           — Project overview and key links

Run:
    python dashboard/app.py
    python -m dashboard.app
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# hf_space/ is inserted FIRST so that `import src.X` resolves to hf_space/src/,
# which contains game_manager, rendering, demos, plus all shared modules
# (mcts, modeling, poc, pde, physics, training, …).
ROOT = Path(__file__).parent.parent
HF_SPACE = ROOT / "hf_space"

sys.path.insert(0, str(ROOT))       # enables `from dashboard.tabs.X import …`
sys.path.insert(0, str(HF_SPACE))   # wins the `src` namespace over ROOT/src/

# ── Imports ──────────────────────────────────────────────────────────────────
import gradio as gr
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=False),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)

# ── Dashboard tab modules ────────────────────────────────────────────────────
from dashboard.tabs.game_tab import create_game_tab
from dashboard.tabs.pde_tab import create_pde_tab
from dashboard.tabs.poc_tab import create_poc_tab
from dashboard.tabs.training_tab import create_training_tab

# ── Existing hf_space demo tab creators ─────────────────────────────────────
try:
    from src.demos.architecture_demo import create_architecture_demo_tab
    from src.demos.benchmark_demo import create_benchmark_demo_tab
    from src.demos.physics_demo import create_physics_demo_tab

    _HF_DEMOS_AVAILABLE = True
except Exception as _exc:
    logger.warning("hf_demos_unavailable", error=str(_exc))
    _HF_DEMOS_AVAILABLE = False

# ── Custom CSS ───────────────────────────────────────────────────────────────
_CSS = """
.tab-nav button { font-size: 14px; padding: 8px 16px; }
footer { display: none !important; }
"""

# ── About markdown ────────────────────────────────────────────────────────────
_ABOUT = """
## About AlphaGalerkin

AlphaGalerkin is a **resolution-independent Go AI** that uses Continuous Operator
Learning — Galerkin Transformers and FNet mixing — instead of discrete CNNs.

### Key innovations

| Property | Detail |
|---|---|
| Zero-shot transfer | Trained on 9×9, plays 13×13 and 19×19 without retraining |
| Galerkin attention | O(N) complexity via Petrov-Galerkin projection (vs O(N²) softmax) |
| FNet mixing | O(N log N) FFT-based token mixing for fast MCTS rollouts |
| LBB stability | dim(Key) ≥ dim(Query) guarantees inf-sup condition β > 0 |
| Physics PoC | Poisson equation MSE = 0.000209 on 19×19 (trained on 9×9) — 240× below threshold |

### Architecture

```
Input positions  →  Fourier positional encoding (resolution-independent)
                 →  Strategy body: N × GalerkinLinearAttention  O(N)
                 →  Tactical head: M × SoftmaxAttention  (exact)
                 →  FNet mixing (optional, O(N log N) rollout speed)
                 →  Policy head + Value head
```

### Key milestones

- **2026-01-26** — Zero-shot transfer validated: physics MSE 0.000209 on 19×19
- **2026-02-01** — Chess support (119-plane AlphaZero encoding)
- **2026-03-30** — CI hardening, config-driven LBB loss, checkpoint migration
- **2026-04-07** — 390+ new tests, distributed trainer, ONNX export

### Repository layout

```
src/           Core library (modeling, mcts, pde, training, physics, …)
hf_space/      HuggingFace Spaces demo (game UI, demos)
dashboard/     This E2E dashboard
scripts/       Training and evaluation CLIs
config/        Hydra / Pydantic YAML configs
tests/         Unit, functional, and integration tests
docs/          Architecture diagrams and SBIR proposals
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


# ── Build the Gradio app ──────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    """Construct and return the full Gradio Blocks application."""
    with gr.Blocks(title="AlphaGalerkin Dashboard", css=_CSS) as demo:
        gr.Markdown(
            "# AlphaGalerkin E2E Dashboard\n"
            "Resolution-independent Go AI · Galerkin Transformers · FNet · Physics PoC"
        )

        # ── Tab 1: Go Game ───────────────────────────────────────────────────
        create_game_tab()

        # ── Tabs 2–4: existing hf_space demos ───────────────────────────────
        if _HF_DEMOS_AVAILABLE:
            create_physics_demo_tab(model=None, device="cpu")
            create_benchmark_demo_tab()
            create_architecture_demo_tab(model=None, device="cpu")
        else:
            with gr.Tab("Demos (unavailable)"):
                gr.Markdown(
                    "Physics, Benchmark, and Architecture demo tabs could not be loaded.\n"
                    "Ensure `hf_space/` is present and dependencies are installed."
                )

        # ── Tab 5: PDE Solver ────────────────────────────────────────────────
        create_pde_tab()

        # ── Tab 6: PoC Scenarios ─────────────────────────────────────────────
        create_poc_tab()

        # ── Tab 7: Training Dashboard ────────────────────────────────────────
        create_training_tab()

        # ── Tab 8: About ─────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(_ABOUT)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaGalerkin E2E Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    args = parser.parse_args()

    logger.info(
        "dashboard_starting",
        host=args.host,
        port=args.port,
        share=args.share,
    )

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )
