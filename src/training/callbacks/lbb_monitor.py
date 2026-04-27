"""LBB stability monitoring callback.

Implements ``config/proposals/nsf_sbir.yaml::lbb_stability_monitoring``
by recording the model's LBB constant on every Nth training step,
writing a CSV trace, and (when matplotlib is available) emitting an
HTML report with a step-vs-σ_min plot.

The LBB constant is read directly from
:class:`src.training.callbacks.CallbackContext.metrics`.  The standard
:class:`Trainer` already populates ``lbb_constant`` in its metrics
dict, so this callback works with no upstream changes.

Usage from YAML::

    training:
      callbacks:
        - name: lbb_monitor
          params:
            output_dir: outputs/lbb_traces
            log_interval: 100
            warn_below: 0.05

Or programmatically::

    from src.training.callbacks.lbb_monitor import LBBStabilityCallback

    cb = LBBStabilityCallback(output_dir="outputs/lbb_traces")
    trainer.add_callback(cb)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from src.training.callbacks import Callback, CallbackContext, register_callback

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic configuration
# ---------------------------------------------------------------------------


class LBBMonitorConfig(BaseModel):
    """Configuration for :class:`LBBStabilityCallback`.

    All fields are surfaced; no hardcoded values.  The Pydantic model
    is also used to validate kwargs handed to the callback when
    constructed via ``CallbackSpec`` from YAML.
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(
        default="outputs/lbb_traces",
        description="Directory where the CSV/HTML trace is written.",
    )
    csv_filename: str = Field(
        default="lbb_trace.csv",
        description="File name for the CSV trace (relative to output_dir).",
    )
    html_filename: str = Field(
        default="lbb_trace.html",
        description="File name for the HTML report.",
    )
    log_interval: int = Field(
        default=100,
        ge=1,
        description="Steps between recorded samples.",
    )
    warn_below: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Optional threshold; values below trigger a structured "
            "log warning.  Set to 0.0 to disable."
        ),
    )
    metric_key: str = Field(
        default="lbb_constant",
        description="Metric key to read from CallbackContext.metrics.",
    )
    flush_every: int = Field(
        default=10,
        ge=1,
        description="Flush the CSV writer every N samples.",
    )
    emit_html: bool = Field(
        default=True,
        description="Emit an HTML summary on on_train_end.",
    )


@dataclass(frozen=True)
class LBBSample:
    """One CSV row."""

    step: int
    lbb_constant: float


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@register_callback("lbb_monitor")
class LBBStabilityCallback(Callback):
    """Stream the LBB constant to CSV (and HTML) during training.

    The callback follows the discipline laid out in
    ``docs/doe_genesis/theory.md §2.3``: it records ``σ_min`` of the
    Key→Value projection so that the post-hoc check ``σ_min > β > 0``
    can be verified against the recorded trace.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Validate kwargs through the Pydantic schema so YAML configs
        # error early rather than during the first step.
        self.config = LBBMonitorConfig(**kwargs)

        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._csv_path = output / self.config.csv_filename
        self._html_path = output / self.config.html_filename

        self._samples: list[LBBSample] = []
        self._fh: Any = None
        self._writer: Any = None
        self._steps_since_flush = 0

        self._log = logger.bind(
            callback="lbb_monitor",
            csv_path=str(self._csv_path),
        )

    # ---------------- lifecycle ----------------

    def on_train_start(self, ctx: CallbackContext) -> None:
        # Open the CSV in append mode so resumed training keeps a
        # continuous trace.  Header is written only on first open.
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._fh = self._csv_path.open("a", newline="")
        self._writer = csv.writer(self._fh)
        if write_header:
            self._writer.writerow(["step", "lbb_constant"])
            self._fh.flush()
        self._log.info("lbb_monitor_started")

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.step % self.config.log_interval != 0:
            return
        value = ctx.metrics.get(self.config.metric_key)
        if value is None:
            return
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return
        sample = LBBSample(step=int(ctx.step), lbb_constant=value_f)
        self._samples.append(sample)

        if self._writer is not None:
            self._writer.writerow([sample.step, sample.lbb_constant])
        self._steps_since_flush += 1
        if self._steps_since_flush >= self.config.flush_every and self._fh is not None:
            self._fh.flush()
            self._steps_since_flush = 0

        if self.config.warn_below > 0.0 and sample.lbb_constant < self.config.warn_below:
            self._log.warning(
                "lbb_below_threshold",
                step=sample.step,
                lbb_constant=sample.lbb_constant,
                threshold=self.config.warn_below,
            )

    def on_train_end(self, ctx: CallbackContext) -> None:
        # Always close the CSV
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None

        # Persist a JSON sidecar with summary statistics
        summary = self._compute_summary()
        if summary is not None:
            (self._csv_path.parent / "lbb_summary.json").write_text(json.dumps(summary, indent=2))

        if self.config.emit_html:
            try:
                self._render_html(summary)
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "lbb_html_render_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        self._log.info(
            "lbb_monitor_finished",
            n_samples=len(self._samples),
            html=str(self._html_path) if self.config.emit_html else None,
        )

    # ---------------- helpers ----------------

    def _compute_summary(self) -> dict[str, Any] | None:
        if not self._samples:
            return None
        values = [s.lbb_constant for s in self._samples]
        return {
            "n_samples": len(values),
            "min": float(min(values)),
            "max": float(max(values)),
            "first": values[0],
            "last": values[-1],
            "violations": sum(
                1 for v in values if self.config.warn_below > 0 and v < self.config.warn_below
            ),
            "warn_below": self.config.warn_below,
        }

    def _render_html(self, summary: dict[str, Any] | None) -> None:
        try:
            import matplotlib  # noqa: F401

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self._log.info("lbb_html_skipped_no_matplotlib")
            return

        if not self._samples:
            return
        steps = [s.step for s in self._samples]
        values = [s.lbb_constant for s in self._samples]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.plot(steps, values, marker=".", markersize=2, linewidth=1)
        if self.config.warn_below > 0:
            ax.axhline(
                y=self.config.warn_below,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"warn_below={self.config.warn_below}",
            )
            ax.legend()
        ax.set_xlabel("step")
        ax.set_ylabel("LBB constant (σ_min)")
        ax.set_title("LBB stability trace")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        plot_path = self._html_path.with_suffix(".png")
        fig.savefig(plot_path)
        plt.close(fig)

        summary_html = "<p>No samples collected.</p>"
        if summary is not None:
            summary_html = (
                "<table><tbody>"
                + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary.items())
                + "</tbody></table>"
            )
        self._html_path.write_text(
            f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LBB Stability</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:1em}}
table{{border-collapse:collapse}}td{{border:1px solid #ccc;padding:.25em .5em}}</style>
</head><body>
<h1>LBB Stability Monitor</h1>
<img src="{plot_path.name}" alt="LBB trace" style="max-width:100%">
<h2>Summary</h2>
{summary_html}
</body></html>
"""
        )
