"""Tests for :class:`LBBStabilityCallback`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.training.callbacks import (
    Callback,
    CallbackContext,
    CallbackRegistry,
    CallbackSpec,
    build_callbacks_from_specs,
)
from src.training.callbacks.lbb_monitor import (
    LBBMonitorConfig,
    LBBStabilityCallback,
)

# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class TestLBBMonitorConfig:
    def test_defaults(self) -> None:
        cfg = LBBMonitorConfig()
        assert cfg.log_interval >= 1
        assert cfg.warn_below == 0.0
        assert cfg.emit_html is True

    def test_negative_log_interval_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LBBMonitorConfig(log_interval=0)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LBBMonitorConfig(unknown=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Callback lifecycle
# ---------------------------------------------------------------------------


class TestLBBCallback:
    def test_emits_csv_after_train(self, tmp_path: Path) -> None:
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            emit_html=False,
        )

        cb.on_train_start(CallbackContext(step=0))
        for step, value in enumerate([0.1, 0.2, 0.3]):
            cb.on_step_end(CallbackContext(step=step, metrics={"lbb_constant": value}))
        cb.on_train_end(CallbackContext(step=2))

        csv_path = tmp_path / "lbb_trace.csv"
        assert csv_path.exists()
        text = csv_path.read_text().strip().splitlines()
        # header + 3 rows
        assert len(text) == 4
        assert text[0] == "step,lbb_constant"

    def test_skips_steps_outside_interval(self, tmp_path: Path) -> None:
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=2,
            emit_html=False,
        )
        cb.on_train_start(CallbackContext(step=0))
        for step, value in enumerate([0.1, 0.2, 0.3, 0.4]):
            cb.on_step_end(CallbackContext(step=step, metrics={"lbb_constant": value}))
        cb.on_train_end(CallbackContext(step=3))

        csv_path = tmp_path / "lbb_trace.csv"
        rows = csv_path.read_text().strip().splitlines()
        # steps 0 and 2 -> 2 data rows + header
        assert len(rows) == 3

    def test_missing_metric_is_silent(self, tmp_path: Path) -> None:
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            emit_html=False,
        )
        cb.on_train_start(CallbackContext(step=0))
        cb.on_step_end(CallbackContext(step=0, metrics={}))  # no lbb_constant
        cb.on_train_end(CallbackContext(step=0))

        rows = (tmp_path / "lbb_trace.csv").read_text().strip().splitlines()
        assert len(rows) == 1  # only header

    def test_warn_below_logs(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            warn_below=0.5,
            emit_html=False,
        )
        cb.on_train_start(CallbackContext(step=0))
        cb.on_step_end(
            CallbackContext(step=0, metrics={"lbb_constant": 0.1}),  # below threshold
        )
        cb.on_train_end(CallbackContext(step=0))

        summary = json.loads((tmp_path / "lbb_summary.json").read_text())
        assert summary["violations"] == 1
        assert summary["warn_below"] == 0.5

    def test_summary_written(self, tmp_path: Path) -> None:
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            emit_html=False,
        )
        cb.on_train_start(CallbackContext(step=0))
        for step, value in enumerate([0.5, 0.3, 0.7]):
            cb.on_step_end(CallbackContext(step=step, metrics={"lbb_constant": value}))
        cb.on_train_end(CallbackContext(step=2))

        summary = json.loads((tmp_path / "lbb_summary.json").read_text())
        assert summary["n_samples"] == 3
        assert summary["min"] == pytest.approx(0.3)
        assert summary["max"] == pytest.approx(0.7)
        assert summary["first"] == pytest.approx(0.5)
        assert summary["last"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Registry roundtrip via spec
# ---------------------------------------------------------------------------


class TestLBBSpecRoundtrip:
    def test_resolves_via_registry(self, tmp_path: Path) -> None:
        # The auto-import in build_callbacks_from_specs ensures the
        # lbb_monitor module is loaded and the decorator registered.
        spec = CallbackSpec(
            name="lbb_monitor",
            params={
                "output_dir": str(tmp_path),
                "log_interval": 1,
                "emit_html": False,
            },
        )
        callbacks = build_callbacks_from_specs([spec])
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], LBBStabilityCallback)
        assert isinstance(callbacks[0], Callback)

    def test_registered_in_registry(self) -> None:
        # Trigger module import in case it hasn't happened yet
        from src.training.callbacks.base import _ensure_builtin_callbacks_imported

        _ensure_builtin_callbacks_imported()
        assert "lbb_monitor" in CallbackRegistry()


class TestLBBHTMLRendering:
    """Exercises the HTML/PNG rendering path when matplotlib is present."""

    def test_html_emitted(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            warn_below=0.1,
            emit_html=True,
        )
        cb.on_train_start(CallbackContext(step=0))
        for step, value in enumerate([0.5, 0.6, 0.4, 0.3]):
            cb.on_step_end(CallbackContext(step=step, metrics={"lbb_constant": value}))
        cb.on_train_end(CallbackContext(step=3))

        html = tmp_path / "lbb_trace.html"
        png = tmp_path / "lbb_trace.png"
        assert html.exists()
        assert png.exists()
        assert "LBB Stability Monitor" in html.read_text()

    def test_html_skipped_with_no_samples(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            emit_html=True,
        )
        cb.on_train_start(CallbackContext(step=0))
        # No on_step_end calls
        cb.on_train_end(CallbackContext(step=0))
        # HTML render is a no-op when there are no samples
        assert not (tmp_path / "lbb_trace.html").exists()

    def test_skip_emit_html_flag(self, tmp_path: Path) -> None:
        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            emit_html=False,
        )
        cb.on_train_start(CallbackContext(step=0))
        cb.on_step_end(CallbackContext(step=0, metrics={"lbb_constant": 0.5}))
        cb.on_train_end(CallbackContext(step=0))
        # When emit_html=False, no HTML file is produced
        assert not (tmp_path / "lbb_trace.html").exists()

    def test_html_skipped_when_matplotlib_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If matplotlib import fails, ``on_train_end`` returns silently."""
        import builtins

        original_import = builtins.__import__

        def _block_matplotlib(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("forced missing matplotlib for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_matplotlib)

        cb = LBBStabilityCallback(
            output_dir=str(tmp_path),
            log_interval=1,
            emit_html=True,
        )
        cb.on_train_start(CallbackContext(step=0))
        cb.on_step_end(CallbackContext(step=0, metrics={"lbb_constant": 0.5}))
        # Must not raise even though matplotlib is unavailable.
        cb.on_train_end(CallbackContext(step=0))
        assert not (tmp_path / "lbb_trace.html").exists()
        # Summary JSON is independent of matplotlib and still present.
        assert (tmp_path / "lbb_summary.json").exists()
