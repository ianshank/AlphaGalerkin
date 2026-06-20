"""Tests for :mod:`src.video_compression.metrics.baselines`.

Every test that would otherwise require a real ffmpeg binary
monkeypatches :func:`_run_subprocess` (and :func:`shutil.which`) so the
suite passes on bare CPU CI without the [video] extra installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from src.video_compression.metrics import baselines as baselines_mod
from src.video_compression.metrics.baselines import (
    DEFAULT_FFMPEG_BIN,
    BaselineRunResult,
    FFmpegBaselineConfig,
    FFmpegBaselineRunner,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class TestFFmpegBaselineConfig:
    def test_defaults_match_documented_constants(self) -> None:
        cfg = FFmpegBaselineConfig(name="cfg")
        assert cfg.codec == "libx265"
        assert cfg.crf == 28
        assert cfg.preset == "medium"
        assert cfg.pixel_format == "yuv420p"
        assert cfg.ffmpeg_bin == DEFAULT_FFMPEG_BIN
        assert cfg.subprocess_timeout_s == pytest.approx(600.0)

    def test_av1_can_omit_crf(self) -> None:
        cfg = FFmpegBaselineConfig(
            name="cfg",
            codec="libaom-av1",
            crf=None,
            extra_encode_flags=["-cq-level", "30"],
        )
        assert cfg.crf is None

    def test_x265_requires_crf(self) -> None:
        with pytest.raises(ValidationError, match="requires a CRF"):
            FFmpegBaselineConfig(name="cfg", codec="libx265", crf=None)

    def test_vp9_requires_crf(self) -> None:
        with pytest.raises(ValidationError, match="requires a CRF"):
            FFmpegBaselineConfig(name="cfg", codec="libvpx-vp9", crf=None)

    def test_unknown_field_ignored(self) -> None:
        cfg = FFmpegBaselineConfig.model_validate(
            {"name": "cfg", "future_field_v2": "x"},
        )
        assert cfg.name == "cfg"

    def test_subprocess_timeout_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            FFmpegBaselineConfig(name="cfg", subprocess_timeout_s=0.0)


# --------------------------------------------------------------------------
# Runner: skip semantics
# --------------------------------------------------------------------------


class TestRunnerSkipSemantics:
    def test_skips_when_ffmpeg_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(baselines_mod.shutil, "which", lambda _name: None)
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=tmp_path / "fake.y4m",
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            n_frames=30,
        )
        assert result.status == "skipped"
        assert "not on PATH" in result.reason
        assert result.entry is None

    def test_skips_when_sequence_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _name: "/usr/bin/ffmpeg",
        )
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=tmp_path / "does_not_exist.y4m",
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            n_frames=30,
        )
        assert result.status == "skipped"
        assert "not found" in result.reason

    def test_is_ffmpeg_available_reflects_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        monkeypatch.setattr(baselines_mod.shutil, "which", lambda _n: None)
        assert runner.is_ffmpeg_available() is False
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/x/y/ffmpeg",
        )
        assert runner.is_ffmpeg_available() is True


# --------------------------------------------------------------------------
# Runner: command-building
# --------------------------------------------------------------------------


class TestCommandBuilding:
    def test_encode_cmd_includes_codec_and_crf(self, tmp_path: Path) -> None:
        runner = FFmpegBaselineRunner(
            FFmpegBaselineConfig(
                name="cfg",
                codec="libx265",
                crf=22,
                preset="slow",
            ),
        )
        src = tmp_path / "input.y4m"
        out = tmp_path / "encoded.mkv"
        cmd = runner._build_encode_cmd(src, out)
        assert "-c:v" in cmd and "libx265" in cmd
        assert "-crf" in cmd and "22" in cmd
        assert "-preset" in cmd and "slow" in cmd
        assert cmd[-1] == str(out)

    def test_encode_cmd_omits_crf_when_none(self, tmp_path: Path) -> None:
        runner = FFmpegBaselineRunner(
            FFmpegBaselineConfig(
                name="cfg",
                codec="libaom-av1",
                crf=None,
                extra_encode_flags=["-cq-level", "30"],
            ),
        )
        cmd = runner._build_encode_cmd(tmp_path / "in.y4m", tmp_path / "o.mkv")
        assert "-crf" not in cmd
        assert "-cq-level" in cmd

    def test_decode_cmd_uses_pixel_format(self, tmp_path: Path) -> None:
        runner = FFmpegBaselineRunner(
            FFmpegBaselineConfig(name="cfg", pixel_format="yuv444p"),
        )
        cmd = runner._build_decode_cmd(tmp_path / "e.mkv", tmp_path / "d.y4m")
        assert "-pix_fmt" in cmd and "yuv444p" in cmd


# --------------------------------------------------------------------------
# Runner: success and failure paths via mocked subprocess
# --------------------------------------------------------------------------


def _fake_subprocess_factory(
    *,
    encoded_size_bytes: int,
    encode_rc: int = 0,
    decode_rc: int = 0,
    encode_stderr: bytes = b"",
    decode_stderr: bytes = b"",
) -> Any:
    """Return a function that emulates ffmpeg encode + decode subprocesses.

    On the first call it writes ``encoded_size_bytes`` of zeroes to
    ``cmd[-1]`` (the encoded output path); on the second it writes a
    1-byte placeholder to the decoded path. This is enough to exercise
    the bpp computation and the "decoded file exists" branch without
    invoking ffmpeg.
    """
    state = {"call": 0}

    def fake_run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[bytes]:
        state["call"] += 1
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if state["call"] == 1:
            # encode
            out_path.write_bytes(b"\0" * encoded_size_bytes)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=encode_rc,
                stdout=b"",
                stderr=encode_stderr,
            )
        else:
            # decode
            out_path.write_bytes(b"\0")
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=decode_rc,
                stdout=b"",
                stderr=decode_stderr,
            )

    return fake_run


class TestRunnerExecution:
    def test_success_without_quality_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/usr/bin/ffmpeg",
        )
        # Encode produces 2880 bytes for 352x288x1 = 101 376 px, giving
        # 0.227 bpp — a realistic-looking number.
        monkeypatch.setattr(
            baselines_mod,
            "_run_subprocess",
            _fake_subprocess_factory(encoded_size_bytes=2880),
        )
        # Force the (PyAV) loader to return None so the runner records
        # entry without psnr/ms_ssim — the no-PyAV CI path.
        monkeypatch.setattr(
            baselines_mod,
            "_load_y4m_to_tensor",
            lambda *_a, **_kw: None,
        )
        src = tmp_path / "in.y4m"
        src.write_bytes(b"YUV4MPEG2 W352 H288 F30:1 Ip A1:1\n")  # marker
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=src,
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            n_frames=1,
        )
        assert result.status == "ok"
        assert result.entry is not None
        assert result.entry.codec == "libx265"
        # bpp = 2880 * 8 / (352 * 288 * 1) = 0.2273
        assert result.entry.bpp == pytest.approx(0.227, abs=1e-3)
        assert result.entry.psnr_db is None
        assert result.entry.ms_ssim is None
        assert result.entry.cell_key.startswith("akiyo|352x288|")

    def test_success_with_quality_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/usr/bin/ffmpeg",
        )
        monkeypatch.setattr(
            baselines_mod,
            "_run_subprocess",
            _fake_subprocess_factory(encoded_size_bytes=4096),
        )
        # Build a deterministic original/reconstruction pair so PSNR has
        # a known finite value (not infinity).
        original = torch.zeros(2, 3, 32, 32)
        reconstruction = torch.full((2, 3, 32, 32), 0.05)
        # Both calls return the same shape; PSNR(original, reconstruction)
        # will be finite. Toggle which one is returned per call.
        load_calls = {"n": 0}

        def fake_load(*_a: Any, **_kw: Any) -> torch.Tensor:
            load_calls["n"] += 1
            return original if load_calls["n"] == 1 else reconstruction

        monkeypatch.setattr(baselines_mod, "_load_y4m_to_tensor", fake_load)
        src = tmp_path / "in.y4m"
        src.write_bytes(b"YUV4MPEG2 W32 H32 F30:1 Ip A1:1\n")
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=src,
            sequence_id="tiny",
            width=32,
            height=32,
            fps=30.0,
            n_frames=2,
        )
        assert result.status == "ok"
        assert result.entry is not None
        assert result.entry.psnr_db is not None
        assert result.entry.psnr_db > 0.0
        assert result.entry.ms_ssim is not None
        assert 0.0 <= result.entry.ms_ssim <= 1.0

    def test_success_with_provided_original_tensor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/usr/bin/ffmpeg",
        )
        monkeypatch.setattr(
            baselines_mod,
            "_run_subprocess",
            _fake_subprocess_factory(encoded_size_bytes=4096),
        )
        # Original passed in by the caller; only the decoded tensor needs
        # to come from the loader.
        decoded = torch.full((2, 3, 16, 16), 0.5)
        monkeypatch.setattr(
            baselines_mod,
            "_load_y4m_to_tensor",
            lambda *_a, **_kw: decoded,
        )
        src = tmp_path / "in.y4m"
        src.write_bytes(b"YUV4MPEG2\n")
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=src,
            sequence_id="tiny",
            width=16,
            height=16,
            fps=30.0,
            n_frames=2,
            original_tensor=torch.zeros(2, 3, 16, 16),
        )
        assert result.status == "ok"
        assert result.entry is not None
        assert result.entry.psnr_db is not None

    def test_success_with_zero_frame_decoded_returns_no_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Simulates the encoder dropping every frame so the decoded
        # tensor has shape[0] == 0; the runner must skip quality
        # computation rather than crash.
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/usr/bin/ffmpeg",
        )
        monkeypatch.setattr(
            baselines_mod,
            "_run_subprocess",
            _fake_subprocess_factory(encoded_size_bytes=4096),
        )
        # Original has 1 frame; decoded has 0. min(...) == 0 -> short-circuit.
        load_calls = {"n": 0}

        def fake_load(*_a: Any, **_kw: Any) -> torch.Tensor:
            load_calls["n"] += 1
            if load_calls["n"] == 1:
                return torch.zeros(1, 3, 8, 8)
            return torch.zeros(0, 3, 8, 8)

        monkeypatch.setattr(baselines_mod, "_load_y4m_to_tensor", fake_load)
        src = tmp_path / "in.y4m"
        src.write_bytes(b"x")
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=src,
            sequence_id="tiny",
            width=8,
            height=8,
            fps=30.0,
            n_frames=1,
        )
        assert result.status == "ok"
        assert result.entry is not None
        # No frames in common -> psnr/ms_ssim are None even though encode
        # succeeded and bpp is recorded.
        assert result.entry.psnr_db is None
        assert result.entry.ms_ssim is None

    def test_encode_failure_returns_failed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/usr/bin/ffmpeg",
        )

        def fake_run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout=b"",
                stderr=b"encode error",
            )

        monkeypatch.setattr(baselines_mod, "_run_subprocess", fake_run)
        src = tmp_path / "in.y4m"
        src.write_bytes(b"x")
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=src,
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            n_frames=1,
        )
        assert result.status == "failed"
        assert "encode failed" in result.reason

    def test_decode_failure_returns_failed_but_keeps_encoded_bytes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/usr/bin/ffmpeg",
        )
        state = {"call": 0}

        def fake_run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[bytes]:
            state["call"] += 1
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            if state["call"] == 1:
                # encode succeeds
                out.write_bytes(b"\0" * 1024)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
            # decode fails
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=2,
                stdout=b"",
                stderr=b"decode error",
            )

        monkeypatch.setattr(baselines_mod, "_run_subprocess", fake_run)
        src = tmp_path / "in.y4m"
        src.write_bytes(b"x")
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=src,
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            n_frames=1,
        )
        assert result.status == "failed"
        assert "decode failed" in result.reason
        assert result.encoded_bytes == 1024  # encode-side bytes preserved

    def test_encode_timeout_returns_failed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            baselines_mod.shutil,
            "which",
            lambda _n: "/usr/bin/ffmpeg",
        )

        def fake_run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[bytes]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s)

        monkeypatch.setattr(baselines_mod, "_run_subprocess", fake_run)
        src = tmp_path / "in.y4m"
        src.write_bytes(b"x")
        runner = FFmpegBaselineRunner(FFmpegBaselineConfig(name="cfg"))
        result = runner.run(
            sequence_path=src,
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            n_frames=1,
        )
        assert result.status == "failed"
        assert "encode timeout" in result.reason


# --------------------------------------------------------------------------
# Helpers: bytes -> bpp
# --------------------------------------------------------------------------


class TestBytesToBpp:
    def test_canonical_conversion(self) -> None:
        # 1 KiB across 100x100x10 = 100 000 px → 0.0819 bpp
        bpp = FFmpegBaselineRunner._bytes_to_bpp(1024, width=100, height=100, n_frames=10)
        assert bpp == pytest.approx(8192 / 100_000)

    def test_zero_dims_raise(self) -> None:
        with pytest.raises(ValueError):
            FFmpegBaselineRunner._bytes_to_bpp(1024, width=0, height=100, n_frames=1)
        with pytest.raises(ValueError):
            FFmpegBaselineRunner._bytes_to_bpp(1024, width=100, height=0, n_frames=1)
        with pytest.raises(ValueError):
            FFmpegBaselineRunner._bytes_to_bpp(1024, width=100, height=100, n_frames=0)

    def test_cell_key_format(self) -> None:
        key = FFmpegBaselineRunner._build_cell_key(
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            codec="libx265",
            crf=28,
            preset="medium",
        )
        assert key == "akiyo|352x288|30|libx265|medium|crf28"

    def test_cell_key_distinguishes_crf_sweep(self) -> None:
        # A CRF sweep at the same (sequence, res, fps) must produce
        # distinct cell_keys; otherwise H265BaselineRegistry silently
        # drops the earlier entry on insert.
        keys = {
            FFmpegBaselineRunner._build_cell_key(
                sequence_id="akiyo",
                width=352,
                height=288,
                fps=30.0,
                codec="libx265",
                crf=crf,
                preset="medium",
            )
            for crf in (22, 28, 35, 38)
        }
        assert len(keys) == 4

    def test_cell_key_distinguishes_codec(self) -> None:
        keys = {
            FFmpegBaselineRunner._build_cell_key(
                sequence_id="akiyo",
                width=352,
                height=288,
                fps=30.0,
                codec=codec,
                crf=28,
                preset="medium",
            )
            for codec in ("libx265", "libaom-av1", "libvpx-vp9")
        }
        assert len(keys) == 3

    def test_cell_key_norate_when_crf_none(self) -> None:
        key = FFmpegBaselineRunner._build_cell_key(
            sequence_id="akiyo",
            width=352,
            height=288,
            fps=30.0,
            codec="libaom-av1",
            crf=None,
            preset="medium",
        )
        assert key.endswith("|norate")


# --------------------------------------------------------------------------
# Result dataclass
# --------------------------------------------------------------------------


class TestResult:
    def test_result_is_frozen(self) -> None:
        r = BaselineRunResult(
            status="skipped",
            entry=None,
            encoded_bytes=None,
            encode_seconds=None,
            decode_seconds=None,
            reason="x",
        )
        with pytest.raises(Exception):
            r.status = "ok"  # type: ignore[misc]
