"""Tests for ``scripts/benchmark_codec.py`` CLI subcommands."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_codec.py"


@pytest.fixture(scope="module")
def cli_module():
    """Load scripts/benchmark_codec.py as a module without invoking main()."""
    spec = importlib.util.spec_from_file_location(
        "benchmark_codec_script",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - sanity
        pytest.fail(f"unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_codec_script"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- helpers


def _smoke_yaml() -> str:
    return """\
name: cli_smoke
resolutions:
  - name: r16
    label: 16x16
    height: 16
    width: 16
batch_sizes: [1]
phases: [forward]
n_warmup: 0
n_repeats: 1
device_preference: cpu
track_gpu_memory: false
pattern: motion
data_seed: 1
codec:
  name: codec_cli_smoke
  encoder:
    name: enc
    in_channels: 3
    latent_channels: 32
    n_layers: 1
    d_model: 64
    n_heads: 2
    d_ffn: 128
    downsample_factor: 4
  decoder:
    name: dec
    latent_channels: 32
    out_channels: 3
    n_layers: 1
    d_model: 64
    n_heads: 2
    d_ffn: 128
    upsample_factor: 4
  quantizer:
    name: q
  entropy:
    name: ent
    hyper_channels: 32
    num_filters: 32
  mcts:
    name: mcts
    state_dim: 64
  training:
    name: train
"""


# -------------------------------------------------------- _load_perf_config


class TestLoadConfig:
    def test_yaml_round_trip(self, cli_module, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_smoke_yaml())
        perf, codec = cli_module._load_perf_config(cfg_path)
        assert perf.name == "cli_smoke"
        assert codec is not None
        assert codec.name == "codec_cli_smoke"

    def test_json_supported(self, cli_module, tmp_path: Path) -> None:
        cfg = {
            "name": "json_cfg",
            "resolutions": [{"name": "r", "label": "8x8", "height": 16, "width": 16}],
            "n_warmup": 0,
            "n_repeats": 1,
            "device_preference": "cpu",
        }
        path = tmp_path / "c.json"
        path.write_text(json.dumps(cfg))
        perf, codec = cli_module._load_perf_config(path)
        assert perf.name == "json_cfg"
        assert codec is None  # no codec block

    def test_unknown_suffix_rejected(self, cli_module, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text("anything")
        with pytest.raises(ValueError, match="suffix"):
            cli_module._load_perf_config(path)

    def test_empty_file_rejected(self, cli_module, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            cli_module._load_perf_config(path)

    def test_root_must_be_mapping(self, cli_module, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="mapping"):
            cli_module._load_perf_config(path)


# -------------------------------------------------------------- argparse


class TestArgparse:
    def test_run_subcommand_parses(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args(["run", "--config", "x.yaml", "--output", "out.json"])
        assert args.cmd == "run"
        assert str(args.config) == "x.yaml"
        assert str(args.output) == "out.json"

    def test_record_subcommand_parses(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args(
            [
                "record-baseline",
                "--config",
                "x.yaml",
                "--output",
                "b.json",
                "--hardware-tag",
                "rtx-3060",
            ]
        )
        assert args.cmd == "record-baseline"
        assert args.hardware_tag == "rtx-3060"

    def test_diff_subcommand_parses(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args(
            [
                "diff",
                "--report",
                "r.json",
                "--baseline",
                "b.json",
                "--tolerance",
                "10.0",
            ]
        )
        assert args.cmd == "diff"
        assert args.tolerance == 10.0

    def test_subcommand_required(self, cli_module) -> None:
        parser = cli_module.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# -------------------------------------------------------- end-to-end runs


class TestRunSubcommand:
    def test_run_completes_zero_exit(self, cli_module, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_smoke_yaml())
        out_path = tmp_path / "out.json"
        rc = cli_module.main(["run", "--config", str(cfg_path), "--output", str(out_path)])
        assert rc == 0
        assert out_path.exists()

    def test_record_then_diff(self, cli_module, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_smoke_yaml())

        baseline_path = tmp_path / "baseline.json"
        rc = cli_module.main(
            [
                "record-baseline",
                "--config",
                str(cfg_path),
                "--output",
                str(baseline_path),
                "--hardware-tag",
                "test",
            ]
        )
        assert rc == 0
        assert baseline_path.exists()

        report_path = tmp_path / "report.json"
        rc = cli_module.main(
            [
                "run",
                "--config",
                str(cfg_path),
                "--output",
                str(report_path),
                "--tolerance",
                "99.0",  # extremely loose so CI noise can't fail
            ]
        )
        assert rc == 0

        rc = cli_module.main(
            [
                "diff",
                "--report",
                str(report_path),
                "--baseline",
                str(baseline_path),
                "--tolerance",
                "99.0",
            ]
        )
        assert rc == 0


# ------------------------------------------------------- _git_sha_or_empty


class TestGitSha:
    def test_returns_string(self, cli_module) -> None:
        # We don't assert non-empty because the test runner may invoke us
        # outside a git workdir. We only assert the function never raises.
        sha = cli_module._git_sha_or_empty()
        assert isinstance(sha, str)
