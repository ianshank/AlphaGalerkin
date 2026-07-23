"""Tests for the SBIR P40 benchmark driver script.

Validates the config-loading, override-merging, and registration helpers
that make ``scripts/run_sbir_p40.py`` reusable rather than hardcoded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from src.research.baselines import PINNConfig, SimplePINNSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_sbir_p40.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_sbir_p40_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_sbir_p40_script"] = module
    spec.loader.exec_module(module)
    return module


# Eagerly load the script module so individual test methods can reference
# its functions without needing a per-method fixture parameter.
script_module = _load_script_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_yaml() -> dict[str, Any]:
    return {
        "suite_name": "test_p40",
        "output_dir": "outputs/test",
        "benchmarks": [
            {
                "name": "ns",
                "pde_type": "navier_stokes",
                "domain": {"dim": 2, "min": [0.0, 0.0], "max": [1.0, 1.0]},
                "parameters": {},
                "refinement_levels": [16, 64],
            }
        ],
        "baselines": [
            {"name": "navier_stokes_fdm", "type": "classical"},
            {"name": "pinn_cpu", "type": "ml"},
            {"name": "pinn_p40", "type": "ml"},
        ],
        "pinn_profiles": {
            "p40": {
                "device": "cuda:0",
                "hidden_dim": 128,
                "n_layers": 3,
                "n_epochs": 100,
                "n_collocation": 1000,
                "learning_rate": 5e-4,
            },
            "cpu": {
                "device": "cpu",
                "hidden_dim": 32,
                "n_layers": 2,
                "n_epochs": 10,
                "n_collocation": 100,
            },
        },
    }


@pytest.fixture()
def sample_yaml(tmp_path: Path) -> Path:
    cfg = _minimal_yaml()
    path = tmp_path / "p40_test.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self) -> None:
        args = script_module.parse_args([])
        assert args.config.endswith("sbir_p40.yaml")
        assert args.output_dir is None
        assert args.device is None
        assert args.n_epochs is None
        assert args.skip_cpu is False
        assert args.require_cuda is False

    def test_overrides(self) -> None:
        args = script_module.parse_args(
            [
                "--config",
                "x.yaml",
                "--output-dir",
                "out",
                "--device",
                "cuda:1",
                "--n-epochs",
                "500",
                "--n-collocation",
                "5000",
                "--refinement-levels",
                "256,1024",
                "--skip-cpu",
                "--require-cuda",
            ]
        )
        assert args.config == "x.yaml"
        assert args.output_dir == "out"
        assert args.device == "cuda:1"
        assert args.n_epochs == 500
        assert args.n_collocation == 5000
        assert args.refinement_levels == "256,1024"
        assert args.skip_cpu is True
        assert args.require_cuda is True


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_minimal(self, sample_yaml: Path) -> None:
        cfg = script_module.load_config(sample_yaml)
        assert cfg["suite_name"] == "test_p40"
        assert "benchmarks" in cfg
        assert "baselines" in cfg
        assert "pinn_profiles" in cfg

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            script_module.load_config(path)

    @pytest.mark.parametrize("missing", ["benchmarks", "baselines", "pinn_profiles"])
    def test_missing_required_key_raises(self, tmp_path: Path, missing: str) -> None:
        cfg = _minimal_yaml()
        del cfg[missing]
        path = tmp_path / "incomplete.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match=missing):
            script_module.load_config(path)


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def _args(self, **overrides: Any) -> SimpleNamespace:
        defaults = {
            "device": None,
            "n_epochs": None,
            "n_collocation": None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_no_overrides_returns_copy(self) -> None:
        profiles = _minimal_yaml()["pinn_profiles"]
        out = script_module.apply_overrides(profiles, self._args())
        assert out == profiles
        assert out is not profiles  # must be a copy
        assert out["p40"] is not profiles["p40"]

    def test_device_override(self) -> None:
        profiles = _minimal_yaml()["pinn_profiles"]
        out = script_module.apply_overrides(profiles, self._args(device="cuda:1"))
        assert out["p40"]["device"] == "cuda:1"
        # CPU profile is untouched
        assert out["cpu"]["device"] == "cpu"

    def test_n_epochs_override(self) -> None:
        profiles = _minimal_yaml()["pinn_profiles"]
        out = script_module.apply_overrides(profiles, self._args(n_epochs=42))
        assert out["p40"]["n_epochs"] == 42
        assert out["cpu"]["n_epochs"] == 10

    def test_n_collocation_override(self) -> None:
        profiles = _minimal_yaml()["pinn_profiles"]
        out = script_module.apply_overrides(profiles, self._args(n_collocation=99))
        assert out["p40"]["n_collocation"] == 99


# ---------------------------------------------------------------------------
# apply_benchmark_overrides
# ---------------------------------------------------------------------------


class TestApplyBenchmarkOverrides:
    def test_no_override_returns_input(self) -> None:
        cfg = _minimal_yaml()
        out = script_module.apply_benchmark_overrides(
            cfg["benchmarks"], SimpleNamespace(refinement_levels=None)
        )
        assert out == cfg["benchmarks"]

    def test_override_replaces_levels(self) -> None:
        cfg = _minimal_yaml()
        out = script_module.apply_benchmark_overrides(
            cfg["benchmarks"], SimpleNamespace(refinement_levels="64,256,1024")
        )
        assert out[0]["refinement_levels"] == [64, 256, 1024]

    def test_override_strips_whitespace(self) -> None:
        cfg = _minimal_yaml()
        out = script_module.apply_benchmark_overrides(
            cfg["benchmarks"], SimpleNamespace(refinement_levels=" 64 , 256 , ")
        )
        assert out[0]["refinement_levels"] == [64, 256]


# ---------------------------------------------------------------------------
# filter_baselines
# ---------------------------------------------------------------------------


class TestFilterBaselines:
    def test_skip_cpu_removes_pinn_cpu(self) -> None:
        cfg = _minimal_yaml()
        out = script_module.filter_baselines(cfg["baselines"], skip_cpu=True)
        names = [b["name"] for b in out]
        assert "pinn_cpu" not in names
        assert "pinn_p40" in names
        assert "navier_stokes_fdm" in names

    def test_no_skip_keeps_all(self) -> None:
        cfg = _minimal_yaml()
        out = script_module.filter_baselines(cfg["baselines"], skip_cpu=False)
        assert out == cfg["baselines"]


# ---------------------------------------------------------------------------
# build_pinn_config + register_pinn_profiles
# ---------------------------------------------------------------------------


class TestPinnConfigBuilders:
    def test_build_pinn_config_from_profile(self) -> None:
        profile = _minimal_yaml()["pinn_profiles"]["cpu"]
        cfg = script_module.build_pinn_config(profile)
        assert isinstance(cfg, PINNConfig)
        assert cfg.device == "cpu"
        assert cfg.hidden_dim == 32
        assert cfg.n_epochs == 10

    def test_register_pinn_profiles_creates_named_solvers(self) -> None:
        profiles = _minimal_yaml()["pinn_profiles"]
        registry: dict[str, type[SimplePINNSolver]] = {}
        script_module.register_pinn_profiles(profiles, registry)
        assert "pinn_p40" in registry
        assert "pinn_cpu" in registry
        assert issubclass(registry["pinn_p40"], SimplePINNSolver)
        # The bound class instantiates without args, using the YAML config.
        instance = registry["pinn_cpu"]()
        assert instance.name == "pinn_cpu"
        assert instance.device_preference == "cpu"
        assert instance.config.hidden_dim == 32

    def test_bound_solver_accepts_kwargs(self) -> None:
        """``get_solver(name, **overrides)`` must not TypeError on the bound class.

        Regression for the gemini-code-assist HIGH finding on PR #83: the
        original ``_bound_init(self)`` signature would raise on any
        keyword argument, so ``get_solver("pinn_cpu", hidden_dim=...)``
        from the canonical registry path failed.
        """
        profiles = _minimal_yaml()["pinn_profiles"]
        registry: dict[str, type[SimplePINNSolver]] = {}
        script_module.register_pinn_profiles(profiles, registry)

        # Per-call override should win over the YAML profile default.
        instance = registry["pinn_cpu"](hidden_dim=99)
        assert instance.hidden_dim == 99
        # YAML profile fields not overridden remain.
        assert instance.config.n_epochs == 10
