r"""SBIR benchmark suite — Tesla P40 high-resolution run.

Drives the canonical SimplePINNSolver with PINN profiles loaded from
``config/benchmarks/sbir_p40.yaml``. Compares NS-FDM against the GPU
PINN at high resolution; the CPU PINN row uses reduced n_epochs so a
meaningful CPU-vs-GPU timing comparison fits in one report.

Replaces the previous ~260-line subclass-based fork: the canonical
solver now honours ``PINNConfig.device`` and auto-detects 2-channel
output for Navier-Stokes, so this script is now config-driven.

Usage::

    cd <repo_root>
    python -u -m scripts.run_sbir_p40
    python -u -m scripts.run_sbir_p40 --device cuda:1 --n-epochs 1000
    python -u -m scripts.run_sbir_p40 --output-dir outputs/p40_quick \
        --refinement-levels 256,1024 --skip-cpu
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.research import baselines as _baselines
from src.research.baselines import PINNConfig, SimplePINNSolver
from src.research.pde_benchmarks import PDEBenchmarkRunner

DEFAULT_CONFIG = "config/benchmarks/sbir_p40.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Every knob defaults to "use the value from the YAML"; flags only
    override that default. This keeps the canonical config in one place
    while letting the user A/B without editing files.
    """
    parser = argparse.ArgumentParser(
        description="Run the SBIR P40 high-resolution benchmark suite",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to benchmark YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory; overrides YAML output_dir"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="GPU device label (e.g. 'cuda:0', 'cuda:1') for the P40 PINN row",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=None, help="Override PINN n_epochs on the GPU profile"
    )
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=None,
        help="Override PINN n_collocation on the GPU profile",
    )
    parser.add_argument(
        "--refinement-levels",
        type=str,
        default=None,
        help="Comma-separated DOF levels (overrides YAML)",
    )
    parser.add_argument("--skip-cpu", action="store_true", help="Drop the CPU PINN comparison row")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if CUDA is unavailable (default: warn only)",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    """Load and lightly validate the SBIR P40 YAML."""
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected YAML mapping, got {type(cfg).__name__}")
    for required in ("benchmarks", "baselines", "pinn_profiles"):
        if required not in cfg:
            raise ValueError(f"{path}: missing required key {required!r}")
    return cfg


def build_pinn_config(profile: dict[str, Any]) -> PINNConfig:
    """Construct a PINNConfig from a YAML profile dict."""
    return PINNConfig(**profile)


def apply_overrides(
    pinn_profiles: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Apply CLI overrides to the YAML pinn_profiles, returning a fresh dict."""
    profiles = {name: dict(p) for name, p in pinn_profiles.items()}
    if args.device is not None and "p40" in profiles:
        profiles["p40"]["device"] = args.device
    if args.n_epochs is not None and "p40" in profiles:
        profiles["p40"]["n_epochs"] = args.n_epochs
    if args.n_collocation is not None and "p40" in profiles:
        profiles["p40"]["n_collocation"] = args.n_collocation
    return profiles


def apply_benchmark_overrides(
    benchmarks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Apply CLI overrides to benchmark refinement_levels."""
    if args.refinement_levels is None:
        return benchmarks
    levels = [int(x) for x in args.refinement_levels.split(",") if x.strip()]
    return [dict(b, refinement_levels=levels) for b in benchmarks]


def filter_baselines(
    baselines: list[dict[str, Any]],
    *,
    skip_cpu: bool,
) -> list[dict[str, Any]]:
    """Drop the CPU PINN row when --skip-cpu is set."""
    if not skip_cpu:
        return baselines
    return [b for b in baselines if b.get("name") != "pinn_cpu"]


def register_pinn_profiles(
    profiles: dict[str, dict[str, Any]],
    registry: dict[str, type[SimplePINNSolver]],
) -> None:
    """Build PINNConfig objects and bind them as named solvers in the registry."""
    for profile_name, profile_dict in profiles.items():
        config_obj = build_pinn_config(profile_dict)
        solver_name = f"pinn_{profile_name}"
        registry[solver_name] = _make_pinn_class(solver_name, config_obj)


def _make_pinn_class(solver_name: str, config: PINNConfig) -> type[SimplePINNSolver]:
    """Return a SimplePINNSolver subclass that defaults to ``config``.

    The bound class accepts ``**kwargs`` so callers like ``get_solver(name,
    **overrides)`` can pass per-call overrides without raising ``TypeError``.
    Per-call kwargs win over the YAML profile, matching the precedence the
    canonical solver already uses (see ``SimplePINNSolver.__init__``).
    """

    class _BoundPINN(SimplePINNSolver):
        name = solver_name

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("config", config)
            super().__init__(**kwargs)

    return _BoundPINN


def _gpu_index_for_profile(profiles: dict[str, dict[str, Any]]) -> int | None:
    """Extract the integer GPU index the GPU PINN profile will run on.

    Returns ``None`` for non-CUDA profiles (``cpu``, ``auto`` falls back),
    or for ``cuda`` (no explicit index — ``torch.cuda.current_device()``
    will pick at runtime). Used to drive the startup banner so the
    reported GPU matches what ``--device cuda:N`` actually selects.
    """
    p40 = profiles.get("p40")
    if p40 is None:
        return None
    device_label = str(p40.get("device", "auto"))
    if device_label.startswith("cuda:"):
        try:
            return int(device_label.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _print_cuda_banner(profiles: dict[str, dict[str, Any]]) -> None:
    """Print a banner describing the GPU the resolved profile will use."""
    if not torch.cuda.is_available():
        return
    gpu_idx = _gpu_index_for_profile(profiles)
    n = torch.cuda.device_count()
    if gpu_idx is None or gpu_idx >= n:
        # Fall back to current_device() so the banner matches what
        # ``.to(device)`` will actually select when device.index is None.
        gpu_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_idx)
    vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
    print(f"GPU : cuda:{gpu_idx}  {gpu_name}  ({vram_gb:.0f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}\n")


def main(argv: list[str] | None = None) -> int:
    """Run the P40 SBIR benchmark suite end-to-end and exit with status."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    args = parse_args(argv)

    if not torch.cuda.is_available():
        msg = "WARNING: No CUDA device found; the GPU PINN row will fail."
        print(msg, file=sys.stderr)
        if args.require_cuda:
            return 1

    cfg = load_config(Path(args.config))
    profiles = apply_overrides(cfg["pinn_profiles"], args)
    _print_cuda_banner(profiles)
    register_pinn_profiles(profiles, _baselines.SOLVER_REGISTRY)

    bench_cfg = dict(cfg)
    bench_cfg["benchmarks"] = apply_benchmark_overrides(cfg["benchmarks"], args)
    bench_cfg["baselines"] = filter_baselines(cfg["baselines"], skip_cpu=args.skip_cpu)
    if args.output_dir is not None:
        bench_cfg["output_dir"] = args.output_dir

    output_dir = Path(bench_cfg.get("output_dir", "outputs/sbir_p40"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(bench_cfg, tmp)
        tmp_path = tmp.name

    try:
        print("=" * 65)
        print("SBIR BENCHMARK SUITE — Tesla P40 High-Resolution Run")
        print("=" * 65)
        print(f"Benchmarks : {len(bench_cfg['benchmarks'])} problems")
        print(f"Solvers    : {[b['name'] for b in bench_cfg['baselines']]}")
        print(f"Output     : {output_dir.resolve()}")
        print("=" * 65 + "\n")

        t_total = time.perf_counter()
        runner = PDEBenchmarkRunner(tmp_path)
        results = runner.run_all()

        if not results:
            print("WARNING: No results produced — check solver registry and dependencies.")
            return 1

        runner.generate_report(results, output_dir)
        _print_summary(results)

        elapsed = time.perf_counter() - t_total
        print(f"\nTotal wall time : {elapsed:.1f}s")
        print(f"Results written : {output_dir}/results.json")
        print(f"                  {output_dir}/results.md")
        print(f"                  {output_dir}/results.csv")
        return 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _print_summary(results: list[Any]) -> None:
    """Print the per-row summary table."""
    print(f"\n{'=' * 65}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(
        f"{'Benchmark':<30} {'Solver':<16} {'DOF':>7} {'L2 Error':>12} {'Time(s)':>9} {'Conv':>6}"
    )
    print("-" * 85)
    for r in sorted(results, key=lambda x: (x.benchmark_name, x.method_name, x.n_dof)):
        l2 = f"{r.l2_error:.2e}" if r.l2_error is not None and not math.isnan(r.l2_error) else "N/A"
        # Use ``is not None`` (not truthy) so a legitimate ``0.0`` rate
        # (emitted when consecutive levels have equal L2 error) is shown
        # as ``0.00`` instead of being hidden as ``-``. Matches the
        # convention in src/research/pde_benchmarks.py:281,349.
        cr = f"{r.convergence_rate:.2f}" if r.convergence_rate is not None else "  -"
        print(
            f"{r.benchmark_name:<30} {r.method_name:<16} {r.n_dof:>7} {l2:>12} "
            f"{r.wall_time_seconds:>9.3f} {cr:>6}"
        )


if __name__ == "__main__":
    sys.exit(main())
