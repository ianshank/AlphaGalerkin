"""E2E journeys for the two shipped scripts that no test exercised at all.

Guards ``docs/E2E_TEST_PLAN.md`` §6.3. Neither script appears in any CLAUDE.md
Regression Surface row, in ``tests/scripts/``, or in any workflow step — this
file is their first coverage, which is why each gets a **real run** rather than
a ``--help`` smoke. ``--help`` proves an ``argparse`` parser was constructed and
nothing else; it would have passed on both scripts throughout their entire
lifetime, including in a state where the run path raised on import.

- ``scripts/demo_pde_solver.py`` drives ``PDETrainer`` (the surface CLAUDE.md's
  *"PDE end-to-end"* row covers as a library) through its shipped CLI.
- ``scripts/export_helix_stl.py`` writes the geometry that the *"Noyron HX
  scenario (SDF, domain, scenario)"* row's ``AnalyticalHelixSDF`` represents.

**Device (plan §1, flow (c)):** both scripts contain zero ``torch`` references
and take no device flag; per the plan's entry-point table they are numpy/scipy
surfaces. No device is forwarded and no device assertion is fabricated here.

Every artifact lands under ``tmp_path`` via ``--output-dir`` / ``--output``, and
each test asserts that nothing appeared outside it.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.conftest import E2E_BENCHMARK_TIMEOUT_S

if TYPE_CHECKING:
    from tests.e2e.conftest import CLIRunnerType

pytestmark = pytest.mark.e2e

#: Exit code both scripts return on success.
EXIT_OK = 0

# --------------------------------------------------------------------------- #
# scripts/demo_pde_solver.py                                                   #
# --------------------------------------------------------------------------- #

DEMO_PDE_MODULE = "scripts.demo_pde_solver"

#: Argv knobs for the demo run. Every one is passed explicitly so the run's
#: cost is set by this file, not by the script's (much larger) defaults.
DEMO_PDE_TYPE = "poisson"
DEMO_N_EPISODES = 1
DEMO_MCTS_SIMS = 2
DEMO_SEED = 1
#: A second seed, used to prove --seed reaches the solver rather than only the
#: report. Any value distinct from DEMO_SEED works; the tests assert a
#: difference between the two runs, never a value from either.
DEMO_ALTERNATE_SEED = 20260904

#: Files ``run_demo`` writes into ``--output-dir``. The plot is written *only*
#: without ``--no-plots``, which is what makes its absence an assertion about
#: the flag rather than an incidental observation.
DEMO_METRICS_FILENAME = "pde_results.json"
DEMO_PLOT_FILENAME = "convergence.png"

#: Keys of the ``DemoMetrics`` dataclass the JSON must round-trip. Named so a
#: renamed field fails here instead of silently shrinking the assertion.
DEMO_ECHOED_KEYS = ("pde_type", "n_episodes", "mcts_simulations", "seed")
DEMO_ERROR_KEYS = ("initial_error", "final_error")

# --------------------------------------------------------------------------- #
# scripts/export_helix_stl.py                                                  #
# --------------------------------------------------------------------------- #

HELIX_STL_MODULE = "scripts.export_helix_stl"

#: One turn keeps the mesh small; the geometry's correctness is
#: ``tests/pde/test_sdf.py``'s job, not this file's.
HELIX_N_TURNS = 1

HELIX_STL_FILENAME = "h.stl"

#: Binary STL layout (the only layout this script emits -- there is no ASCII
#: ``solid`` branch in ``_write_binary_stl``, so this is an exact structure, not
#: one arm of an either/or):
#:   [0, 80)      free-form header
#:   [80, 84)     little-endian uint32 triangle count
#:   then         ``count`` fixed-size records of
#:                float32[3] normal + 3 * float32[3] vertex + uint16 attribute
STL_HEADER_BYTES = 80
STL_COUNT_FIELD_BYTES = 4
STL_COUNT_STRUCT = "<I"
STL_TRIANGLE_RECORD_BYTES = 50
STL_PREAMBLE_BYTES = STL_HEADER_BYTES + STL_COUNT_FIELD_BYTES


def _files_under(root: Path) -> set[Path]:
    """Every regular file beneath *root*.

    Args:
        root: Directory to walk.

    Returns:
        Absolute paths of all regular files found.

    """
    return {path for path in root.rglob("*") if path.is_file()}


def _run_pde_demo(
    cli_runner: CLIRunnerType, output_dir: Path, seed: int = DEMO_SEED
) -> dict[str, Any]:
    """Run the PDE demo into *output_dir* and return its parsed metrics JSON.

    Shared by the demo journeys so each asserts a different property of the same
    real run without either duplicating the argv.

    Args:
        cli_runner: The subprocess runner fixture.
        output_dir: ``--output-dir`` for the run; must be under ``tmp_path``.
        seed: ``--seed`` for the run. A parameter rather than a constant because
            the only way to prove the flag reaches the *solver* -- as opposed to
            being copied into the report from a second read of the same config
            field -- is to vary it and observe the numbers move.

    Returns:
        The decoded ``metrics.json`` payload.

    """
    result = cli_runner(
        DEMO_PDE_MODULE,
        [
            "--pde-type",
            DEMO_PDE_TYPE,
            "--n-episodes",
            str(DEMO_N_EPISODES),
            "--mcts-sims",
            str(DEMO_MCTS_SIMS),
            "--output-dir",
            str(output_dir),
            "--no-plots",
            "--seed",
            str(seed),
        ],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    assert result.returncode == EXIT_OK, result.output
    payload: dict[str, Any] = json.loads(
        (output_dir / DEMO_METRICS_FILENAME).read_text(encoding="utf-8")
    )
    return payload


def test_demo_pde_solver_runs_an_episode_and_writes_only_where_told(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    """The demo completes a real episode and writes exactly one artifact.

    First coverage of ``scripts/demo_pde_solver.py`` (plan §6.3): it drives the
    same ``PDETrainer`` the CLAUDE.md *"PDE end-to-end"* row covers as a
    library, but through the shipped CLI and as a process, so an ``argparse``
    default drifting away from ``PDETrainingConfig``'s schema fails here.

    Two properties, both of which a ``--help`` smoke would miss: the run
    reaches completion (``sys.exit(0)`` after ``run_demo``), and ``--no-plots``
    actually suppresses the plot — asserted as an exact file set, so a stray
    artifact escaping ``--output-dir`` fails too.

    Numpy-only surface (plan §1, flow (c)); no device is forwarded.
    """
    output_dir = tmp_path / "pde_demo"

    _run_pde_demo(cli_runner, output_dir)

    assert _files_under(tmp_path) == {output_dir / DEMO_METRICS_FILENAME}
    assert not (output_dir / DEMO_PLOT_FILENAME).exists(), "--no-plots must suppress the plot"


def test_demo_pde_solver_metrics_json_echoes_argv_and_reports_finite_errors(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    """The demo's JSON round-trips its argv and carries a finite error history.

    First coverage of ``scripts/demo_pde_solver.py``'s metrics contract (plan
    §6.3). The echo assertion is the load-bearing half: a script that ignored
    ``--n-episodes`` or ``--seed`` and ran its defaults would still exit 0 and
    still write this file, so the previous test alone cannot tell the two
    apart.

    No value of any error is asserted, and ``total_time_seconds`` is
    deliberately read but never compared — determinism is not assumed and
    wall-clock is never an assertion (plan §2 rules 5 and 9). What is asserted
    is structure: one final error per requested episode, all finite.

    Numpy-only surface (plan §1, flow (c)); no device is forwarded.
    """
    output_dir = tmp_path / "pde_demo"

    payload = _run_pde_demo(cli_runner, output_dir)
    requested = {
        "pde_type": DEMO_PDE_TYPE,
        "n_episodes": DEMO_N_EPISODES,
        "mcts_simulations": DEMO_MCTS_SIMS,
        "seed": DEMO_SEED,
    }
    assert {key: payload[key] for key in DEMO_ECHOED_KEYS} == requested

    per_episode = payload["per_episode_errors"]
    assert len(per_episode) == DEMO_N_EPISODES
    assert all(math.isfinite(error) for error in per_episode)
    assert all(math.isfinite(payload[key]) for key in DEMO_ERROR_KEYS)
    assert len(payload["episode_summaries"]) == DEMO_N_EPISODES
    assert "total_time_seconds" in payload  # present; never asserted on (rule 5)


def test_demo_pde_solver_seed_reaches_the_solver_not_just_the_report(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    """``--seed`` changes the numbers, and the same seed reproduces them.

    The sibling test above asserts the JSON *echoes* the requested seed. That
    is not the same property, and the difference is not academic:
    ``scripts/demo_pde_solver.py`` reads ``cfg.seed`` twice -- once into
    ``DemoMetrics.seed`` (the echo) and once into ``PDETrainingConfig(seed=...)``
    (the solver). Severing the second read leaves the first intact.

    Verified: hardcoding ``seed=42`` in the trainer construction left the whole
    file green. This test fails on that mutation, because two different
    requested seeds then produce identical error histories.

    Asserts a *difference*, never a value -- no number here is a threshold.
    Numpy-only surface; no device is forwarded.
    """
    first = _run_pde_demo(cli_runner, tmp_path / "seed_a", seed=DEMO_SEED)
    second = _run_pde_demo(cli_runner, tmp_path / "seed_b", seed=DEMO_ALTERNATE_SEED)
    repeat = _run_pde_demo(cli_runner, tmp_path / "seed_a_again", seed=DEMO_SEED)

    assert first["per_episode_errors"] != second["per_episode_errors"], (
        f"seeds {DEMO_SEED} and {DEMO_ALTERNATE_SEED} produced identical error "
        "histories: --seed is echoed into the report but never reaches the solver"
    )
    assert first["per_episode_errors"] == repeat["per_episode_errors"], (
        "the same seed did not reproduce: the run is not seed-deterministic, so "
        "the difference above cannot be attributed to the seed"
    )


def _run_helix_export(cli_runner: CLIRunnerType, output: Path) -> None:
    """Export the helix STL to *output*, asserting only that it exited 0.

    Args:
        cli_runner: The subprocess runner fixture.
        output: ``--output`` path for the run; must be under ``tmp_path``.

    """
    result = cli_runner(
        HELIX_STL_MODULE,
        ["--n-turns", str(HELIX_N_TURNS), "--output", str(output)],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    assert result.returncode == EXIT_OK, result.output


def test_export_helix_stl_writes_only_the_requested_file(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    """The exporter honours ``--output`` and creates nothing else.

    First coverage of ``scripts/export_helix_stl.py`` (plan §6.3). The script's
    ``--output`` default points at ``outputs/poc/noyron_hx/noyron_hx.stl``
    *inside the repo*, so an ignored flag would write into the working tree;
    the exact-file-set assertion is what catches that, and it is why the run is
    real rather than a ``--help`` smoke.

    Numpy-only surface (plan §1, flow (c)); no device is forwarded.
    """
    output = tmp_path / HELIX_STL_FILENAME

    _run_helix_export(cli_runner, output)

    assert _files_under(tmp_path) == {output}


def test_export_helix_stl_writes_a_nonempty_wellformed_binary_stl(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    """The exported file is a binary STL whose declared count matches its size.

    Guards the geometry artifact behind the CLAUDE.md *"Noyron HX scenario
    (SDF, domain, scenario)"* row: ``AnalyticalHelixSDF`` is what the scenario
    trains against, and this script is the only thing that turns it into a mesh
    a reviewer can open. A truncated or mis-counted write produces a file that
    still exists, is still non-empty, and still exits 0.

    ``triangle_count > 0`` is asserted **separately from** the size/count
    consistency check, and is the assertion that carries the weight: an
    exporter emitting *zero* triangles writes exactly the 84-byte preamble, for
    which the consistency relation holds trivially (``0 == (84 - 84) // 50``).
    Consistency alone would pass on an empty mesh.

    Numpy-only surface (plan §1, flow (c)); no device is forwarded.
    """
    output = tmp_path / HELIX_STL_FILENAME

    _run_helix_export(cli_runner, output)

    data = output.read_bytes()
    assert len(data) >= STL_PREAMBLE_BYTES, "file is shorter than a binary STL preamble"

    (triangle_count,) = struct.unpack(STL_COUNT_STRUCT, data[STL_HEADER_BYTES:STL_PREAMBLE_BYTES])
    assert triangle_count > 0, "an STL declaring no triangles is not a mesh"
    assert triangle_count == (len(data) - STL_PREAMBLE_BYTES) // STL_TRIANGLE_RECORD_BYTES
    assert len(data) == STL_PREAMBLE_BYTES + triangle_count * STL_TRIANGLE_RECORD_BYTES, (
        "trailing or missing bytes: the declared triangle count does not tile the payload"
    )
