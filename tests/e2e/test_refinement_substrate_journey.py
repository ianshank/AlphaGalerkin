"""E2E journey: substrate registry -> refinement sweep -> adequacy verdict.

Guards (per ``docs/E2E_TEST_PLAN.md`` §4.3):

- ``specs/refinement_substrate.spec.md`` AC5 (one measurement, both substrates),
  AC7 (adaptive must beat uniform on the element-local substrate **and the same
  predicate must reject the tensor-product control**), and AC8's tripwire half
  (a rate that is too *good* is as diagnostic as one that is too bad).
- CLAUDE.md Regression Surface rows *"Element-local refinement substrate"* and
  *"Substrate contract defects (D1-D5)"* -- specifically D5: the substrate
  registry had no registrants, no export and no callers, and its guard must be
  read **out of process**.

Everything runs through :func:`py_runner` in a fresh interpreter, deliberately.
``RefinementSubstrateRegistry`` is a process-global singleton that
``tests/refinement/test_substrate.py`` and ``tests/research/test_skfem_substrate.py``
both ``clear()``, so an in-process assertion here would pass or fail on
collection order. A fresh interpreter sees only the import-time registration,
which is the property under test.

The journey resolves each substrate **through the registry** -- the lookup path
``src/refinement/substrate_registry.py`` promises so that "callers need not
import the concrete module directly" -- rather than through the class name.

**Surface: numpy-only.** ``src/research/substrates/*`` and
``src/research/lshape_amr_compare.py`` import no torch. No device is passed and
none is asserted (plan §1, flow (c)).
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from src.research.substrates.config import (
    RATE_FIT_MIN_POINTS,
    SUBSTRATE_KIND_SKFEM_TRI,
    SUBSTRATE_KIND_TENSOR_GRID,
)
from src.research.substrates.sweep import default_adequacy_gate
from tests.e2e.conftest import E2E_BENCHMARK_TIMEOUT_S, PyRunnerType

pytestmark = pytest.mark.e2e

#: Exit code of a child that completed its journey.
EXIT_SUCCESS: int = 0

#: Prefix of the single machine-readable line the child prints. ``structlog``
#: writes its own lines into the same stream, so the payload is located by this
#: marker rather than by position.
RESULT_MARKER: str = "E2E_SUBSTRATE_JSON "

#: Concrete class names the registry must resolve each kind to. Asserted so that
#: a registry which aliased both kinds to one class -- the mutation this file is
#: checked against -- cannot look like a passing gate.
EXPECTED_CLASS_NAMES: dict[str, str] = {
    SUBSTRATE_KIND_TENSOR_GRID: "TensorGridSubstrate",
    SUBSTRATE_KIND_SKFEM_TRI: "SkfemTriSubstrate",
}

#: ``describe()`` must publish the DOF convention it counts in; the two
#: substrates count different things, and a ratio between them would be
#: meaningless without it.
DOF_CONVENTION_KEY: str = "dof_convention"
SKFEM_DOF_CONVENTION: str = "fem_basis_dofs"

# --------------------------------------------------------------------------- #
# Budget for the tensor-grid completeness journey (test 1 only)                #
# --------------------------------------------------------------------------- #
#
# This journey asks "does the registry path produce a *complete* sweep", not
# "is the substrate adequate", so it may use a reduced budget. The adequacy
# journeys below use the PINNED window from ``AdequacyGateConfig`` instead --
# a reduced range there would change what the gate measures.
SMALL_SWEEP_MAX_DOF: int = 900
SMALL_SWEEP_MAX_LEVELS_ADAPTIVE: int = 12
SMALL_SWEEP_MAX_LEVELS_UNIFORM: int = 6
#: Wide enough that both arms contribute at least ``RATE_FIT_MIN_POINTS``
#: (measured: 8 adaptive, 4 uniform).
SMALL_SWEEP_DOF_RANGE: tuple[float, float] = (10.0, float(SMALL_SWEEP_MAX_DOF))


#: Imports, geometry and registry lookup shared by all three child programs.
#:
#: ``scale`` / ``initial_side`` / ``theta`` are read off ``ComparisonParams``
#: rather than retyped: they are the pinned benchmark parameters, and a rate
#: quoted without its theta is not a fact (the same substrate reads -1.31 at
#: theta=0.5 and -1.25 at theta=0.3).
_CHILD_PREAMBLE: str = '''\
import json

# Importing the concrete modules is what REGISTERS them. The journey then
# resolves each class through the registry, never through the import name.
import src.research.substrates.skfem_tri  # noqa: F401
import src.research.substrates.tensor_grid  # noqa: F401
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator
from src.refinement.substrate_registry import RefinementSubstrateRegistry
from src.research.lshape_amr_compare import ComparisonParams, lshape_inside_predicate
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.sweep import (
    default_adequacy_gate,
    gate_violations,
    measure_rate_separation,
    run_refinement_sweep,
)

MARKER = {marker!r}
TENSOR_GRID = {tensor_grid!r}
SKFEM_TRI = {skfem_tri!r}
PARAMS = ComparisonParams()


def build(kind):
    """Resolve *kind* through the registry and construct it."""
    cls = RefinementSubstrateRegistry().get_or_raise(kind)
    operator = LShapedPoissonOperator(
        PDEConfig(
            name="poisson_lshaped",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-PARAMS.scale, -PARAMS.scale],
            domain_max=[PARAMS.scale, PARAMS.scale],
        )
    )
    if kind == TENSOR_GRID:
        substrate = cls(
            operator,
            inside=lshape_inside_predicate(PARAMS.scale),
            config=SubstrateConfig(
                name="e2e_" + kind, kind=kind, initial_side=PARAMS.initial_side
            ),
        )
    else:
        substrate = cls(operator, config=SubstrateConfig(name="e2e_" + kind, kind=kind))
    return substrate, cls


def levels(points):
    return [
        {{
            "n_dof": p.n_dof,
            "n_dof_free": p.n_dof_free,
            "n_units": p.n_units,
            "l2_error": p.l2_error,
        }}
        for p in points
    ]


def payload(substrate, cls, separation, adaptive, uniform, dof_range):
    return {{
        "class_name": cls.__name__,
        "describe": dict(substrate.describe()),
        "adaptive_rate": separation.adaptive_rate,
        "uniform_rate": separation.uniform_rate,
        "error_ratio_at_matched_dof": separation.error_ratio_at_matched_dof,
        "matched_dof": separation.matched_dof,
        "n_adaptive_points": separation.n_adaptive_points,
        "n_uniform_points": separation.n_uniform_points,
        "adaptive_levels": levels(adaptive),
        "uniform_levels": levels(uniform),
        "dof_range": [float(dof_range[0]), float(dof_range[1])],
    }}


def emit(document):
    print(MARKER + json.dumps(document))
'''

#: Test 1: does the registry path yield a complete sweep on the control?
_TENSOR_GRID_SWEEP_BODY: str = """
substrate, cls = build(TENSOR_GRID)
adaptive = run_refinement_sweep(
    substrate,
    policy="adaptive",
    theta=PARAMS.marking_fraction,
    max_levels={max_levels_adaptive!r},
    max_dof={max_dof!r},
)
uniform = run_refinement_sweep(
    substrate,
    policy="uniform",
    theta=PARAMS.marking_fraction,
    max_levels={max_levels_uniform!r},
    max_dof={max_dof!r},
)
dof_range = ({dof_low!r}, {dof_high!r})
separation = measure_rate_separation(adaptive, uniform, dof_range)
emit(payload(substrate, cls, separation, adaptive, uniform, dof_range))
"""

#: Tests 2 and 3: the PINNED adequacy gate, applied identically to both kinds.
_ADEQUACY_BODY: str = """
gate = default_adequacy_gate()
substrate, cls = build({kind!r})
adaptive = run_refinement_sweep(
    substrate,
    policy="adaptive",
    theta=PARAMS.marking_fraction,
    max_levels=gate.max_levels_adaptive,
    max_dof=gate.max_sweep_dof,
)
uniform = run_refinement_sweep(
    substrate,
    policy="uniform",
    theta=PARAMS.marking_fraction,
    max_levels=gate.max_levels_uniform,
    max_dof=gate.max_sweep_dof,
)
separation = measure_rate_separation(adaptive, uniform, gate.rate_fit_dof_range)
document = payload(substrate, cls, separation, adaptive, uniform, gate.rate_fit_dof_range)
document["violations"] = gate_violations(separation, gate)
registry = RefinementSubstrateRegistry()
document["registered_class_names"] = {{
    kind: registry.get_or_raise(kind).__name__ for kind in (TENSOR_GRID, SKFEM_TRI)
}}
emit(document)
"""


def _child_source(body: str) -> str:
    """Assemble a complete child program from the shared preamble and *body*.

    Args:
        body: Journey-specific source, already formatted.

    Returns:
        Python source to run via ``python -c``.

    """
    preamble = _CHILD_PREAMBLE.format(
        marker=RESULT_MARKER,
        tensor_grid=SUBSTRATE_KIND_TENSOR_GRID,
        skfem_tri=SUBSTRATE_KIND_SKFEM_TRI,
    )
    return preamble + body


def _payload_from(stdout: str) -> dict[str, Any]:
    """Extract the child's single marked JSON line.

    Args:
        stdout: The child's stdout.

    Returns:
        The parsed payload.

    Raises:
        AssertionError: If no marked line was printed -- a child that produced
            no payload must fail loudly rather than yield an empty mapping that
            makes every downstream assertion vacuous.

    """
    for line in stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            parsed: dict[str, Any] = json.loads(line[len(RESULT_MARKER) :])
            return parsed
    raise AssertionError(f"no {RESULT_MARKER!r} line in child stdout:\n{stdout}")


def test_tensor_grid_from_registry_produces_a_complete_sweep(py_runner: PyRunnerType) -> None:
    """The registry lookup path yields a usable, fully-populated measurement.

    Guards ``specs/refinement_substrate.spec.md`` AC5 and D5's registry
    contract: ``RefinementSubstrateRegistry().get(...)`` must return the concrete
    substrate, which must then drive both marking policies to a rate separation
    with enough points on each arm to mean anything.

    **No sign assertion.** Whether adaptive beats uniform here is a research
    outcome at a reduced budget; the falsifiable verdict is the pinned adequacy
    gate, asserted in the two tests below.
    """
    body = _TENSOR_GRID_SWEEP_BODY.format(
        max_levels_adaptive=SMALL_SWEEP_MAX_LEVELS_ADAPTIVE,
        max_levels_uniform=SMALL_SWEEP_MAX_LEVELS_UNIFORM,
        max_dof=SMALL_SWEEP_MAX_DOF,
        dof_low=SMALL_SWEEP_DOF_RANGE[0],
        dof_high=SMALL_SWEEP_DOF_RANGE[1],
    )
    result = py_runner(_child_source(body), E2E_BENCHMARK_TIMEOUT_S, None)
    assert result.returncode == EXIT_SUCCESS, result.output

    payload = _payload_from(result.stdout)
    assert payload["class_name"] == EXPECTED_CLASS_NAMES[SUBSTRATE_KIND_TENSOR_GRID]

    assert payload["adaptive_levels"], "the adaptive arm produced no levels"
    assert payload["uniform_levels"], "the uniform arm produced no levels"
    assert payload["n_adaptive_points"] >= RATE_FIT_MIN_POINTS
    assert payload["n_uniform_points"] >= RATE_FIT_MIN_POINTS

    for key in ("adaptive_rate", "uniform_rate", "error_ratio_at_matched_dof", "matched_dof"):
        assert math.isfinite(payload[key]), f"{key} = {payload[key]!r}"

    describe = payload["describe"]
    assert describe[DOF_CONVENTION_KEY], (
        "describe() must publish the DOF convention it counts in; without it a "
        "cross-substrate ratio has no defined meaning"
    )


@pytest.mark.fem_required
def test_skfem_from_registry_passes_the_pinned_gate(py_runner: PyRunnerType) -> None:
    """AC7, positive half, driven from the registry over the **pinned** window.

    The DOF window is ``AdequacyGateConfig.rate_fit_dof_range`` as shipped, not
    a reduced one: below that window neither arm has separated, so a shortened
    range would very likely miss ``adaptive_rate_min`` and turn a real gate into
    a flaky one. The parent asserts the child actually used the pinned pair.

    Also pins ``n_dof_free <= n_dof`` at every level -- free DOF exclude the
    Dirichlet boundary, so the reverse would mean the boundary condition was
    never imposed, which is precisely the silent defect behind the 2026-08-16
    L-shape retraction.
    """
    body = _ADEQUACY_BODY.format(kind=SUBSTRATE_KIND_SKFEM_TRI)
    result = py_runner(_child_source(body), E2E_BENCHMARK_TIMEOUT_S, None)
    assert result.returncode == EXIT_SUCCESS, result.output

    payload = _payload_from(result.stdout)
    assert payload["class_name"] == EXPECTED_CLASS_NAMES[SUBSTRATE_KIND_SKFEM_TRI]
    assert payload["dof_range"] == list(default_adequacy_gate().rate_fit_dof_range), (
        "the adequacy verdict must be measured over the pinned window"
    )

    assert payload["violations"] == [], (
        f"the element-local substrate must satisfy the adequacy gate; got {payload}"
    )
    assert payload["describe"][DOF_CONVENTION_KEY] == SKFEM_DOF_CONVENTION

    for arm in ("adaptive_levels", "uniform_levels"):
        for level in payload[arm]:
            assert level["n_dof_free"] <= level["n_dof"], f"{arm}: {level}"


def test_gate_is_not_vacuous(py_runner: PyRunnerType) -> None:
    """AC7, discriminating half: the same predicate must **reject** the control.

    "A gate that passes on both substrates is not a gate." Driven from the
    registry path so the promoted ``gate_violations`` is exercised exactly as a
    library caller would reach it.

    Carries no ``fem_required`` marker on purpose: the discriminating half must
    run on every CPU job, since a gate that silently stopped discriminating
    would otherwise be hidden on precisely the runs where substrate work is most
    likely to regress.

    Also asserts the registry does not **alias** the two kinds to one class. If
    it did, "the same measurement fails on the other substrate" would be a
    statement about the same substrate twice -- and the positive half above,
    being ``fem_required``, would skip without complaint on a CPU runner.
    """
    body = _ADEQUACY_BODY.format(kind=SUBSTRATE_KIND_TENSOR_GRID)
    result = py_runner(_child_source(body), E2E_BENCHMARK_TIMEOUT_S, None)
    assert result.returncode == EXIT_SUCCESS, result.output

    payload = _payload_from(result.stdout)
    assert payload["violations"], (
        "the adequacy gate must REJECT the legacy tensor-product substrate -- "
        f"a gate that passes on both substrates measures nothing; got {payload}"
    )

    registered = payload["registered_class_names"]
    assert registered == EXPECTED_CLASS_NAMES, (
        "each substrate kind must resolve to its own class; an aliased registry "
        f"makes the two halves of AC7 the same measurement. Got {registered}"
    )
