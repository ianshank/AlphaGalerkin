"""Correctness gates for ``MultiCornerPoissonOperator`` (the polyomino AMR testbed).

The chief risk is a silently wrong substrate: on 2026-08-16 an L-shape result was
retracted because a boundary-condition defect made the substrate diverge while
every number stayed finite and plausible. These tests make the analogous defects
in this operator impossible to miss, each against an independent reference:

* **Generalisation.** The L preset equals ``LShapedPoissonOperator`` to <= 1e-12
  (float64) at random interior *and* boundary points.
* **Harmonicity.** A five-point Laplacian is below a *derived* truncation+rounding
  bound (Hypothesis over points and strengths), autograd's Laplacian is ~0, and
  the Laplacian stays bounded *across* the internal cell interfaces -- where a
  misplaced branch cut would put a kink or a jump.
* **Boundary.** Each singular term vanishes on its own corner's two edges,
  located from the geometry (not from the formula), including ``-0.0`` and
  ``+/-1e-300`` perturbations either side of the edge.
* **Validator.** Presets accepted; a cut through the interior, a grazing cut, a
  bisector into the domain, an off-bisector, an undeclared corner, a wrong angle,
  a degenerate or mis-boxed config -- each refused with a named exception.

Nothing here needs scikit-fem, so every guard runs on every CPU lane.

Mutation kills (harden-a-guard; planted defect -> named test that went red):

* BranchCutError raise disabled -> ``TestBranchCutValidator::
  test_cut_ray_through_the_interior_is_rejected`` and
  ``::test_cut_ray_grazing_the_closure_is_rejected``.
* Bisector check disabled -> ``::test_bisector_pointing_into_the_domain_is_rejected``,
  ``::test_bisector_off_the_exterior_bisector_is_rejected``.
* Wrap moved onto the edge (``phi = mod(theta - b - offset, 2pi)``), numpy and
  torch separately -> ``TestBoundaryBehaviour::test_singular_term_vanishes_on_its_own_edges``
  rows of the matching backend (L and both Z corners). The first draft of this
  test **survived** that mutation: its only off-edge rows were ``+/-1e-300``,
  and ``theta - b`` absorbs a 1e-300 angle before the mod ever sees it. The
  EDGE_PROBE_OFFSET rows exist because of that survivor.
* L-style ``[0, 2pi)`` wrap shifted by the bisector (both backends) -> every row
  of ``::test_singular_term_vanishes_on_its_own_edges``,
  ``::test_the_lshape_operators_own_wrap_fails_where_this_one_holds``, the B|C
  rows of ``TestHarmonicity::test_laplacian_is_bounded_across_internal_interfaces``
  and ``::test_five_point_laplacian_is_within_the_derived_bound`` -- while
  ``TestLPresetReproducesTheLShapeOperator`` stays **green**: on the L this
  mutation *is* the L operator, so the equivalence alone cannot see it.
* Exponent inverted (``w / pi``) -> ``::test_singular_term_vanishes_on_its_own_edges``
  and ``TestLPresetReproducesTheLShapeOperator``.
* Angular exponent dropped (``r**lam * sin(phi)``, torch) ->
  ``TestHarmonicity::test_five_point_laplacian_is_within_the_derived_bound``,
  ``::test_autograd_residual_vanishes_on_the_exact_solution`` and every
  interface row.
* All-zero guard removed -> ``TestConstructionGuards::test_all_zero_coefficients_are_rejected``.
* Double-``where`` removed -> ``TestCornerGuard::test_gradient_at_the_corner_is_finite``.
* Box check removed -> ``TestConstructionGuards::test_config_box_must_be_the_domain_box``.
* Duplicate check keyed on the *declared* position (the pre-fix check) ->
  ``::test_a_near_duplicate_corner_is_rejected`` and
  ``::test_every_declaration_matching_a_declared_corner_is_rejected``. Keyed on the
  declared position rounded to 12 places -> only the Hypothesis sweep; the
  reviewer's single case stays green, which is why the sweep exists.
* Box ``rtol`` back to ``np.allclose``'s default (the pre-fix call), dropped on
  ``domain_max`` only, or pinned to ``1e-9`` ->
  ``::test_the_box_tolerance_is_the_stated_absolute_one`` ``twice_the_atol`` rows.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry_polyomino import (
    LSHAPE_POLYOMINO_CELLS,
    ZSHAPE_POLYOMINO_CELLS,
    PolyominoCell,
    PolyominoDomain,
)
from src.pde.operators import LShapedPoissonOperator, MultiCornerPoissonOperator
from src.pde.operators import multi_corner_poisson as mcp
from src.pde.operators.multi_corner_poisson import (
    DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
    DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
    LSHAPE_EXTERIOR_BISECTOR,
    BranchCutError,
    CornerDeclarationError,
    SingularCornerTerm,
    build_lshape_multi_corner_operator,
    build_zshape_poisson_operator,
    singular_term_tensor,
    singular_term_values,
)
from src.pde.registry import get_pde_operator, list_pde_operators

C = PolyominoCell

#: float64 generalisation tolerance: both operators evaluate the same analytic
#: function; they differ only in where each rounds (measured ~1e-15).
L_EQUIVALENCE_ATOL = 1e-12

#: Numpy paths return float32; values are <= 2**(2/3) in magnitude, so a few
#: float32 ulps is ~1e-6.
FLOAT32_ATOL = 1e-6

#: ``s_i`` on its own edges is ``r**lam * sin(k*pi)`` in float64 (sin(pi) =
#: 1.2e-16) plus O(1e-16)-radian atan2/mod rounding, with r <= 1 on these edges.
EDGE_VANISHING_ATOL = 1e-14

#: Offsets (either side of an edge) that probe the *exact* edge: ``-0.0`` and a
#: ``1e-300`` nudge are absorbed by ``theta - b`` and must read as on-edge.
EDGE_EXACT_OFFSETS = (0.0, -0.0, 1e-300, -1e-300)

#: An offset that is *not* absorbed by ``theta - b`` (ulp(pi/4)/2 ~ 5.6e-17
#: radians is), so it reaches the wrap. A wrap placed on an edge maps the
#: outside of that edge to ``phi ~ 2*pi`` and reads ``|s| ~ 0.87 r**lam``.
EDGE_PROBE_OFFSET = 1e-9

#: Allowed ``|s_i|`` at EDGE_PROBE_OFFSET from an edge: ``|grad s_i| = lam *
#: r**(lam - 1)``, which is <= 1.81 for r >= 0.05 (the closest probe), so 2x the
#: offset bounds it -- eight orders of magnitude below the wrapped value.
EDGE_PROBE_ATOL = 2.0 * EDGE_PROBE_OFFSET

#: Five-point stencil step: small enough that the O(h^2) truncation stays below
#: ~1e-4 at MIN_CORNER_DISTANCE, large enough that float64 rounding (amplified by
#: 1/h^2 = 1e6) stays near 1e-9.
FD_STEP = 1e-3

#: Stencil centres stay this far from every corner: the truncation constant
#: grows like r**(lam - 4).
MIN_CORNER_DISTANCE = 0.2

#: Rounding allowance per stencil, in ulps of the largest |u| it touches.
ROUNDING_ULPS = 64

#: Autograd Laplacian tolerance: each second derivative is at most
#: ``|lam (lam - 1)| r**(lam - 2) * |c| <~ 4`` here, so float64 cancellation
#: leaves ~1e-14; a thousandfold margin.
AUTOGRAD_LAPLACIAN_ATOL = 1e-10

#: Strength range the Hypothesis sweeps draw from.
COEFFICIENT_BOUND = 2.0

TALL_U_CELLS = (
    C(0.0, 1.0, 0.0, 1.0),
    C(1.0, 2.0, 0.0, 1.0),
    C(2.0, 3.0, 0.0, 1.0),
    C(0.0, 1.0, 1.0, 2.0),
    C(0.0, 1.0, 2.0, 3.0),
    C(2.0, 3.0, 1.0, 2.0),
    C(2.0, 3.0, 2.0, 3.0),
)
SHORT_U_CELLS = TALL_U_CELLS[:4] + (TALL_U_CELLS[5],)


def _operator(
    cells: tuple[PolyominoCell, ...], corners: tuple[SingularCornerTerm, ...]
) -> MultiCornerPoissonOperator:
    domain = PolyominoDomain(cells)
    low, high = domain.bounding_box()
    config = PDEConfig(
        name="multi_corner_test",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=list(low),
        domain_max=list(high),
    )
    return MultiCornerPoissonOperator(config, domain=domain, corners=corners)


def _lshape_operator() -> LShapedPoissonOperator:
    return LShapedPoissonOperator(
        PDEConfig(
            name="lshape_reference",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-1.0, -1.0],
            domain_max=[1.0, 1.0],
        )
    )


def _fourth_derivative_constant(lam: float) -> float:
    """``|d^4/dz^4 z**lam| = |lam (lam-1) (lam-2) (lam-3)| r**(lam-4)`` bounds u_xxxx, u_yyyy."""
    return abs(lam * (lam - 1.0) * (lam - 2.0) * (lam - 3.0))


def _five_point_laplacian(operator: MultiCornerPoissonOperator, x: float, y: float) -> float:
    h = FD_STEP
    stencil = torch.tensor(
        [[x, y], [x + h, y], [x - h, y], [x, y + h], [x, y - h]], dtype=torch.float64
    )
    u = operator.exact_solution(stencil)
    assert isinstance(u, torch.Tensor)
    return float((u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / h**2)


def _laplacian_bound(operator: MultiCornerPoissonOperator, x: float, y: float) -> float:
    """Derived bound on the five-point Laplacian of a harmonic ``u``.

    Truncation: ``|L_h u - L u| <= (h^2/12)(max|u_xxxx| + max|u_yyyy|)``, and each
    ``s_i`` is the imaginary part of a rotated ``(z - p_i)**lam``, so both fourth
    derivatives are bounded by ``C4(lam) r**(lam-4)`` with r the stencil's
    closest approach to the corner. Rounding: ROUNDING_ULPS ulps of the
    largest ``|u|`` on the stencil, divided by ``h^2``.
    """
    truncation = 0.0
    magnitude = 0.0
    for term in operator.corners:
        r = math.hypot(x - term.x, y - term.y)
        lam = term.exponent
        truncation += (
            abs(term.coefficient)
            * FD_STEP**2
            / 6.0
            * _fourth_derivative_constant(lam)
            * (r - FD_STEP) ** (lam - 4.0)
        )
        magnitude += abs(term.coefficient) * (r + FD_STEP) ** lam
    rounding = ROUNDING_ULPS * np.finfo(np.float64).eps * magnitude / FD_STEP**2
    return truncation + rounding


def _stencil_is_admissible(operator: MultiCornerPoissonOperator, x: float, y: float) -> bool:
    far_from_corners = all(
        math.hypot(x - t.x, y - t.y) >= MIN_CORNER_DISTANCE for t in operator.corners
    )
    h = FD_STEP
    stencil = torch.tensor(
        [[x, y], [x + h, y], [x - h, y], [x, y + h], [x, y - h]], dtype=torch.float64
    )
    return far_from_corners and bool(operator.geometry.contains_point(stencil).all())


def _edge_points(domain: PolyominoDomain, corner_index: int) -> list[tuple[np.ndarray, float]]:
    """``(points, atol)`` on the two boundary segments at a corner, both sides of each edge.

    Located from the *geometry*: an edge test that derived its points from the
    bisector would share any defect in the formula it is checking. Exact-edge
    offsets carry EDGE_VANISHING_ATOL; the EDGE_PROBE_OFFSET rows carry the
    gradient-derived EDGE_PROBE_ATOL.
    """
    corner = domain.reentrant_corners[corner_index].position
    t = np.linspace(0.05, 1.0, 20)
    offsets = [(offset, EDGE_VANISHING_ATOL) for offset in EDGE_EXACT_OFFSETS] + [
        (EDGE_PROBE_OFFSET, EDGE_PROBE_ATOL),
        (-EDGE_PROBE_OFFSET, EDGE_PROBE_ATOL),
    ]
    blocks: list[tuple[np.ndarray, float]] = []
    for segment in domain.boundary_segments:
        if corner not in (segment.start, segment.end):
            continue
        far = segment.end if segment.start == corner else segment.start
        base = np.array(corner) + t[:, None] * (np.array(far) - np.array(corner))
        normal_axis = 1 if segment.start[1] == segment.end[1] else 0
        on_line = corner[normal_axis]
        for offset, atol in offsets:
            shifted = base.copy()
            # ``0.0 + (-0.0)`` is ``+0.0``: keep a signed zero when the edge's line is zero.
            shifted[:, normal_axis] = offset if on_line == 0.0 else on_line + offset
            blocks.append((shifted, atol))
    assert len(blocks) == 2 * len(offsets), "a reentrant corner has exactly two boundary segments"
    return blocks


class TestSingularCornerTerm:
    def test_properties(self) -> None:
        term = SingularCornerTerm(x=1.0, y=2.0, exterior_bisector=0.5)
        assert term.position == (1.0, 2.0)
        assert term.exponent == pytest.approx(2.0 / 3.0)
        assert term.wedge_offset == pytest.approx(math.pi / 4)
        assert term.coefficient == 1.0

    @pytest.mark.parametrize("angle", [math.pi, 2.0 * math.pi, 0.5])
    def test_interior_angle_must_be_reentrant(self, angle: float) -> None:
        with pytest.raises(ValueError, match="interior_angle"):
            SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=0.0, interior_angle=angle)

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_values_are_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError):
            SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=0.0, coefficient=bad)

    def test_frozen_and_extra_forbidden(self) -> None:
        term = SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=0.0)
        with pytest.raises(ValueError):
            term.coefficient = 3.0  # type: ignore[misc]
        with pytest.raises(ValueError):
            SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=0.0, strength=1.0)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def lshape_points() -> torch.Tensor:
    """Random interior and boundary points of the closed L, plus exact edge points."""
    domain = PolyominoDomain(LSHAPE_POLYOMINO_CELLS)
    generator = torch.Generator().manual_seed(2026)
    interior = domain.sample_interior(3000, generator=generator)
    boundary = domain.sample_boundary(3000, generator=generator)
    exact_edges = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, -0.5], [1.0, 0.0], [0.0, -1.0]])
    return torch.cat([interior, boundary, exact_edges]).double()


class TestLPresetReproducesTheLShapeOperator:
    """The generalisation proof: one corner, unit strength, is the L benchmark."""

    @pytest.fixture
    def points(self, lshape_points: torch.Tensor) -> torch.Tensor:
        return lshape_points

    def test_torch_float64_agrees_to_1e12(self, points: torch.Tensor) -> None:
        mine = build_lshape_multi_corner_operator().exact_solution(points)
        reference = _lshape_operator().exact_solution(points)
        assert isinstance(mine, torch.Tensor) and isinstance(reference, torch.Tensor)
        assert mine.dtype == torch.float64
        assert float((mine - reference).abs().max()) <= L_EQUIVALENCE_ATOL

    def test_numpy_path_agrees_to_float32_precision(self, points: torch.Tensor) -> None:
        pts = points.numpy()
        mine = build_lshape_multi_corner_operator().exact_solution(pts)
        reference = _lshape_operator().exact_solution(pts)
        assert isinstance(mine, np.ndarray) and mine.dtype == np.float32
        np.testing.assert_allclose(mine, reference, rtol=0.0, atol=FLOAT32_ATOL)

    @settings(max_examples=60, deadline=None)
    @given(
        cell_index=st.integers(min_value=0, max_value=len(LSHAPE_POLYOMINO_CELLS) - 1),
        u=st.floats(min_value=0.0, max_value=1.0),
        v=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_agreement_holds_everywhere_on_the_closed_lshape(
        self, cell_index: int, u: float, v: float
    ) -> None:
        cell = LSHAPE_POLYOMINO_CELLS[cell_index]
        point = torch.tensor(
            [[cell.x0 + u * (cell.x1 - cell.x0), cell.y0 + v * (cell.y1 - cell.y0)]],
            dtype=torch.float64,
        )
        mine = build_lshape_multi_corner_operator().exact_solution(point)
        reference = _lshape_operator().exact_solution(point)
        assert isinstance(mine, torch.Tensor) and isinstance(reference, torch.Tensor)
        assert abs(float(mine[0] - reference[0])) <= L_EQUIVALENCE_ATOL


PRESET_CORNERS = [
    pytest.param(LSHAPE_POLYOMINO_CELLS, build_lshape_multi_corner_operator, 0, id="L"),
    pytest.param(ZSHAPE_POLYOMINO_CELLS, build_zshape_poisson_operator, 0, id="Z-primary"),
    pytest.param(ZSHAPE_POLYOMINO_CELLS, build_zshape_poisson_operator, 1, id="Z-secondary"),
]


class TestBoundaryBehaviour:
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize(("cells", "build", "corner_index"), PRESET_CORNERS)
    def test_singular_term_vanishes_on_its_own_edges(
        self, cells: tuple[PolyominoCell, ...], build: object, corner_index: int, backend: str
    ) -> None:
        operator = build()  # type: ignore[operator]
        term = operator.corners[corner_index]
        for block, atol in _edge_points(PolyominoDomain(cells), corner_index):
            if backend == "numpy":
                values = singular_term_values(block, term)
            else:
                values = singular_term_tensor(
                    torch.tensor(block, dtype=torch.float64), term
                ).numpy()
            assert float(np.abs(values).max()) <= atol, (backend, block[:2].tolist())

    def test_the_edge_probe_rows_include_a_negative_zero(self) -> None:
        """Guards the guard: ``0.0 + (-0.0)`` is ``+0.0``, which once hid this row."""
        blocks = _edge_points(PolyominoDomain(LSHAPE_POLYOMINO_CELLS), 0)
        assert any(
            np.signbit(block[0][:, 1]).all() and not block[0][:, 1].any() for block in blocks
        )

    def test_the_lshape_operators_own_wrap_fails_where_this_one_holds(self) -> None:
        """Characterisation: why the wrap sits on the bisector, not on an edge.

        A node 1e-300 below the L's ``{y=0, x>0}`` edge is on the boundary for
        every practical purpose; ``LShapedPoissonOperator`` wraps it to
        ``theta ~ 2*pi`` and returns ``sin(4*pi/3) * 0.5**(2/3)``.
        """
        point = np.array([[0.5, -1e-300]])
        assert float(_lshape_operator().exact_solution(point)[0]) == pytest.approx(
            math.sin(4 * math.pi / 3) * 0.5 ** (2 / 3), rel=1e-6
        )
        assert float(build_lshape_multi_corner_operator().exact_solution(point)[0]) == 0.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_boundary_value_is_the_exact_trace(self, backend: str) -> None:
        operator = build_zshape_poisson_operator()
        boundary = operator.geometry.sample_boundary(
            500, generator=torch.Generator().manual_seed(4)
        )
        coords: object = boundary.numpy() if backend == "numpy" else boundary.double()
        trace = operator.boundary_value(coords)  # type: ignore[arg-type]
        exact = operator.exact_solution(coords)  # type: ignore[arg-type]
        assert np.array_equal(np.asarray(trace), np.asarray(exact))

    def test_numpy_and_torch_paths_agree(self) -> None:
        operator = build_zshape_poisson_operator(secondary_coefficient=-0.7)
        pts = operator.geometry.sample_interior(2000, generator=torch.Generator().manual_seed(9))
        via_numpy = operator.exact_solution(pts.numpy())
        via_torch = operator.exact_solution(pts.double())
        assert isinstance(via_torch, torch.Tensor)
        np.testing.assert_allclose(via_numpy, via_torch.numpy(), rtol=0.0, atol=FLOAT32_ATOL)


class TestHarmonicity:
    @settings(max_examples=60, deadline=None)
    @given(
        cell_index=st.integers(min_value=0, max_value=len(ZSHAPE_POLYOMINO_CELLS) - 1),
        u=st.floats(min_value=0.0, max_value=1.0),
        v=st.floats(min_value=0.0, max_value=1.0),
        primary=st.floats(min_value=0.05, max_value=COEFFICIENT_BOUND),
        secondary=st.floats(min_value=-COEFFICIENT_BOUND, max_value=COEFFICIENT_BOUND),
    )
    def test_five_point_laplacian_is_within_the_derived_bound(
        self, cell_index: int, u: float, v: float, primary: float, secondary: float
    ) -> None:
        operator = build_zshape_poisson_operator(
            primary_coefficient=primary, secondary_coefficient=secondary
        )
        cell = ZSHAPE_POLYOMINO_CELLS[cell_index]
        x = cell.x0 + u * (cell.x1 - cell.x0)
        y = cell.y0 + v * (cell.y1 - cell.y0)
        assume(_stencil_is_admissible(operator, x, y))
        assert abs(_five_point_laplacian(operator, x, y)) <= _laplacian_bound(operator, x, y)

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (0.0, 0.3),  # A|B interface
            (0.0, 0.7),
            (0.3, 0.0),  # B|C interface: the naive [0, 2pi) cut of corner 1 lives here
            (0.7, 0.0),
            (1.0, -0.3),  # C|D interface
            (1.0, -0.7),
        ],
    )
    def test_laplacian_is_bounded_across_internal_interfaces(self, x: float, y: float) -> None:
        """A cut through an interface is a kink or jump: an O(1/h) Laplacian here."""
        operator = build_zshape_poisson_operator()
        assert _stencil_is_admissible(operator, x, y)
        assert abs(_five_point_laplacian(operator, x, y)) <= _laplacian_bound(operator, x, y)

    def test_the_bound_is_discriminating(self) -> None:
        """Guards the guard: a unit Laplacian must exceed the bound by orders of magnitude."""
        operator = build_zshape_poisson_operator()
        assert _laplacian_bound(operator, 0.5, 0.5) < 1e-3

    @settings(max_examples=40, deadline=None)
    @given(
        cell_index=st.integers(min_value=0, max_value=len(ZSHAPE_POLYOMINO_CELLS) - 1),
        u=st.floats(min_value=0.0, max_value=1.0),
        v=st.floats(min_value=0.0, max_value=1.0),
        primary=st.floats(min_value=0.05, max_value=COEFFICIENT_BOUND),
        secondary=st.floats(min_value=-COEFFICIENT_BOUND, max_value=COEFFICIENT_BOUND),
    )
    def test_autograd_residual_vanishes_on_the_exact_solution(
        self, cell_index: int, u: float, v: float, primary: float, secondary: float
    ) -> None:
        operator = build_zshape_poisson_operator(
            primary_coefficient=primary, secondary_coefficient=secondary
        )
        cell = ZSHAPE_POLYOMINO_CELLS[cell_index]
        x = cell.x0 + u * (cell.x1 - cell.x0)
        y = cell.y0 + v * (cell.y1 - cell.y0)
        assume(all(math.hypot(x - t.x, y - t.y) >= MIN_CORNER_DISTANCE for t in operator.corners))
        coords = torch.tensor([[x, y]], dtype=torch.float64, requires_grad=True)
        solution = operator.exact_solution(coords)
        assert isinstance(solution, torch.Tensor)
        residual = operator.residual(solution, coords)
        assert residual.max_norm <= AUTOGRAD_LAPLACIAN_ATOL


class TestCornerGuard:
    def test_value_at_the_corner_is_zero(self) -> None:
        term = build_zshape_poisson_operator().corners[0]
        origin = np.zeros((1, 2))
        assert singular_term_values(origin, term).tolist() == [0.0]
        assert singular_term_tensor(torch.zeros(1, 2), term).tolist() == [0.0]

    def test_gradient_at_the_corner_is_finite(self) -> None:
        term = build_zshape_poisson_operator().corners[0]
        coords = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        singular_term_tensor(coords, term).sum().backward()
        assert coords.grad is not None
        assert bool(torch.isfinite(coords.grad).all())


class TestBranchCutValidator:
    def test_presets_are_accepted(self) -> None:
        assert len(build_lshape_multi_corner_operator().corners) == 1
        assert len(build_zshape_poisson_operator().corners) == 2

    def test_cut_ray_through_the_interior_is_rejected(self) -> None:
        corner = SingularCornerTerm(x=1.0, y=1.0, exterior_bisector=math.pi / 4)
        with pytest.raises(BranchCutError, match="branch cut"):
            _operator(TALL_U_CELLS, (corner,))

    def test_cut_ray_grazing_the_closure_is_rejected(self) -> None:
        corner = SingularCornerTerm(x=1.0, y=1.0, exterior_bisector=math.pi / 4)
        with pytest.raises(BranchCutError, match="branch cut"):
            _operator(SHORT_U_CELLS, (corner,))

    def test_bisector_pointing_into_the_domain_is_rejected(self) -> None:
        corner = SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=0.75 * math.pi)
        with pytest.raises(CornerDeclarationError, match="exterior_bisector"):
            _operator(LSHAPE_POLYOMINO_CELLS, (corner,))

    def test_bisector_off_the_exterior_bisector_is_rejected(self) -> None:
        """Still pointing outside -- but s_i would no longer vanish on the corner's edges."""
        corner = SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR + 0.3)
        with pytest.raises(CornerDeclarationError, match="exterior_bisector"):
            _operator(LSHAPE_POLYOMINO_CELLS, (corner,))

    def test_bisector_is_compared_modulo_a_full_turn(self) -> None:
        shifted = SingularCornerTerm(
            x=0.0, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR + 2.0 * math.pi
        )
        operator = _operator(LSHAPE_POLYOMINO_CELLS, (shifted,))
        pts = torch.tensor([[0.3, 0.4], [-0.5, -0.5], [-0.2, 0.9]], dtype=torch.float64)
        expected = build_lshape_multi_corner_operator().exact_solution(pts)
        assert torch.allclose(operator.exact_solution(pts), expected, atol=1e-12)  # type: ignore[arg-type]

    @pytest.mark.parametrize("position", [(1.0, 1.0), (0.5, 0.0), (5.0, 5.0)])
    def test_a_position_that_is_not_a_reentrant_corner_is_rejected(
        self, position: tuple[float, float]
    ) -> None:
        corner = SingularCornerTerm(
            x=position[0], y=position[1], exterior_bisector=LSHAPE_EXTERIOR_BISECTOR
        )
        with pytest.raises(CornerDeclarationError, match="not a reentrant corner"):
            _operator(LSHAPE_POLYOMINO_CELLS, (corner,))

    def test_a_wrong_interior_angle_is_rejected(self) -> None:
        corner = SingularCornerTerm(
            x=0.0, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR, interior_angle=1.75 * math.pi
        )
        with pytest.raises(CornerDeclarationError, match="interior angle"):
            _operator(LSHAPE_POLYOMINO_CELLS, (corner,))


class TestConstructionGuards:
    def test_no_corners_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _operator(LSHAPE_POLYOMINO_CELLS, ())

    def test_duplicate_corners_are_rejected(self) -> None:
        corner = SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR)
        with pytest.raises(ValueError, match="more than once"):
            _operator(LSHAPE_POLYOMINO_CELLS, (corner, corner))

    def test_a_near_duplicate_corner_is_rejected(self) -> None:
        """Two declarations matching ONE geometric corner would add its term twice.

        The reviewer's case: the duplicate check compared declared positions exactly
        while matching allows ``CORNER_POSITION_ATOL``, so (0, 0) and (5e-13, 0) were
        both accepted and u(0.5, 0.5) was 0.7937 -- twice the one-corner 0.3969.
        """
        first = SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR)
        near = SingularCornerTerm(x=5e-13, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR)
        with pytest.raises(ValueError, match="more than once"):
            _operator(LSHAPE_POLYOMINO_CELLS, (first, near))

    @settings(max_examples=40, deadline=None)
    @given(
        dx=st.floats(min_value=-mcp.CORNER_POSITION_ATOL, max_value=mcp.CORNER_POSITION_ATOL),
        dy=st.floats(min_value=-mcp.CORNER_POSITION_ATOL, max_value=mcp.CORNER_POSITION_ATOL),
    )
    def test_every_declaration_matching_a_declared_corner_is_rejected(
        self, dx: float, dy: float
    ) -> None:
        """Across the matching box, on the Z's *secondary* corner (x = 1, not the origin).

        ``assume`` keeps declarations that match: at the box's edge ``1.0 + 1e-12``
        rounds past the tolerance, and such a point is no corner at all
        (``CornerDeclarationError``), not a duplicate.
        """
        z_corners = build_zshape_poisson_operator().corners
        near = z_corners[1].model_copy(update={"x": z_corners[1].x + dx, "y": z_corners[1].y + dy})
        domain = PolyominoDomain(ZSHAPE_POLYOMINO_CELLS)
        assume(domain.corner_at(near.position, atol=mcp.CORNER_POSITION_ATOL) is not None)
        with pytest.raises(ValueError, match="more than once"):
            _operator(ZSHAPE_POLYOMINO_CELLS, (*z_corners, near))

    def test_all_zero_coefficients_are_rejected(self) -> None:
        """``u == 0`` would make every measured error 0.0 -- the degenerate-substrate class."""
        with pytest.raises(ValueError, match="zero"):
            build_zshape_poisson_operator(primary_coefficient=0.0, secondary_coefficient=0.0)

    def test_one_zero_coefficient_is_allowed(self) -> None:
        only_primary = build_zshape_poisson_operator(secondary_coefficient=0.0)
        pts = np.array([[0.4, 0.4], [1.5, -0.5]])
        expected = singular_term_values(pts, only_primary.corners[0]).astype(np.float32)
        np.testing.assert_array_equal(only_primary.exact_solution(pts), expected)

    def test_config_box_must_be_the_domain_box(self) -> None:
        domain = PolyominoDomain(LSHAPE_POLYOMINO_CELLS)
        corner = SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR)
        config = PDEConfig(name="wrong_box", pde_type=PDEType.POISSON)
        with pytest.raises(ValueError, match="bounding box"):
            MultiCornerPoissonOperator(config, domain=domain, corners=(corner,))

    @pytest.mark.parametrize("coordinate", range(4), ids=["x_min", "y_min", "x_max", "y_max"])
    @pytest.mark.parametrize(
        ("offset", "accepted"),
        [
            (0.5 * mcp.DOMAIN_BOUNDS_ATOL, True),
            (2.0 * mcp.DOMAIN_BOUNDS_ATOL, False),
            (1.5e-5, False),
        ],
        ids=["half_the_atol", "twice_the_atol", "reviewer_1.5e-5"],
    )
    def test_the_box_tolerance_is_the_stated_absolute_one(
        self, coordinate: int, offset: float, accepted: bool
    ) -> None:
        """``DOMAIN_BOUNDS_ATOL`` is the tolerance applied, on each coordinate of the Z box.

        ``np.allclose``'s default ``rtol=1e-5``, scaled by a coordinate of up to 2,
        used to dominate it: a box off by 1.5e-5 or by twice the atol was accepted.
        """
        domain = PolyominoDomain(ZSHAPE_POLYOMINO_CELLS)
        box = [value for bound in domain.bounding_box() for value in bound]
        box[coordinate] += offset
        config = PDEConfig(
            name="z_box", pde_type=PDEType.POISSON, domain_min=box[:2], domain_max=box[2:]
        )
        corners = build_zshape_poisson_operator().corners
        if accepted:
            operator = MultiCornerPoissonOperator(config, domain=domain, corners=corners)
            assert operator.corners == corners
            return
        with pytest.raises(ValueError, match="bounding box"):
            MultiCornerPoissonOperator(config, domain=domain, corners=corners)

    def test_config_must_be_planar(self) -> None:
        domain = PolyominoDomain(LSHAPE_POLYOMINO_CELLS)
        corner = SingularCornerTerm(x=0.0, y=0.0, exterior_bisector=LSHAPE_EXTERIOR_BISECTOR)
        config = PDEConfig(
            name="three_d",
            pde_type=PDEType.POISSON,
            domain_dim=3,
            domain_min=[-1.0, -1.0, -1.0],
            domain_max=[1.0, 1.0, 1.0],
            advection_coeff=[0.0, 0.0, 0.0],
        )
        with pytest.raises(ValueError, match="planar"):
            MultiCornerPoissonOperator(config, domain=domain, corners=(corner,))

    def test_presets_take_their_box_from_the_domain(self) -> None:
        operator = build_zshape_poisson_operator()
        assert operator.domain_min.tolist() == [-1.0, -1.0]
        assert operator.domain_max.tolist() == [2.0, 1.0]

    def test_default_z_strengths_are_the_named_constants(self) -> None:
        coefficients = [t.coefficient for t in build_zshape_poisson_operator().corners]
        assert coefficients == [
            DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
            DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
        ]
        custom = build_zshape_poisson_operator(primary_coefficient=2.0, secondary_coefficient=0.5)
        assert [t.coefficient for t in custom.corners] == [2.0, 0.5]


class TestOperatorSurface:
    def test_registry_round_trip(self) -> None:
        assert "poisson_multi_corner" in list_pde_operators()
        assert get_pde_operator("poisson_multi_corner") is MultiCornerPoissonOperator

    def test_exported_from_the_operators_package_like_its_siblings(self) -> None:
        """Re-exported and in ``__all__``; the submodule binding is not leaked."""
        import src.pde.operators as operators_package

        assert operators_package.MultiCornerPoissonOperator is MultiCornerPoissonOperator
        assert "MultiCornerPoissonOperator" in operators_package.__all__
        assert "multi_corner_poisson" not in dir(operators_package)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_source_is_zero(self, backend: str) -> None:
        operator = build_zshape_poisson_operator()
        coords: object = np.ones((4, 2), np.float32) if backend == "numpy" else torch.ones(4, 2)
        source = operator.source_term(coords)  # type: ignore[arg-type]
        assert np.asarray(source).tolist() == [0.0] * 4

    def test_residual_of_a_non_harmonic_field(self) -> None:
        """``u = x^2 + y^2`` has Laplacian 4, so the residual is -4 everywhere."""
        operator = build_zshape_poisson_operator()
        coords = torch.tensor([[0.5, 0.5], [1.5, -0.5]], dtype=torch.float64, requires_grad=True)
        u = coords[:, 0] ** 2 + coords[:, 1] ** 2
        residual = operator.residual(u, coords, compute_derivatives=False)
        assert torch.allclose(residual.values, torch.full((2,), -4.0, dtype=torch.float64))
        assert residual.derivatives == {}
        assert residual.max_norm == pytest.approx(4.0)

    def test_compute_error(self) -> None:
        operator = build_zshape_poisson_operator()
        coords = operator.geometry.sample_interior(50, generator=torch.Generator().manual_seed(1))
        exact = operator.exact_solution(coords)
        assert isinstance(exact, torch.Tensor)
        assert operator.compute_error(exact, coords) == {
            "l2_error": 0.0,
            "linf_error": 0.0,
            "mse": 0.0,
        }
        shifted = operator.compute_error(exact + 1.0, coords)
        assert shifted["l2_error"] == pytest.approx(1.0)
        assert shifted["linf_error"] == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_is_boundary_point(self, backend: str) -> None:
        operator = build_zshape_poisson_operator()
        pts = np.array([[0.0, -0.5], [0.5, 0.5], [1.5, 0.0]], dtype=np.float32)
        coords: object = pts if backend == "numpy" else torch.from_numpy(pts)
        mask = operator.is_boundary_point(coords)  # type: ignore[arg-type]
        assert np.asarray(mask).tolist() == [True, False, True]

    def test_collocation_points_are_inside_and_reproducible(self) -> None:
        operator = build_zshape_poisson_operator()
        state = torch.get_rng_state()
        first = operator.generate_collocation_points(300, seed=5)
        second = operator.generate_collocation_points(300, seed=5)
        assert torch.equal(torch.get_rng_state(), state), "a seed must not re-seed torch globally"
        np.testing.assert_array_equal(first, second)
        assert first.dtype == np.float32
        assert bool(operator.geometry.contains_point(torch.from_numpy(first)).all())
        assert operator.generate_collocation_points(10).shape == (10, 2)

    def test_boundary_points_cover_every_face(self) -> None:
        operator = build_zshape_poisson_operator()
        pts = operator.generate_boundary_points(7, seed=3)
        assert pts.shape == (7 * len(operator.geometry.boundary_segments), 2)
        assert bool(operator.geometry.is_boundary(torch.from_numpy(pts)).all())

    def test_to_dict_carries_the_corners_and_cells(self) -> None:
        data = build_zshape_poisson_operator(secondary_coefficient=0.5).to_dict()
        assert data["name"] == "poisson_multi_corner"
        assert [c["coefficient"] for c in data["corners"]] == [1.0, 0.5]
        assert len(data["cells"]) == len(ZSHAPE_POLYOMINO_CELLS)


class _RecordingLogger:
    """Stand-in for the module logger (structlog caches bound loggers)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def _record(self, level: str, event: str, **kwargs: object) -> None:
        self.events.append((level, event, kwargs))

    def info(self, event: str, **kwargs: object) -> None:
        self._record("info", event, **kwargs)

    def debug(self, event: str, **kwargs: object) -> None:
        self._record("debug", event, **kwargs)

    def names(self, level: str) -> list[str]:
        return [event for lvl, event, _ in self.events if lvl == level]


class TestLogging:
    def test_construction_logs_a_summary_and_per_corner_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _RecordingLogger()
        monkeypatch.setattr(mcp, "logger", recorder)
        build_zshape_poisson_operator()
        assert recorder.names("info") == [
            "branch_cut_validation_passed",
            "multi_corner_poisson_operator_created",
        ]
        assert recorder.names("debug") == ["singular_corner_validated"] * 2
        summary = recorder.events[-1][2]
        assert summary["corners"] == [(0.0, 0.0, 1.0), (1.0, 0.0, 0.25)]

    def test_a_substituted_collocation_method_is_logged_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operator = build_zshape_poisson_operator()
        recorder = _RecordingLogger()
        monkeypatch.setattr(mcp, "logger", recorder)
        operator.generate_collocation_points(5, method="lhs")
        operator.generate_collocation_points(5, method="uniform")
        operator.generate_collocation_points(5)
        assert recorder.names("info") == ["collocation_method_substituted"]
