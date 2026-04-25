"""Tests for the analytical helix SDF and the PicoGK lazy-import stub."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pde.sdf import AnalyticalHelixSDF, PicoGKSDFEvaluator, SDFEvaluator

# Use a slightly larger helix-tube ratio than production defaults so the
# Newton projector has a forgiving curvature regime in unit tests.
HELIX_R_MAJOR = 0.05
HELIX_R_MINOR = 0.012
HELIX_PITCH = 0.02
HELIX_N_TURNS = 3


@pytest.fixture
def helix() -> AnalyticalHelixSDF:
    """Standard helical-tube SDF used across the test suite."""
    return AnalyticalHelixSDF(
        R_major=HELIX_R_MAJOR,
        r_minor=HELIX_R_MINOR,
        pitch=HELIX_PITCH,
        n_turns=HELIX_N_TURNS,
    )


class TestAnalyticalHelixSDFConstruction:
    """Validation logic in __init__."""

    def test_defaults_construct(self) -> None:
        sdf = AnalyticalHelixSDF()
        assert sdf.dim == 3
        assert sdf.R_major > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"R_major": 0.0},
            {"R_major": -1.0},
            {"r_minor": 0.0},
            {"pitch": 0.0},
            {"n_turns": 0},
            {"newton_max_iters": 0},
            {"newton_deriv_tol": 0.0},
            {"newton_deriv_tol": -1e-9},
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            AnalyticalHelixSDF(**kwargs)

    def test_newton_overrides_take_effect(self) -> None:
        """Custom Newton settings must propagate to instance state."""
        sdf = AnalyticalHelixSDF(newton_max_iters=4, newton_deriv_tol=1e-6)
        assert sdf.newton_max_iters == 4
        assert sdf.newton_deriv_tol == pytest.approx(1e-6)

    def test_self_intersection_blocked(self) -> None:
        # r_minor >= R_major creates a self-intersecting torus.
        with pytest.raises(ValueError, match="self-intersection"):
            AnalyticalHelixSDF(R_major=0.01, r_minor=0.01)


class TestAnalyticalHelixSDFGeometry:
    """Behavior of the SDF on known geometric configurations."""

    def test_bounding_box_contains_helix(self, helix: AnalyticalHelixSDF) -> None:
        (mins, maxs) = helix.bounding_box()
        assert len(mins) == 3
        assert len(maxs) == 3
        # The centerline x and y oscillate between +/- R_major; the
        # bounding box must include at least that range plus the tube
        # radius.
        outer = HELIX_R_MAJOR + HELIX_R_MINOR
        assert mins[0] <= -outer + 1e-9
        assert maxs[0] >= outer - 1e-9
        assert mins[2] <= 0.0 + 1e-9
        assert maxs[2] >= HELIX_PITCH * HELIX_N_TURNS - 1e-9

    def test_centerline_is_inside(self, helix: AnalyticalHelixSDF) -> None:
        # Pick a sample of t values along the centerline; sdf there should
        # be close to -r_minor.
        ts = torch.linspace(0.1, HELIX_N_TURNS - 0.1, steps=20)
        centerline = helix._centerline(ts)
        values = helix.sdf(centerline)
        assert torch.all(values < 0)
        # Values must be close to -r_minor (but Newton may not converge
        # exactly; allow a generous tolerance).
        assert torch.allclose(values, torch.full_like(values, -HELIX_R_MINOR), atol=1e-4)

    def test_far_points_outside(self, helix: AnalyticalHelixSDF) -> None:
        # Points far from the helical tube must have sdf > 0.
        far = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 100.0],
            ]
        )
        values = helix.sdf(far)
        assert torch.all(values > 0)

    def test_central_axis_is_outside(self, helix: AnalyticalHelixSDF) -> None:
        # The helix axis (x=y=0) is at distance R_major from the
        # centerline, which is >> r_minor, so it must be exterior.
        axis = torch.tensor([[0.0, 0.0, HELIX_PITCH * HELIX_N_TURNS / 2]])
        values = helix.sdf(axis)
        assert float(values[0]) > 0
        # Distance from axis to centerline is exactly R_major, so
        # sdf == R_major - r_minor.
        assert float(values[0]) == pytest.approx(HELIX_R_MAJOR - HELIX_R_MINOR, abs=1e-4)

    def test_volume_estimate_matches_analytical(self, helix: AnalyticalHelixSDF) -> None:
        # The closed-form tube volume is pi * r^2 * arc_length.
        volume = helix.volume()
        arc = HELIX_N_TURNS * float(np.sqrt((2 * np.pi * HELIX_R_MAJOR) ** 2 + HELIX_PITCH**2))
        expected = np.pi * HELIX_R_MINOR**2 * arc
        assert volume == pytest.approx(expected, rel=1e-6)

    def test_rejects_2d_points(self, helix: AnalyticalHelixSDF) -> None:
        bad = torch.zeros(4, 2)
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            helix.sdf(bad)

    def test_satisfies_protocol(self, helix: AnalyticalHelixSDF) -> None:
        # AnalyticalHelixSDF must satisfy the SDFEvaluator runtime Protocol
        # so PicoGKDomain accepts it.
        assert isinstance(helix, SDFEvaluator)


class TestAnalyticalHelixSDFGradient:
    """Finite-difference sanity check on the SDF gradient direction."""

    def test_gradient_satisfies_eikonal(self, helix: AnalyticalHelixSDF) -> None:
        # Any well-defined SDF satisfies the eikonal equation
        # ``||grad(sdf)|| == 1`` away from the medial axis. We sample
        # well-separated exterior points (radial offset >> r_minor) and
        # check the central-difference gradient norm. A few outlier
        # points near the multi-valued medial axis can have residual
        # noise, so we require the median to be very close to 1.
        torch.manual_seed(0)
        n = 128
        radius = HELIX_R_MAJOR + 3 * HELIX_R_MINOR
        theta = torch.rand(n) * 2 * np.pi
        z = torch.rand(n) * (HELIX_PITCH * HELIX_N_TURNS)
        points = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), z], dim=-1)

        eps = 1e-4
        grads = []
        for d in range(3):
            step = torch.zeros_like(points)
            step[:, d] = eps
            fwd = helix.sdf(points + step)
            bwd = helix.sdf(points - step)
            grads.append((fwd - bwd) / (2 * eps))
        grad = torch.stack(grads, dim=-1)
        gnorm = torch.linalg.norm(grad, dim=-1)

        # Median norm must be very close to 1 (true eikonal property);
        # individual outliers near the medial axis are tolerated.
        median = torch.quantile(gnorm, 0.5)
        assert abs(float(median) - 1.0) < 0.05


class TestPicoGKSDFEvaluator:
    """The .NET-backed evaluator must fail loud when the extra is absent."""

    def test_missing_dependency_raises_clear_error(self, tmp_path) -> None:
        fake = tmp_path / "voxel.stl"
        fake.touch()
        # Either pythonnet/PicoGK is not installed (ImportError with our
        # message) or it is installed but voxel ingestion is still
        # unimplemented (NotImplementedError). Both are acceptable failures
        # in v1.
        with pytest.raises((ImportError, NotImplementedError)) as excinfo:
            PicoGKSDFEvaluator(fake)
        msg = str(excinfo.value)
        assert "alphagalerkin[picogk]" in msg or "PicoGK voxel ingestion" in msg
