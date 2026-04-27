"""Tests for the SDF-backed ``PicoGKDomain`` adapter.

These tests use the closed-form ``AnalyticalHelixSDF`` only so that the
suite never depends on the optional ``[picogk]`` extra (.NET / pythonnet).
"""

from __future__ import annotations

import pytest
import torch

from src.pde.geometry import (
    DomainGeometry,
    GeometryConfig,
    GeometryType,
    create_geometry,
)
from src.pde.geometry_picogk import PicoGKDomain
from src.pde.sdf import AnalyticalHelixSDF

HELIX_R_MAJOR = 0.05
HELIX_R_MINOR = 0.012
HELIX_PITCH = 0.02
HELIX_N_TURNS = 3


@pytest.fixture
def sdf() -> AnalyticalHelixSDF:
    return AnalyticalHelixSDF(
        R_major=HELIX_R_MAJOR,
        r_minor=HELIX_R_MINOR,
        pitch=HELIX_PITCH,
        n_turns=HELIX_N_TURNS,
    )


@pytest.fixture
def domain(sdf: AnalyticalHelixSDF) -> PicoGKDomain:
    # Smaller volume_samples keeps test init fast; we only need O(1) MC
    # accuracy for the area smoke check.
    return PicoGKDomain(sdf_evaluator=sdf, volume_samples=2048)


class TestPicoGKDomainConstruction:
    def test_implements_domain_geometry_abc(self, domain: PicoGKDomain) -> None:
        assert isinstance(domain, DomainGeometry)

    def test_dim_matches_sdf(self, domain: PicoGKDomain) -> None:
        assert domain.dim == 3

    def test_bounding_box_matches_sdf(self, domain: PicoGKDomain, sdf: AnalyticalHelixSDF) -> None:
        assert domain.bounding_box() == sdf.bounding_box()

    def test_invalid_oversample_factor(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="oversample_factor"):
            PicoGKDomain(sdf_evaluator=sdf, oversample_factor=1.0)

    def test_invalid_boundary_tolerance(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="boundary_tolerance"):
            PicoGKDomain(sdf_evaluator=sdf, boundary_tolerance=0.0)

    def test_invalid_volume_samples(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="volume_samples"):
            PicoGKDomain(sdf_evaluator=sdf, volume_samples=0)

    def test_invalid_grad_epsilon(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="grad_epsilon"):
            PicoGKDomain(sdf_evaluator=sdf, volume_samples=64, grad_epsilon=0.0)

    def test_invalid_max_oversample(self, sdf: AnalyticalHelixSDF) -> None:
        # max_oversample must be strictly greater than oversample_factor.
        with pytest.raises(ValueError, match="max_oversample"):
            PicoGKDomain(
                sdf_evaluator=sdf,
                volume_samples=64,
                oversample_factor=10.0,
                max_oversample=10.0,
            )

    def test_invalid_projection_max_iters(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="projection_max_iters"):
            PicoGKDomain(
                sdf_evaluator=sdf,
                volume_samples=64,
                projection_max_iters=0,
            )

    def test_invalid_min_grad_norm_sq(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="min_grad_norm_sq"):
            PicoGKDomain(
                sdf_evaluator=sdf,
                volume_samples=64,
                min_grad_norm_sq=0.0,
            )

    def test_inverted_bounding_box_raises(self) -> None:
        """An SDF with min >= max in any axis must be rejected."""

        class _BadBoundsSDF:
            dim = 3

            def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
                # x-axis is inverted (min == max == 1.0) → degenerate bbox.
                return (1.0, -1.0, -1.0), (1.0, 1.0, 1.0)

            def sdf(self, points):  # pragma: no cover - never called
                import torch as _torch

                return _torch.zeros(points.shape[0])

        with pytest.raises(ValueError, match="max .* <= min"):
            PicoGKDomain(sdf_evaluator=_BadBoundsSDF())  # type: ignore[arg-type]

    def test_mismatched_bounding_box_lengths_raises(self) -> None:
        class _BadBoundsSDF:
            dim = 3

            def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
                return (0.0, 0.0), (1.0, 1.0, 1.0)  # len 2 vs len 3

            def sdf(self, points):  # pragma: no cover
                import torch as _torch

                return _torch.zeros(points.shape[0])

        with pytest.raises(ValueError, match="length mismatch"):
            PicoGKDomain(sdf_evaluator=_BadBoundsSDF())  # type: ignore[arg-type]


class TestPicoGKDomainEmptySDFFailures:
    """``sample_interior`` / ``sample_boundary`` must raise when the SDF is unworkable."""

    def test_sample_interior_raises_when_sdf_always_positive(self) -> None:
        """If the SDF reports every point as exterior, sampling must fail loud."""
        from src.pde.geometry_picogk import PicoGKDomain

        class _AlwaysOutsideSDF:
            dim = 3

            def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
                return (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)

            def sdf(self, points):
                import torch as _torch

                return _torch.ones(points.shape[0])

        domain = PicoGKDomain(
            sdf_evaluator=_AlwaysOutsideSDF(),  # type: ignore[arg-type]
            volume_samples=64,
        )
        with pytest.raises(RuntimeError, match="any interior points"):
            domain.sample_interior(8)

    def test_sample_boundary_raises_when_sdf_has_no_zero_crossing(self) -> None:
        """If the SDF never gets close to zero, the projector must fail loud."""
        from src.pde.geometry_picogk import PicoGKDomain

        class _FlatPositiveSDF:
            dim = 3

            def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
                return (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)

            def sdf(self, points):
                import torch as _torch

                # Constant 1.0 with zero gradient → projection cannot converge.
                return _torch.ones(points.shape[0])

        domain = PicoGKDomain(
            sdf_evaluator=_FlatPositiveSDF(),  # type: ignore[arg-type]
            volume_samples=64,
            boundary_tolerance=1e-6,
        )
        with pytest.raises(RuntimeError, match="zero level set"):
            domain.sample_boundary(4)

    def test_volume_estimate_within_bbox_volume(
        self, domain: PicoGKDomain, sdf: AnalyticalHelixSDF
    ) -> None:
        (mins, maxs) = sdf.bounding_box()
        bbox_volume = 1.0
        for lo, hi in zip(mins, maxs, strict=True):
            bbox_volume *= hi - lo
        assert 0 < domain.area < bbox_volume


class TestPicoGKDomainContains:
    def test_centerline_points_are_inside(
        self, domain: PicoGKDomain, sdf: AnalyticalHelixSDF
    ) -> None:
        ts = torch.linspace(0.1, HELIX_N_TURNS - 0.1, steps=20)
        points = sdf._centerline(ts)
        assert torch.all(domain.contains_point(points))

    def test_far_points_are_outside(self, domain: PicoGKDomain) -> None:
        far = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [0.0, 0.0, 100.0],
            ]
        )
        assert not torch.any(domain.contains_point(far))

    def test_is_boundary_consistent_with_sdf(
        self, domain: PicoGKDomain, sdf: AnalyticalHelixSDF
    ) -> None:
        # Build a small batch of points: some interior, some on the
        # boundary, some far. Check is_boundary matches |sdf| < tol.
        ts = torch.linspace(0.2, HELIX_N_TURNS - 0.2, steps=8)
        # Surface points: centerline + r * normal, where normal is the
        # principal normal (toward the helix axis projected onto xy).
        center = sdf._centerline(ts)
        # Use radial outward direction in xy as a rough normal.
        radial = torch.stack([center[:, 0], center[:, 1], torch.zeros_like(ts)], dim=-1)
        radial = radial / torch.linalg.norm(radial, dim=-1, keepdim=True)
        on_surface = center + HELIX_R_MINOR * radial
        # Tolerance must be loose enough to absorb the Newton residual.
        boundary_mask = domain.is_boundary(on_surface, tol=5e-4)
        assert torch.sum(boundary_mask).item() >= ts.shape[0] - 1


class TestPicoGKDomainSampling:
    def test_sample_interior_returns_correct_shape(self, domain: PicoGKDomain) -> None:
        points = domain.sample_interior(128)
        assert points.shape == (128, 3)
        # All must be strictly interior.
        assert torch.all(domain.contains_point(points))

    def test_sample_interior_invalid_n(self, domain: PicoGKDomain) -> None:
        with pytest.raises(ValueError):
            domain.sample_interior(0)

    def test_sample_boundary_returns_correct_shape(self, domain: PicoGKDomain) -> None:
        points = domain.sample_boundary(64)
        assert points.shape == (64, 3)
        # Newton projection must converge to within boundary_tolerance.
        residual = domain.sdf_evaluator.sdf(points).abs()
        assert torch.all(residual < domain.boundary_tolerance)

    def test_sample_boundary_invalid_n(self, domain: PicoGKDomain) -> None:
        with pytest.raises(ValueError):
            domain.sample_boundary(0)

    def test_sample_interior_is_within_bounding_box(self, domain: PicoGKDomain) -> None:
        (mins, maxs) = domain.bounding_box()
        points = domain.sample_interior(256)
        for d, (lo, hi) in enumerate(zip(mins, maxs, strict=True)):
            assert torch.all(points[:, d] >= lo - 1e-6)
            assert torch.all(points[:, d] <= hi + 1e-6)


class TestPicoGKDomainAcceptRate:
    """The ``volume_accept_rate`` property must reflect the MC pass."""

    def test_accept_rate_in_unit_interval(self, domain: PicoGKDomain) -> None:
        rate = domain.volume_accept_rate
        assert 0.0 < rate < 1.0

    def test_accept_rate_consistent_with_volume(
        self, domain: PicoGKDomain, sdf: AnalyticalHelixSDF
    ) -> None:
        # area = accept_rate * bbox_volume
        (mins, maxs) = sdf.bounding_box()
        bbox_volume = 1.0
        for lo, hi in zip(mins, maxs, strict=True):
            bbox_volume *= hi - lo
        derived = domain.volume_accept_rate * bbox_volume
        assert derived == pytest.approx(domain.area, rel=1e-6)


class TestPicoGKDomainBisectionFallback:
    """Bisection fallback recovers points where Newton stalls."""

    def test_disable_fallback_round_trips(self, sdf: AnalyticalHelixSDF) -> None:
        # Constructing with the fallback disabled must still satisfy every
        # invariant; only points that previously needed the fallback may
        # come back slightly off-surface (still bounded by the Newton
        # tolerance for typical helix params).
        d = PicoGKDomain(
            sdf_evaluator=sdf,
            volume_samples=512,
            enable_bisection_fallback=False,
        )
        pts = d.sample_boundary(32)
        assert pts.shape == (32, 3)

    def test_bracket_factor_validation(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="bisection_bracket_factor"):
            PicoGKDomain(
                sdf_evaluator=sdf,
                volume_samples=64,
                bisection_bracket_factor=0.0,
            )

    def test_max_iters_validation(self, sdf: AnalyticalHelixSDF) -> None:
        with pytest.raises(ValueError, match="bisection_max_iters"):
            PicoGKDomain(
                sdf_evaluator=sdf,
                volume_samples=64,
                bisection_max_iters=0,
            )

    def test_fallback_drives_residual_below_tolerance(self, sdf: AnalyticalHelixSDF) -> None:
        """The fallback must drive every accepted point under the tolerance.

        Use a tight tolerance and a tight Newton budget so the fallback
        is the path that closes the gap.
        """
        d = PicoGKDomain(
            sdf_evaluator=sdf,
            volume_samples=512,
            boundary_tolerance=1e-5,
            projection_max_iters=2,  # tiny Newton budget
            enable_bisection_fallback=True,
            bisection_max_iters=24,
        )
        pts = d.sample_boundary(32)
        residual = d.sdf_evaluator.sdf(pts).abs()
        assert torch.all(residual < d.boundary_tolerance)

    def test_flat_sdf_still_fails_loud(self) -> None:
        """A degenerate SDF must still raise even with the fallback enabled."""
        from src.pde.geometry_picogk import PicoGKDomain

        class _FlatPositiveSDF:
            dim = 3

            def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
                return (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)

            def sdf(self, points):
                return torch.ones(points.shape[0])

        domain = PicoGKDomain(
            sdf_evaluator=_FlatPositiveSDF(),  # type: ignore[arg-type]
            volume_samples=64,
            boundary_tolerance=1e-6,
            enable_bisection_fallback=True,
        )
        with pytest.raises(RuntimeError, match="zero level set"):
            domain.sample_boundary(4)

    def test_bisection_fallback_no_op_when_all_converged(
        self, domain: PicoGKDomain, sdf: AnalyticalHelixSDF
    ) -> None:
        """No-op when every input point already satisfies |sdf| < tol.

        Covers the ``n_failing == 0`` early-exit branch of
        ``_bisection_fallback``.
        """
        ts = torch.linspace(0.2, sdf.n_turns - 0.2, steps=8)
        center = sdf._centerline(ts)
        radial = torch.stack([center[:, 0], center[:, 1], torch.zeros_like(ts)], dim=-1)
        radial = radial / torch.linalg.norm(radial, dim=-1, keepdim=True)
        on_surface = center + sdf.r_minor * radial
        # Loose tolerance so every point trivially satisfies |sdf| < tol.
        d = PicoGKDomain(
            sdf_evaluator=sdf,
            volume_samples=64,
            boundary_tolerance=1.0,
            enable_bisection_fallback=True,
        )
        out = d._bisection_fallback(on_surface)
        assert torch.equal(out, on_surface)

    def test_projection_converged_log_branch(self, sdf: AnalyticalHelixSDF) -> None:
        """First-iteration convergence triggers the projection-converged log.

        Covers the ``picogk_projection_converged`` break path inside
        ``_project_to_surface`` by relaxing ``boundary_tolerance`` enough
        that any candidate already satisfies it.
        """
        d = PicoGKDomain(
            sdf_evaluator=sdf,
            volume_samples=64,
            boundary_tolerance=10.0,  # any candidate already satisfies |sdf| < tol
            projection_max_iters=4,
        )
        # Sample some interior points — any random batch will do — then
        # run them through the projector. The first iter checks the
        # tolerance and breaks.
        pts = d.sample_interior(8)
        projected = d._project_to_surface(pts)
        # No crash; shape preserved.
        assert projected.shape == pts.shape


class TestCreateGeometryFactory:
    """The PICOGK enum branch of create_geometry must build a PicoGKDomain."""

    def test_factory_dispatches_to_picogk_domain(self) -> None:
        config = GeometryConfig(
            geometry_type=GeometryType.PICOGK,
            sdf_kind="analytical_helix",
            helix_R_major=HELIX_R_MAJOR,
            helix_r_minor=HELIX_R_MINOR,
            helix_pitch=HELIX_PITCH,
            helix_n_turns=HELIX_N_TURNS,
        )
        domain = create_geometry(config)
        assert isinstance(domain, PicoGKDomain)
        assert domain.dim == 3

    def test_factory_picogk_kind_requires_voxel_path(self) -> None:
        config = GeometryConfig(
            geometry_type=GeometryType.PICOGK,
            sdf_kind="picogk",
            picogk_voxel_path=None,
        )
        with pytest.raises(ValueError, match="picogk_voxel_path"):
            create_geometry(config)
