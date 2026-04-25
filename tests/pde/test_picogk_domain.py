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
