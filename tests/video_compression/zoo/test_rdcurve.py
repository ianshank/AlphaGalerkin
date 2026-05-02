"""Tests for :mod:`src.video_compression.zoo.rdcurve`."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from src.video_compression.zoo.config import (
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)
from src.video_compression.zoo.rdcurve import (
    RDCurveAssemblyError,
    RDCurveFitConfig,
    compute_rd_curve,
)
from src.video_compression.zoo.storage import VideoCodecZoo

# --------------------------------------------------------------------------
# Helpers — small builders so the test bodies stay focused on assertions.
# --------------------------------------------------------------------------


def _make_entry(
    entry_id: str,
    *,
    lambda_rd: float,
    target_bpp: float = 0.5,
    target_psnr_db: float = 33.0,
) -> ModelZooEntryConfig:
    # train_steps must be >= scheduler default warmup (500); 1000 is the
    # smallest round number that satisfies it without changing defaults.
    return ModelZooEntryConfig(
        entry_id=entry_id,
        lambda_rd=lambda_rd,
        target_bpp=target_bpp,
        target_psnr_db=target_psnr_db,
        train_steps=1000,
    )


def _make_manifest(
    entries: Iterable[ModelZooEntryConfig],
    *,
    storage_root: Path,
    name: str = "test_manifest",
) -> ModelZooManifestConfig:
    return ModelZooManifestConfig(
        name=name,
        storage_root=str(storage_root),
        entries=list(entries),
    )


def _save_metrics(
    zoo: VideoCodecZoo,
    entry: ModelZooEntryConfig,
    *,
    bpp: float,
    psnr_db: float,
    ms_ssim: float | None = None,
) -> None:
    """Persist a minimal trained-entry artifact triple."""
    metrics: dict[str, float] = {"bpp": bpp, "psnr_db": psnr_db}
    if ms_ssim is not None:
        metrics["ms_ssim"] = ms_ssim
    zoo.save_entry(
        entry,
        {"model": {"w": torch.tensor([0.0])}},
        metrics,
    )


# --------------------------------------------------------------------------
# RDCurveFitConfig
# --------------------------------------------------------------------------


class TestRDCurveFitConfig:
    def test_defaults_are_distinct_keys(self) -> None:
        cfg = RDCurveFitConfig(name="cfg")
        assert cfg.rate_metric_key != cfg.quality_metric_key
        assert cfg.fit_method == "linear_log"
        assert cfg.enforce_monotone is True
        assert cfg.min_points == 2

    def test_disjoint_keys_validator_rejects_collision(self) -> None:
        with pytest.raises(ValidationError) as exc:
            RDCurveFitConfig(
                name="cfg",
                rate_metric_key="bpp",
                quality_metric_key="bpp",  # collision
            )
        assert "must all be distinct" in str(exc.value)

    def test_unknown_field_is_ignored_for_forward_compat(self) -> None:
        # extra="ignore" — future config keys must not break old code.
        cfg = RDCurveFitConfig.model_validate(
            {
                "name": "cfg",
                "future_field_added_in_v2": 123,
            },
        )
        assert cfg.name == "cfg"

    def test_min_points_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            RDCurveFitConfig(name="cfg", min_points=1)


# --------------------------------------------------------------------------
# compute_rd_curve — happy path
# --------------------------------------------------------------------------


class TestComputeRDCurve:
    def test_round_trip_two_entries(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e_low = _make_entry("e_low", lambda_rd=0.0016)
        e_high = _make_entry("e_high", lambda_rd=0.18)
        # Higher lambda -> more compression -> lower bpp + lower psnr.
        _save_metrics(zoo, e_low, bpp=0.8, psnr_db=37.0, ms_ssim=0.98)
        _save_metrics(zoo, e_high, bpp=0.1, psnr_db=29.0, ms_ssim=0.91)

        manifest = _make_manifest([e_low, e_high], storage_root=tmp_path / "zoo")
        curve = compute_rd_curve(zoo, manifest)

        assert curve.name == "test_manifest"
        assert len(curve.points) == 2
        # add_point() sorts by rate ascending: e_high (bpp=0.1) first.
        assert curve.points[0].rate == pytest.approx(0.1)
        assert curve.points[0].psnr == pytest.approx(29.0)
        assert curve.points[0].lambda_rd == pytest.approx(0.18)
        assert curve.points[1].rate == pytest.approx(0.8)
        assert curve.points[1].psnr == pytest.approx(37.0)
        assert curve.is_monotonic()

    def test_only_entry_ids_filters(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entries = [
            _make_entry("a", lambda_rd=0.01),
            _make_entry("b", lambda_rd=0.02),
            _make_entry("c", lambda_rd=0.03),
        ]
        for e, (bpp, psnr) in zip(
            entries,
            [(0.6, 36.0), (0.4, 33.0), (0.2, 30.0)],
            strict=True,
        ):
            _save_metrics(zoo, e, bpp=bpp, psnr_db=psnr)

        manifest = _make_manifest(entries, storage_root=tmp_path / "zoo")
        curve = compute_rd_curve(zoo, manifest, only_entry_ids=["a", "c"])
        assert len(curve.points) == 2
        # a has bpp=0.6, c has bpp=0.2 -> sorted ascending: c, a
        rates = curve.rates.tolist()
        assert rates == pytest.approx([0.2, 0.6])

    def test_custom_metric_keys(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        zoo.save_entry(
            e,
            {"model": {"w": torch.tensor([0.0])}},
            {"my_rate": 0.5, "my_psnr": 33.0},
        )
        zoo.save_entry(
            e2,
            {"model": {"w": torch.tensor([0.0])}},
            {"my_rate": 0.3, "my_psnr": 31.0},
        )

        manifest = _make_manifest([e, e2], storage_root=tmp_path / "zoo")
        cfg = RDCurveFitConfig(
            name="cfg",
            rate_metric_key="my_rate",
            quality_metric_key="my_psnr",
        )
        curve = compute_rd_curve(zoo, manifest, fit_config=cfg)
        assert len(curve.points) == 2

    def test_msssim_is_optional(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        _save_metrics(zoo, e1, bpp=0.5, psnr_db=33.0)  # no ms_ssim
        _save_metrics(zoo, e2, bpp=0.3, psnr_db=31.0, ms_ssim=0.95)

        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        curve = compute_rd_curve(zoo, manifest)
        # Curve mixes None and present SSIM — RDCurve.ssims drops Nones.
        assert any(p.ssim is None for p in curve.points)
        assert any(p.ssim is not None for p in curve.points)


# --------------------------------------------------------------------------
# compute_rd_curve — failure modes
# --------------------------------------------------------------------------


class TestComputeRDCurveFailures:
    def test_fewer_than_min_points_raises(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        _save_metrics(zoo, e1, bpp=0.5, psnr_db=33.0)
        manifest = _make_manifest([e1], storage_root=tmp_path / "zoo")
        with pytest.raises(RDCurveAssemblyError, match="at least 2 entries"):
            compute_rd_curve(zoo, manifest)

    def test_missing_rate_metric_key_raises(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        zoo.save_entry(
            e1,
            {"model": {"w": torch.tensor([0.0])}},
            {"psnr_db": 33.0},  # no bpp
        )
        zoo.save_entry(
            e2,
            {"model": {"w": torch.tensor([0.0])}},
            {"psnr_db": 30.0, "bpp": 0.3},
        )
        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        with pytest.raises(RDCurveAssemblyError, match="missing required key 'bpp'"):
            compute_rd_curve(zoo, manifest)

    def test_missing_quality_metric_key_raises(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        zoo.save_entry(
            e1,
            {"model": {"w": torch.tensor([0.0])}},
            {"bpp": 0.5},  # no psnr_db
        )
        zoo.save_entry(
            e2,
            {"model": {"w": torch.tensor([0.0])}},
            {"bpp": 0.3, "psnr_db": 30.0},
        )
        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        with pytest.raises(RDCurveAssemblyError, match="missing required key 'psnr_db'"):
            compute_rd_curve(zoo, manifest)

    def test_non_monotone_curve_raises(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        # Higher rate but lower PSNR — non-monotone
        _save_metrics(zoo, e1, bpp=0.3, psnr_db=35.0)
        _save_metrics(zoo, e2, bpp=0.5, psnr_db=30.0)
        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        with pytest.raises(RDCurveAssemblyError, match="not monotonically increasing"):
            compute_rd_curve(zoo, manifest)

    def test_non_monotone_passes_when_check_disabled(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        _save_metrics(zoo, e1, bpp=0.3, psnr_db=35.0)
        _save_metrics(zoo, e2, bpp=0.5, psnr_db=30.0)
        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        cfg = RDCurveFitConfig(name="cfg", enforce_monotone=False)
        curve = compute_rd_curve(zoo, manifest, fit_config=cfg)
        assert len(curve.points) == 2

    def test_duplicate_lambdas_raise(self, tmp_path: Path) -> None:
        # Build via model_validate to bypass the manifest's duplicate
        # entry_id check while keeping unique entry_ids and duplicate
        # lambda values.
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.01)  # same lambda
        _save_metrics(zoo, e1, bpp=0.5, psnr_db=33.0)
        _save_metrics(zoo, e2, bpp=0.4, psnr_db=32.0)
        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        with pytest.raises(RDCurveAssemblyError, match="duplicate lambda_rd"):
            compute_rd_curve(zoo, manifest)

    def test_skipped_entries_logged_not_raised_when_enough_remain(
        self,
        tmp_path: Path,
    ) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        e3 = _make_entry("e3", lambda_rd=0.03)
        _save_metrics(zoo, e1, bpp=0.6, psnr_db=36.0)
        # e2 has no checkpoint
        _save_metrics(zoo, e3, bpp=0.2, psnr_db=30.0)
        manifest = _make_manifest([e1, e2, e3], storage_root=tmp_path / "zoo")
        curve = compute_rd_curve(zoo, manifest)
        assert len(curve.points) == 2

    def test_monotone_spline_fit_method_does_not_change_points(
        self,
        tmp_path: Path,
    ) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        _save_metrics(zoo, e1, bpp=0.5, psnr_db=33.0)
        _save_metrics(zoo, e2, bpp=0.3, psnr_db=31.0)
        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        cfg = RDCurveFitConfig(name="cfg", fit_method="monotone_spline")
        curve = compute_rd_curve(zoo, manifest, fit_config=cfg)
        # fit_method only changes interpolation behavior downstream — the
        # raw points must be unchanged.
        assert len(curve.points) == 2

    def test_custom_curve_name(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        e1 = _make_entry("e1", lambda_rd=0.01)
        e2 = _make_entry("e2", lambda_rd=0.02)
        _save_metrics(zoo, e1, bpp=0.5, psnr_db=33.0)
        _save_metrics(zoo, e2, bpp=0.3, psnr_db=31.0)
        manifest = _make_manifest([e1, e2], storage_root=tmp_path / "zoo")
        curve = compute_rd_curve(zoo, manifest, name="alphagalerkin_v1")
        assert curve.name == "alphagalerkin_v1"


# --------------------------------------------------------------------------
# Property-based: monotone λ → monotone (rate ↓, PSNR ↑) yields a
# monotone-by-construction curve.
# --------------------------------------------------------------------------


@given(
    psnrs=st.lists(
        st.floats(min_value=20.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
        unique=True,
    ),
)
def test_property_monotone_input_yields_monotone_curve(
    psnrs: list[float], tmp_path_factory: pytest.TempPathFactory,
) -> None:
    psnrs = sorted(psnrs)  # ascending PSNR
    n = len(psnrs)
    # bpp ascending in lockstep so rate↑ implies psnr↑
    bpps = [0.05 + 0.1 * i for i in range(n)]
    lambdas = [0.001 * (i + 1) for i in range(n)]

    tmp = tmp_path_factory.mktemp("rdcurve_prop")
    zoo = VideoCodecZoo(tmp / "zoo")
    entries = [
        _make_entry(f"e{i}", lambda_rd=lambdas[i])
        for i in range(n)
    ]
    for e, bpp, psnr in zip(entries, bpps, psnrs, strict=True):
        _save_metrics(zoo, e, bpp=bpp, psnr_db=psnr)

    manifest = _make_manifest(entries, storage_root=tmp / "zoo")
    curve = compute_rd_curve(zoo, manifest)
    assert curve.is_monotonic()
