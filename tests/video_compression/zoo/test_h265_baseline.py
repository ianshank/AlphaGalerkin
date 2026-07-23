"""Tests for :mod:`src.video_compression.zoo.h265_baseline`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video_compression.zoo.h265_baseline import (
    H265_BASELINE_SCHEMA_VERSION,
    H265BaselineDocument,
    H265BaselineEntry,
    H265BaselineMigrationError,
    H265BaselineRegistry,
)


def _make_entry(
    *,
    cell_key: str,
    sequence_id: str = "akiyo",
    bpp: float,
    psnr: float | None,
    crf: int = 28,
    codec: str = "libx265",
    width: int = 352,
    height: int = 288,
    fps: float = 30.0,
    ms_ssim: float | None = None,
) -> H265BaselineEntry:
    return H265BaselineEntry(
        name=cell_key,
        cell_key=cell_key,
        sequence_id=sequence_id,
        codec=codec,
        crf=crf,
        width=width,
        height=height,
        fps=fps,
        bpp=bpp,
        psnr_db=psnr,
        ms_ssim=ms_ssim,
    )


def _make_document(
    entries: list[H265BaselineEntry],
    *,
    name: str = "test_doc",
) -> H265BaselineDocument:
    return H265BaselineDocument(name=name, entries=entries)


# --------------------------------------------------------------------------
# Save / load round-trip
# --------------------------------------------------------------------------


class TestRegistryRoundTrip:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        entries = [
            _make_entry(cell_key=f"akiyo|cif|30|libx265|crf{crf}", bpp=bpp, psnr=psnr, crf=crf)
            for crf, bpp, psnr in [(22, 0.6, 38.5), (28, 0.3, 35.0), (35, 0.1, 31.0)]
        ]
        doc = _make_document(entries)
        registry = H265BaselineRegistry(doc)

        path = tmp_path / "h265_baseline.json"
        registry.save(path)
        assert path.exists()

        loaded = H265BaselineRegistry.load(path)
        assert len(loaded.entries) == 3
        assert loaded.get("akiyo|cif|30|libx265|crf22") is not None
        assert loaded.document.schema_version == H265_BASELINE_SCHEMA_VERSION

    def test_load_unversioned_document_promotes_to_v1(self, tmp_path: Path) -> None:
        # Simulate a hand-written baseline file with no schema_version.
        payload = {
            "name": "akiyo_baseline",
            "entries": [
                {
                    "name": "akiyo|cif|30|libx265|crf28",
                    "cell_key": "akiyo|cif|30|libx265|crf28",
                    "sequence_id": "akiyo",
                    "codec": "libx265",
                    "crf": 28,
                    "width": 352,
                    "height": 288,
                    "fps": 30.0,
                    "bpp": 0.3,
                    "psnr_db": 35.0,
                },
            ],
        }
        path = tmp_path / "unv.json"
        path.write_text(json.dumps(payload))
        registry = H265BaselineRegistry.load(path)
        assert registry.document.schema_version == 1

    def test_load_newer_schema_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "future.json"
        path.write_text(
            json.dumps(
                {
                    "name": "future",
                    "schema_version": H265_BASELINE_SCHEMA_VERSION + 1,
                    "entries": [],
                }
            )
        )
        with pytest.raises(H265BaselineMigrationError, match="newer than this binary"):
            H265BaselineRegistry.load(path)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            H265BaselineRegistry.load(tmp_path / "does_not_exist.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            H265BaselineRegistry.load(path)

    def test_load_non_object_root_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(H265BaselineMigrationError, match="must be a JSON object"):
            H265BaselineRegistry.load(path)


# --------------------------------------------------------------------------
# Forward-compat
# --------------------------------------------------------------------------


class TestForwardCompat:
    def test_unknown_field_at_document_level_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "fwd.json"
        path.write_text(
            json.dumps(
                {
                    "name": "fwd",
                    "schema_version": 1,
                    "future_doc_field_v2": "hello",
                    "entries": [],
                }
            )
        )
        registry = H265BaselineRegistry.load(path)
        assert registry.document.schema_version == 1

    def test_unknown_field_at_entry_level_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "fwd.json"
        path.write_text(
            json.dumps(
                {
                    "name": "fwd",
                    "schema_version": 1,
                    "entries": [
                        {
                            "name": "e",
                            "schema_version": 1,
                            "cell_key": "akiyo|cif|30|libx265|crf28",
                            "sequence_id": "akiyo",
                            "codec": "libx265",
                            "crf": 28,
                            "width": 352,
                            "height": 288,
                            "fps": 30.0,
                            "bpp": 0.3,
                            "psnr_db": 35.0,
                            "future_entry_field_v2": 1.5,
                        },
                    ],
                }
            )
        )
        registry = H265BaselineRegistry.load(path)
        assert len(registry.entries) == 1


# --------------------------------------------------------------------------
# Filter + to_curve
# --------------------------------------------------------------------------


class TestFilterAndCurve:
    def test_filter_by_sequence_and_codec(self) -> None:
        entries = [
            _make_entry(cell_key="akiyo|cif|30|libx265|crf28", bpp=0.3, psnr=35.0),
            _make_entry(
                cell_key="akiyo|cif|30|libaom-av1|crf28", bpp=0.25, psnr=35.5, codec="libaom-av1"
            ),
            _make_entry(
                cell_key="foreman|cif|30|libx265|crf28", bpp=0.4, psnr=33.0, sequence_id="foreman"
            ),
        ]
        registry = H265BaselineRegistry(_make_document(entries))
        akiyo_x265 = registry.filter(sequence_id="akiyo", codec="libx265")
        assert len(akiyo_x265) == 1
        assert akiyo_x265[0].codec == "libx265"

    def test_filter_by_resolution(self) -> None:
        entries = [
            _make_entry(
                cell_key="a|cif|30|libx265|crf28", bpp=0.3, psnr=35.0, width=352, height=288
            ),
            _make_entry(
                cell_key="a|hd|30|libx265|crf28", bpp=0.5, psnr=33.0, width=1920, height=1080
            ),
        ]
        registry = H265BaselineRegistry(_make_document(entries))
        cif = registry.filter(sequence_id="akiyo", width=352, height=288)
        assert len(cif) == 1

    def test_to_curve_happy_path(self) -> None:
        entries = [
            _make_entry(cell_key=f"a|cif|30|libx265|crf{crf}", bpp=bpp, psnr=psnr, crf=crf)
            for crf, bpp, psnr in [(22, 0.6, 38.5), (28, 0.3, 35.0), (35, 0.1, 31.0)]
        ]
        registry = H265BaselineRegistry(_make_document(entries))
        curve = registry.to_curve(sequence_id="akiyo")
        assert curve.name == "libx265_akiyo"
        assert len(curve.points) == 3
        # add_point sorts by rate ascending
        assert curve.rates.tolist() == pytest.approx([0.1, 0.3, 0.6])

    def test_to_curve_custom_name(self) -> None:
        entries = [
            _make_entry(cell_key=f"a|cif|30|libx265|crf{crf}", bpp=bpp, psnr=psnr, crf=crf)
            for crf, bpp, psnr in [(22, 0.6, 38.5), (35, 0.1, 31.0)]
        ]
        registry = H265BaselineRegistry(_make_document(entries))
        curve = registry.to_curve(sequence_id="akiyo", name="ref_v1")
        assert curve.name == "ref_v1"

    def test_to_curve_too_few_psnr_raises(self) -> None:
        entries = [
            _make_entry(cell_key="a|cif|30|libx265|crf28", bpp=0.3, psnr=35.0),
            _make_entry(cell_key="a|cif|30|libx265|crf35", bpp=0.1, psnr=None),
        ]
        registry = H265BaselineRegistry(_make_document(entries))
        with pytest.raises(ValueError, match=">=2 baseline entries with PSNR"):
            registry.to_curve(sequence_id="akiyo")

    def test_to_curve_drops_null_psnr_entries_but_keeps_valid_ones(self) -> None:
        # Mixed nullability: of 4 baseline entries, only the 2 with
        # non-null psnr_db must end up on the curve. Forward-compat for
        # future entries that ship with VMAF-only / SSIM-only fields.
        entries = [
            _make_entry(cell_key="a|cif|30|libx265|crf22", bpp=0.6, psnr=38.5, crf=22),
            _make_entry(cell_key="a|cif|30|libx265|crf28", bpp=0.3, psnr=None, crf=28),
            _make_entry(cell_key="a|cif|30|libx265|crf32", bpp=0.2, psnr=33.0, crf=32),
            _make_entry(cell_key="a|cif|30|libx265|crf38", bpp=0.1, psnr=None, crf=38),
        ]
        registry = H265BaselineRegistry(_make_document(entries))
        curve = registry.to_curve(sequence_id="akiyo")
        assert len(curve.points) == 2
        assert curve.psnrs.tolist() == pytest.approx([33.0, 38.5])

    def test_to_curve_other_sequence_excluded(self) -> None:
        entries = [
            _make_entry(cell_key=f"akiyo|cif|30|libx265|crf{crf}", bpp=bpp, psnr=psnr, crf=crf)
            for crf, bpp, psnr in [(22, 0.6, 38.5), (28, 0.3, 35.0)]
        ]
        entries.append(
            _make_entry(
                cell_key="foreman|cif|30|libx265|crf28", bpp=0.4, psnr=33.0, sequence_id="foreman"
            ),
        )
        registry = H265BaselineRegistry(_make_document(entries))
        curve = registry.to_curve(sequence_id="akiyo")
        assert len(curve.points) == 2

    def test_to_curve_passes_through_msssim(self) -> None:
        entries = [
            _make_entry(
                cell_key=f"a|cif|30|libx265|crf{crf}", bpp=bpp, psnr=psnr, crf=crf, ms_ssim=ms
            )
            for crf, bpp, psnr, ms in [
                (22, 0.6, 38.5, 0.98),
                (28, 0.3, 35.0, 0.95),
            ]
        ]
        registry = H265BaselineRegistry(_make_document(entries))
        curve = registry.to_curve(sequence_id="akiyo")
        assert all(p.ssim is not None for p in curve.points)


class TestEntryValidation:
    def test_negative_bpp_rejected(self) -> None:
        with pytest.raises(Exception):  # ValidationError
            _make_entry(cell_key="x", bpp=-0.1, psnr=30.0)

    def test_invalid_crf_range_rejected(self) -> None:
        with pytest.raises(Exception):
            _make_entry(cell_key="x", bpp=0.3, psnr=30.0, crf=100)
