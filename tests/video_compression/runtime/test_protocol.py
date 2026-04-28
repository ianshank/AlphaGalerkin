"""Tests for the DecoderRuntime Protocol and per-cell context schema."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from src.video_compression.runtime import DecoderRuntime, DecoderRuntimeContext

# ----------------------------------------------------- DecoderRuntimeContext


class TestDecoderRuntimeContext:
    def test_minimal_construction(self) -> None:
        ctx = DecoderRuntimeContext(
            name="ctx",
            batch_size=1,
            latent_channels=32,
            latent_height=4,
            latent_width=4,
            dtype="float32",
            device="cpu",
            model_hash="abc",
        )
        assert ctx.batch_size == 1
        assert ctx.latent_channels == 32
        assert ctx.dtype == "float32"
        assert ctx.artifact_path is None

    def test_dtype_torch_resolution(self) -> None:
        for dtype_str, expected in [
            ("float32", torch.float32),
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
        ]:
            ctx = DecoderRuntimeContext(
                name="c",
                batch_size=1,
                latent_channels=32,
                latent_height=4,
                latent_width=4,
                dtype=dtype_str,
                device="cpu",
                model_hash="h",
            )
            assert ctx.torch_dtype() is expected

    def test_unknown_dtype_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in supported set"):
            DecoderRuntimeContext(
                name="c",
                batch_size=1,
                latent_channels=32,
                latent_height=4,
                latent_width=4,
                dtype="float64",
                device="cpu",
                model_hash="h",
            )

    @pytest.mark.parametrize(
        "field, bad_value",
        [
            ("batch_size", 0),
            ("batch_size", 5000),
            ("latent_channels", 0),
            ("latent_height", 0),
            ("latent_width", 0),
        ],
    )
    def test_field_bounds(self, field: str, bad_value: int) -> None:
        kwargs: dict[str, object] = {
            "name": "c",
            "batch_size": 1,
            "latent_channels": 32,
            "latent_height": 4,
            "latent_width": 4,
            "dtype": "float32",
            "device": "cpu",
            "model_hash": "h",
        }
        kwargs[field] = bad_value
        with pytest.raises(ValidationError):
            DecoderRuntimeContext(**kwargs)  # type: ignore[arg-type]

    def test_extra_fields_forbidden_on_transient_context(self) -> None:
        # Transient (not persisted) so extras must be programmer errors.
        with pytest.raises(ValidationError):
            DecoderRuntimeContext(
                name="c",
                batch_size=1,
                latent_channels=32,
                latent_height=4,
                latent_width=4,
                dtype="float32",
                device="cpu",
                model_hash="h",
                future_field="should reject",  # type: ignore[call-arg]
            )

    def test_model_hash_required_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            DecoderRuntimeContext(
                name="c",
                batch_size=1,
                latent_channels=32,
                latent_height=4,
                latent_width=4,
                dtype="float32",
                device="cpu",
                model_hash="",
            )


# ------------------------------------------------------- DecoderRuntime Protocol


class _FakeRuntime:
    """Minimal duck-typed implementation used purely for Protocol tests.

    Lives in the test module so the Protocol's runtime_checkable
    semantics are validated independently of the bundled
    BaseDecoderRuntime / PyTorchEagerRuntime.
    """

    @property
    def name(self) -> str:
        return "fake"

    @property
    def metadata(self):  # type: ignore[no-untyped-def]
        return None

    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
        pass

    def decode(self, latent):  # type: ignore[no-untyped-def]
        return latent

    def teardown(self) -> None:
        pass


class TestDecoderRuntimeProtocol:
    def test_duck_typed_implementation_satisfies(self) -> None:
        rt = _FakeRuntime()
        assert isinstance(rt, DecoderRuntime)

    def test_missing_method_fails_protocol_check(self) -> None:
        class Incomplete:
            @property
            def name(self) -> str:
                return "x"

            # decode/prepare/teardown intentionally missing

        assert not isinstance(Incomplete(), DecoderRuntime)
