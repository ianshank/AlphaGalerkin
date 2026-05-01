"""Tests for the ``onnx-cuda`` decoder runtime.

Covers:
- Protocol compliance (``DecoderRuntime``)
- Registration in ``RuntimeRegistry``
- Full ``prepare`` / ``decode`` / ``teardown`` lifecycle (CPU only)
- Decode-before-prepare guard
- Shape and hash mismatch rejection
- Latent channels mismatch
- Graceful skip when onnxruntime is not installed
- Metadata and provenance tags
- Artifact save path
"""

from __future__ import annotations

import pytest
import torch

from src.video_compression.config import CodecConfig
from src.video_compression.runtime.metadata import CompiledArtifactMetadata
from src.video_compression.runtime.onnx_runtime import (
    ONNX_RUNTIME_NAME,
    ONNXDecoderRuntime,
    _onnx_available,
)
from src.video_compression.runtime.protocol import (
    DecoderRuntime,
    DecoderRuntimeContext,
)
from src.video_compression.runtime.registry import (
    RuntimeRegistry,
    create_runtime,
)

_SKIP_NO_ONNX = pytest.mark.skipif(
    not _onnx_available(),
    reason="onnxruntime not installed",
)


class TestONNXRuntimeRegistration:
    """Verify the runtime is discoverable in the registry."""

    def test_registered_name(self) -> None:
        assert RuntimeRegistry().is_registered(ONNX_RUNTIME_NAME)

    def test_create_runtime_returns_instance(self) -> None:
        rt = create_runtime(ONNX_RUNTIME_NAME)
        assert isinstance(rt, ONNXDecoderRuntime)

    def test_create_runtime_with_codec_config(
        self,
        tiny_codec_config,
    ) -> None:
        rt = create_runtime(
            ONNX_RUNTIME_NAME,
            codec_config=tiny_codec_config,
        )
        assert isinstance(rt, ONNXDecoderRuntime)


class TestONNXRuntimeProtocol:
    """Structural compliance with the DecoderRuntime Protocol."""

    def test_implements_protocol(self) -> None:
        rt = ONNXDecoderRuntime()
        assert isinstance(rt, DecoderRuntime)

    def test_name_property(self) -> None:
        rt = ONNXDecoderRuntime()
        assert rt.name == ONNX_RUNTIME_NAME

    def test_metadata_none_before_prepare(self) -> None:
        rt = ONNXDecoderRuntime()
        assert rt.metadata is None


class TestONNXRuntimeLifecycle:
    """Full lifecycle tests (requires onnxruntime)."""

    def _make_context(
        self,
        codec_config: CodecConfig,
        *,
        batch_size: int = 1,
        height: int = 16,
        width: int = 16,
        dtype: str = "float32",
    ) -> DecoderRuntimeContext:
        downsample = codec_config.encoder.downsample_factor
        return DecoderRuntimeContext(
            name="test_ctx",
            batch_size=batch_size,
            latent_channels=codec_config.encoder.latent_channels,
            latent_height=height // downsample,
            latent_width=width // downsample,
            dtype=dtype,
            device="cpu",
            model_hash=codec_config.compute_hash(),
        )

    @_SKIP_NO_ONNX
    def test_prepare_sets_metadata(self, tiny_codec_config) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        assert isinstance(rt.metadata, CompiledArtifactMetadata)
        assert rt.metadata.runtime_name == ONNX_RUNTIME_NAME
        assert rt.metadata.backend == "onnx"
        assert rt.metadata.artifact_size_bytes is not None
        assert rt.metadata.artifact_size_bytes > 0
        rt.teardown()

    @_SKIP_NO_ONNX
    def test_decode_produces_valid_output(
        self,
        tiny_codec_config,
    ) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        latent = torch.randn(
            ctx.batch_size,
            ctx.latent_channels,
            ctx.latent_height,
            ctx.latent_width,
        )
        output = rt.decode(latent)
        assert output.shape[0] == ctx.batch_size
        assert output.shape[1] == 3  # RGB
        rt.teardown()

    def test_decode_before_prepare_raises(self) -> None:
        rt = ONNXDecoderRuntime()
        with pytest.raises(RuntimeError, match="before prepare"):
            rt.decode(torch.randn(1, 32, 4, 4))

    @_SKIP_NO_ONNX
    def test_teardown_clears_state(self, tiny_codec_config) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        rt.teardown()
        assert rt.metadata is None
        assert rt._session is None

    @_SKIP_NO_ONNX
    def test_reprepare_for_new_shape(
        self,
        tiny_codec_config,
    ) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        ctx1 = self._make_context(tiny_codec_config, height=16, width=16)
        rt.prepare(ctx=ctx1)
        rt.teardown()

        ctx2 = self._make_context(tiny_codec_config, height=32, width=32)
        rt.prepare(ctx=ctx2)
        latent = torch.randn(
            ctx2.batch_size,
            ctx2.latent_channels,
            ctx2.latent_height,
            ctx2.latent_width,
        )
        output = rt.decode(latent)
        assert output.shape[0] == ctx2.batch_size
        rt.teardown()


class TestONNXRuntimeValidation:
    """Error paths."""

    def _make_context(
        self,
        codec_config: CodecConfig,
        **overrides,
    ) -> DecoderRuntimeContext:
        downsample = codec_config.encoder.downsample_factor
        defaults = {
            "name": "test_ctx",
            "batch_size": 1,
            "latent_channels": codec_config.encoder.latent_channels,
            "latent_height": 16 // downsample,
            "latent_width": 16 // downsample,
            "dtype": "float32",
            "device": "cpu",
            "model_hash": codec_config.compute_hash(),
        }
        defaults.update(overrides)
        return DecoderRuntimeContext(**defaults)

    @_SKIP_NO_ONNX
    def test_latent_channels_mismatch_raises(
        self,
        tiny_codec_config,
    ) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(
            tiny_codec_config,
            latent_channels=999,
        )
        with pytest.raises(ValueError, match="latent_channels"):
            rt.prepare(ctx=ctx)

    @_SKIP_NO_ONNX
    def test_model_hash_mismatch_raises(
        self,
        tiny_codec_config,
    ) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(
            tiny_codec_config,
            model_hash="wrong_hash",
        )
        with pytest.raises(ValueError, match="model_hash"):
            rt.prepare(ctx=ctx)

    @_SKIP_NO_ONNX
    def test_shape_mismatch_on_decode_raises(
        self,
        tiny_codec_config,
    ) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        wrong_shape = torch.randn(1, ctx.latent_channels, 99, 99)
        with pytest.raises(ValueError, match="latent shape"):
            rt.decode(wrong_shape)
        rt.teardown()


class TestONNXRuntimeArtifactSave:
    """Test ONNX model saving to disk."""

    @_SKIP_NO_ONNX
    def test_saves_onnx_to_artifact_path(
        self,
        tiny_codec_config,
        tmp_path,
    ) -> None:
        rt = ONNXDecoderRuntime(codec_config=tiny_codec_config)
        onnx_path = tmp_path / "decoder.onnx"
        downsample = tiny_codec_config.encoder.downsample_factor
        ctx = DecoderRuntimeContext(
            name="test_ctx",
            batch_size=1,
            latent_channels=tiny_codec_config.encoder.latent_channels,
            latent_height=16 // downsample,
            latent_width=16 // downsample,
            dtype="float32",
            device="cpu",
            model_hash=tiny_codec_config.compute_hash(),
            artifact_path=str(onnx_path),
        )
        rt.prepare(ctx=ctx)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0
        rt.teardown()


class TestONNXRuntimeMetadataTags:
    """Verify metadata extra_tags capture ONNX settings."""

    @_SKIP_NO_ONNX
    def test_extra_tags_contain_opset(
        self,
        tiny_codec_config,
    ) -> None:
        rt = ONNXDecoderRuntime(
            codec_config=tiny_codec_config,
            opset_version=17,
        )
        downsample = tiny_codec_config.encoder.downsample_factor
        ctx = DecoderRuntimeContext(
            name="test_ctx",
            batch_size=1,
            latent_channels=tiny_codec_config.encoder.latent_channels,
            latent_height=16 // downsample,
            latent_width=16 // downsample,
            dtype="float32",
            device="cpu",
            model_hash=tiny_codec_config.compute_hash(),
        )
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        assert rt.metadata.extra_tags["opset_version"] == "17"
        assert "providers" in rt.metadata.extra_tags
        assert rt.metadata.build_time_s >= 0.0
        rt.teardown()
