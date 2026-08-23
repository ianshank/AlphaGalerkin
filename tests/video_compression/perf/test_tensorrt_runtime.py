"""Tests for the ``tensorrt`` decoder runtime.

Covers:
- Protocol compliance (``DecoderRuntime``)
- Registration in ``RuntimeRegistry``
- Full ``prepare`` / ``decode`` / ``teardown`` lifecycle (CUDA)
- Decode-before-prepare guard
- Shape and hash mismatch rejection
- CPU device rejection
- BF16-to-FP16 mapping
- Graceful skip when torch_tensorrt is not installed
- Metadata and provenance tags
"""

from __future__ import annotations

import pytest
import torch

from src.video_compression.config import CodecConfig
from src.video_compression.runtime.metadata import CompiledArtifactMetadata
from src.video_compression.runtime.protocol import (
    DecoderRuntime,
    DecoderRuntimeContext,
)
from src.video_compression.runtime.registry import (
    RuntimeRegistry,
    create_runtime,
)
from src.video_compression.runtime.tensorrt_runtime import (
    TENSORRT_RUNTIME_NAME,
    TensorRTRuntime,
    _tensorrt_available,
)

_SKIP_NO_TRT = pytest.mark.skipif(
    not _tensorrt_available(),
    reason="torch_tensorrt not installed",
)

_SKIP_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


class TestTensorRTRegistration:
    """Verify the runtime is discoverable in the registry."""

    def test_registered_name(self) -> None:
        assert RuntimeRegistry().is_registered(TENSORRT_RUNTIME_NAME)

    def test_create_runtime_returns_instance(self) -> None:
        rt = create_runtime(TENSORRT_RUNTIME_NAME)
        assert isinstance(rt, TensorRTRuntime)

    def test_create_runtime_with_codec_config(
        self,
        tiny_codec_config,
    ) -> None:
        rt = create_runtime(
            TENSORRT_RUNTIME_NAME,
            codec_config=tiny_codec_config,
        )
        assert isinstance(rt, TensorRTRuntime)


class TestTensorRTProtocol:
    """Structural compliance with the DecoderRuntime Protocol."""

    def test_implements_protocol(self) -> None:
        rt = TensorRTRuntime()
        assert isinstance(rt, DecoderRuntime)

    def test_name_property(self) -> None:
        rt = TensorRTRuntime()
        assert rt.name == TENSORRT_RUNTIME_NAME

    def test_metadata_none_before_prepare(self) -> None:
        rt = TensorRTRuntime()
        assert rt.metadata is None


class TestTensorRTLifecycle:
    """Full lifecycle tests (requires CUDA + torch_tensorrt)."""

    def _make_context(
        self,
        codec_config: CodecConfig,
        *,
        batch_size: int = 1,
        height: int = 16,
        width: int = 16,
        dtype: str = "float32",
        device: str = "cuda:0",
    ) -> DecoderRuntimeContext:
        downsample = codec_config.encoder.downsample_factor
        return DecoderRuntimeContext(
            name="test_ctx",
            batch_size=batch_size,
            latent_channels=codec_config.encoder.latent_channels,
            latent_height=height // downsample,
            latent_width=width // downsample,
            dtype=dtype,
            device=device,
            model_hash=codec_config.compute_hash(),
        )

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_prepare_sets_metadata(self, tiny_codec_config) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        assert isinstance(rt.metadata, CompiledArtifactMetadata)
        assert rt.metadata.runtime_name == TENSORRT_RUNTIME_NAME
        assert rt.metadata.backend == "tensorrt"
        assert rt.metadata.build_time_s > 0.0
        rt.teardown()

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_decode_produces_valid_output(
        self,
        tiny_codec_config,
    ) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        latent = torch.randn(
            ctx.batch_size,
            ctx.latent_channels,
            ctx.latent_height,
            ctx.latent_width,
            device="cuda:0",
        )
        output = rt.decode(latent)
        assert output.shape[0] == ctx.batch_size
        assert output.shape[1] == 3  # RGB
        rt.teardown()

    def test_decode_before_prepare_raises(self) -> None:
        rt = TensorRTRuntime()
        with pytest.raises(RuntimeError, match="before prepare"):
            rt.decode(torch.randn(1, 32, 4, 4))

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_teardown_clears_state(self, tiny_codec_config) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        rt.teardown()
        assert rt.metadata is None
        assert rt._compiled_decoder is None

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_reprepare_for_new_shape(
        self,
        tiny_codec_config,
    ) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
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
            device="cuda:0",
        )
        output = rt.decode(latent)
        assert output.shape[0] == ctx2.batch_size
        rt.teardown()


class TestTensorRTValidation:
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
            "device": "cuda:0",
            "model_hash": codec_config.compute_hash(),
        }
        defaults.update(overrides)
        return DecoderRuntimeContext(**defaults)

    @_SKIP_NO_TRT
    def test_cpu_device_raises(self, tiny_codec_config) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config, device="cpu")
        with pytest.raises(ValueError, match="CUDA device"):
            rt.prepare(ctx=ctx)

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_latent_channels_mismatch_raises(
        self,
        tiny_codec_config,
    ) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(
            tiny_codec_config,
            latent_channels=999,
        )
        with pytest.raises(ValueError, match="latent_channels"):
            rt.prepare(ctx=ctx)

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_model_hash_mismatch_raises(
        self,
        tiny_codec_config,
    ) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(
            tiny_codec_config,
            model_hash="wrong_hash",
        )
        with pytest.raises(ValueError, match="model_hash"):
            rt.prepare(ctx=ctx)

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_shape_mismatch_on_decode_raises(
        self,
        tiny_codec_config,
    ) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        wrong_shape = torch.randn(
            1,
            ctx.latent_channels,
            99,
            99,
            device="cuda:0",
        )
        with pytest.raises(ValueError, match="latent shape"):
            rt.decode(wrong_shape)
        rt.teardown()


class TestTensorRTBF16Mapping:
    """Verify BF16 requests are mapped to FP16."""

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_bf16_mapped_to_fp16(self, tiny_codec_config) -> None:
        rt = TensorRTRuntime(codec_config=tiny_codec_config)
        downsample = tiny_codec_config.encoder.downsample_factor
        ctx = DecoderRuntimeContext(
            name="test_ctx",
            batch_size=1,
            latent_channels=tiny_codec_config.encoder.latent_channels,
            latent_height=16 // downsample,
            latent_width=16 // downsample,
            dtype="bfloat16",
            device="cuda:0",
            model_hash=tiny_codec_config.compute_hash(),
        )
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        # BF16 should be mapped to FP16 in metadata.
        assert rt.metadata.precision == "float16"

        # Decode must succeed with an FP16 latent — the engine input dtype
        # must match the precision metadata, otherwise we would have built
        # an FP32 engine but advertised it as FP16.
        latent_fp16 = torch.randn(
            ctx.batch_size,
            ctx.latent_channels,
            ctx.latent_height,
            ctx.latent_width,
            device="cuda:0",
            dtype=torch.float16,
        )
        output = rt.decode(latent_fp16)
        assert output.shape[0] == ctx.batch_size
        assert output.shape[1] == 3  # RGB

        # And FP32 input must be rejected — silent autocasting would hide
        # miswired benchmark profiles.
        latent_fp32 = latent_fp16.to(dtype=torch.float32)
        with pytest.raises(ValueError, match="dtype"):
            rt.decode(latent_fp32)
        rt.teardown()


class TestTensorRTMetadataTags:
    """Verify metadata extra_tags capture TensorRT settings."""

    @_SKIP_NO_TRT
    @_SKIP_NO_CUDA
    def test_extra_tags_contain_optimization_level(
        self,
        tiny_codec_config,
    ) -> None:
        rt = TensorRTRuntime(
            codec_config=tiny_codec_config,
            optimization_level=3,
        )
        downsample = tiny_codec_config.encoder.downsample_factor
        ctx = DecoderRuntimeContext(
            name="test_ctx",
            batch_size=1,
            latent_channels=tiny_codec_config.encoder.latent_channels,
            latent_height=16 // downsample,
            latent_width=16 // downsample,
            dtype="float32",
            device="cuda:0",
            model_hash=tiny_codec_config.compute_hash(),
        )
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        assert rt.metadata.extra_tags["optimization_level"] == "3"
        assert "enabled_precisions" in rt.metadata.extra_tags
        assert rt.metadata.build_time_s >= 0.0
        rt.teardown()
