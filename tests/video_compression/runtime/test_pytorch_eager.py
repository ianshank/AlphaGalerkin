"""Tests for the PyTorch eager runtime — the equivalence baseline."""

from __future__ import annotations

import pytest
import torch

from src.video_compression.runtime import (
    DecoderRuntime,
    DecoderRuntimeContext,
    PyTorchEagerRuntime,
)


class TestProtocolConformance:
    def test_satisfies_protocol(self) -> None:
        rt = PyTorchEagerRuntime()
        assert isinstance(rt, DecoderRuntime)


class TestLifecycleCPU:
    """All CPU-runnable lifecycle checks."""

    def test_metadata_none_before_prepare(self) -> None:
        # Returns None (not raises) so isinstance(rt, DecoderRuntime)
        # remains usable outside the prepare/teardown window.
        rt = PyTorchEagerRuntime()
        assert rt.metadata is None

    def test_decode_before_prepare_raises(self, tiny_codec_config) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        with pytest.raises(RuntimeError, match="before prepare"):
            rt.decode(torch.zeros(1, 32, 4, 4))

    def test_prepare_populates_metadata(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        meta = rt.metadata
        assert meta is not None
        assert meta.runtime_name == "pytorch-eager"
        assert meta.backend == "pytorch"
        assert meta.precision == "float32"
        assert meta.batch_size == 1
        assert meta.build_time_s == 0.0
        assert meta.artifact_path is None

    def test_metadata_records_device_label(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        meta = rt.metadata
        assert meta is not None
        # CPU label is the bare type
        assert meta.device_label == "cpu"

    def test_decode_returns_correct_shape(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        latent = torch.zeros(
            tiny_context.batch_size,
            tiny_context.latent_channels,
            tiny_context.latent_height,
            tiny_context.latent_width,
        )
        out = rt.decode(latent)
        # decoder upsamples by upsample_factor (4 in tiny config)
        upsample = tiny_codec_config.decoder.upsample_factor
        assert out.shape == (
            tiny_context.batch_size,
            tiny_codec_config.decoder.out_channels,
            tiny_context.latent_height * upsample,
            tiny_context.latent_width * upsample,
        )

    def test_decode_no_grad_path(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        latent = torch.zeros(
            tiny_context.batch_size,
            tiny_context.latent_channels,
            tiny_context.latent_height,
            tiny_context.latent_width,
            requires_grad=True,
        )
        out = rt.decode(latent)
        # Eager runtime wraps in torch.no_grad; output must be detached.
        assert not out.requires_grad

    def test_teardown_then_re_prepare(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        rt.teardown()
        # After teardown metadata is None (not raises) — see Protocol
        # docstring on why we chose None over RuntimeError.
        assert rt.metadata is None
        # Re-prepare with a different shape works.
        new_ctx = tiny_context.with_overrides(latent_height=8, latent_width=8)
        rt.prepare(ctx=new_ctx)
        meta = rt.metadata
        assert meta is not None
        assert meta.latent_height == 8

    def test_teardown_without_prepare_is_safe(self) -> None:
        rt = PyTorchEagerRuntime()
        rt.teardown()  # must not raise

    def test_latent_channels_mismatch_rejected(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        bad_ctx = tiny_context.with_overrides(
            latent_channels=tiny_codec_config.encoder.latent_channels + 1,
        )
        with pytest.raises(ValueError, match="latent_channels"):
            rt.prepare(ctx=bad_ctx)

    def test_decode_device_mismatch_rejected(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        # tiny_context.device is "cpu". Build a meta-device tensor —
        # always available, regardless of CUDA presence — that's
        # guaranteed not to be on cpu.
        meta_latent = torch.zeros(
            tiny_context.batch_size,
            tiny_context.latent_channels,
            tiny_context.latent_height,
            tiny_context.latent_width,
            device="meta",
        )
        with pytest.raises(ValueError, match="runtime prepared"):
            rt.decode(meta_latent)


class TestDefaultCodecConfig:
    def test_default_codec_config_works(self) -> None:
        # Constructor with no arg uses the default codec config; this
        # is the path the registry's create_runtime() takes.
        rt = PyTorchEagerRuntime()
        assert rt.name == "pytorch-eager"


class TestPrepareValidation:
    """Prepare-time validation regression tests.

    Added in response to PR #76 review feedback: ``prepare`` must
    reject mismatched dtype and codec hash before allocating any GPU
    state.
    """

    def test_dtype_other_than_fp32_rejected(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        bad_ctx = tiny_context.with_overrides(dtype="float16")
        with pytest.raises(ValueError, match="float32"):
            rt.prepare(ctx=bad_ctx)

    def test_model_hash_mismatch_rejected(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        bad_ctx = tiny_context.with_overrides(model_hash="not-the-real-hash")
        with pytest.raises(ValueError, match="model_hash"):
            rt.prepare(ctx=bad_ctx)


@pytest.mark.gpu_required
class TestCudaDeviceNormalization:
    """Regression tests for index-less ``cuda`` device normalization.

    Caller passes ``device='cuda'`` (no index); the resulting decode
    call must succeed without a false device-mismatch — fixed in the
    review pass on PR #76.
    """

    def test_unindexed_cuda_normalized_to_indexed(
        self,
        tiny_codec_config,
    ) -> None:
        ctx = DecoderRuntimeContext(
            name="cuda_unindexed",
            batch_size=1,
            latent_channels=32,
            latent_height=4,
            latent_width=4,
            dtype="float32",
            device="cuda",  # No index
            model_hash=tiny_codec_config.compute_hash(),
        )
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=ctx)
        meta = rt.metadata
        assert meta is not None
        # Normalization makes the recorded device label include the
        # concrete index (e.g. 'cuda:0:NVIDIA-...')
        assert meta.device_label.startswith("cuda:")
        assert ":" in meta.device_label[len("cuda:") :]

        # And the decode path works without a false mismatch — this
        # is the original bug Devin/Copilot flagged.
        latent = torch.zeros(
            ctx.batch_size,
            ctx.latent_channels,
            ctx.latent_height,
            ctx.latent_width,
            device="cuda",
        )
        out = rt.decode(latent)
        assert out.device.type == "cuda"


class TestDecodeShapeValidation:
    """Regression tests for the decode-time shape validation."""

    def test_wrong_shape_rejected(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        # Right channel count, wrong spatial dims -> reject
        bad_latent = torch.zeros(
            tiny_context.batch_size,
            tiny_context.latent_channels,
            tiny_context.latent_height + 1,
            tiny_context.latent_width,
        )
        with pytest.raises(ValueError, match="latent shape"):
            rt.decode(bad_latent)

    def test_wrong_batch_rejected(
        self,
        tiny_codec_config,
        tiny_context: DecoderRuntimeContext,
    ) -> None:
        rt = PyTorchEagerRuntime(codec_config=tiny_codec_config)
        rt.prepare(ctx=tiny_context)
        bad_latent = torch.zeros(
            tiny_context.batch_size + 1,
            tiny_context.latent_channels,
            tiny_context.latent_height,
            tiny_context.latent_width,
        )
        with pytest.raises(ValueError, match="latent shape"):
            rt.decode(bad_latent)
