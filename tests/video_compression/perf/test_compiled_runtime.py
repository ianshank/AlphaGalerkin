"""Tests for the ``pytorch-compiled`` decoder runtime.

Covers:
- Protocol compliance (``DecoderRuntime`` + ``BenchmarkSubject`` via wrapper)
- Full ``prepare`` / ``decode`` / ``teardown`` lifecycle
- Decode-before-prepare guard
- Shape and hash mismatch rejection
- FP16 / BF16 autocast paths
- Fullgraph fallback on compilation failure
- Re-prepare across different shapes
- Registration in ``RuntimeRegistry``
- Factory lookup via ``create_runtime``
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
import torch

from src.video_compression.config import CodecConfig
from src.video_compression.runtime.metadata import CompiledArtifactMetadata


def _inductor_compiler_available() -> bool:
    """Check if the inductor backend's C++ compiler is reachable."""
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler

        compiler = get_cpp_compiler()
        result = subprocess.run(  # noqa: S603
            [compiler, "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


_SKIP_NO_INDUCTOR = pytest.mark.skipif(
    not _inductor_compiler_available(),
    reason="torch.compile inductor backend requires a C++ compiler (cl/gcc/clang)",
)

from src.video_compression.runtime.protocol import (
    DecoderRuntime,
    DecoderRuntimeContext,
)
from src.video_compression.runtime.pytorch_compiled import (
    DEFAULT_COMPILE_MODE,
    DEFAULT_FULLGRAPH,
    PYTORCH_COMPILED_RUNTIME_NAME,
    PyTorchCompiledRuntime,
)
from src.video_compression.runtime.registry import (
    RuntimeRegistry,
    create_runtime,
)


class TestPyTorchCompiledRuntimeRegistration:
    """Verify the runtime is discoverable in the registry."""

    def test_registered_name(self) -> None:
        assert RuntimeRegistry().is_registered(PYTORCH_COMPILED_RUNTIME_NAME)

    def test_create_runtime_returns_instance(self) -> None:
        rt = create_runtime(PYTORCH_COMPILED_RUNTIME_NAME)
        assert isinstance(rt, PyTorchCompiledRuntime)

    def test_create_runtime_with_codec_config(self, tiny_codec_config) -> None:
        rt = create_runtime(
            PYTORCH_COMPILED_RUNTIME_NAME,
            codec_config=tiny_codec_config,
        )
        assert isinstance(rt, PyTorchCompiledRuntime)
        assert rt._codec_config is tiny_codec_config


class TestPyTorchCompiledRuntimeProtocol:
    """Verify structural compliance with the ``DecoderRuntime`` Protocol."""

    def test_implements_decoder_runtime_protocol(self) -> None:
        rt = PyTorchCompiledRuntime()
        assert isinstance(rt, DecoderRuntime)

    def test_name_property(self) -> None:
        rt = PyTorchCompiledRuntime()
        assert rt.name == PYTORCH_COMPILED_RUNTIME_NAME

    def test_metadata_none_before_prepare(self) -> None:
        rt = PyTorchCompiledRuntime()
        assert rt.metadata is None

    def test_default_config_values(self) -> None:
        rt = PyTorchCompiledRuntime()
        assert rt._compile_mode == DEFAULT_COMPILE_MODE
        assert rt._fullgraph == DEFAULT_FULLGRAPH


class TestPyTorchCompiledRuntimeLifecycle:
    """Full prepare/decode/teardown lifecycle on CPU."""

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

    def test_prepare_sets_metadata(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        assert isinstance(rt.metadata, CompiledArtifactMetadata)
        assert rt.metadata.runtime_name == PYTORCH_COMPILED_RUNTIME_NAME
        assert rt.metadata.backend == "compiled"
        assert rt.metadata.precision == "float32"
        rt.teardown()

    @_SKIP_NO_INDUCTOR
    def test_decode_produces_valid_output(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
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
        rt = PyTorchCompiledRuntime()
        with pytest.raises(RuntimeError, match="before prepare"):
            rt.decode(torch.randn(1, 32, 4, 4))

    def test_teardown_clears_state(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        rt.teardown()
        assert rt.metadata is None
        assert rt._compiled_decoder is None

    @_SKIP_NO_INDUCTOR
    def test_reprepare_for_new_shape(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        # First cell: 16x16
        ctx1 = self._make_context(tiny_codec_config, height=16, width=16)
        rt.prepare(ctx=ctx1)
        latent1 = torch.randn(
            ctx1.batch_size,
            ctx1.latent_channels,
            ctx1.latent_height,
            ctx1.latent_width,
        )
        rt.decode(latent1)
        rt.teardown()

        # Second cell: 32x32
        ctx2 = self._make_context(tiny_codec_config, height=32, width=32)
        rt.prepare(ctx=ctx2)
        latent2 = torch.randn(
            ctx2.batch_size,
            ctx2.latent_channels,
            ctx2.latent_height,
            ctx2.latent_width,
        )
        output2 = rt.decode(latent2)
        assert output2.shape[0] == ctx2.batch_size
        rt.teardown()


class TestPyTorchCompiledRuntimeValidation:
    """Error paths: hash mismatch, shape mismatch, latent channel mismatch."""

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

    def test_latent_channels_mismatch_raises(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(
            tiny_codec_config,
            latent_channels=999,
        )
        with pytest.raises(ValueError, match="latent_channels"):
            rt.prepare(ctx=ctx)

    def test_model_hash_mismatch_raises(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(
            tiny_codec_config,
            model_hash="wrong_hash",
        )
        with pytest.raises(ValueError, match="model_hash"):
            rt.prepare(ctx=ctx)

    def test_shape_mismatch_on_decode_raises(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(tiny_codec_config)
        rt.prepare(ctx=ctx)
        wrong_shape = torch.randn(1, ctx.latent_channels, 99, 99)
        with pytest.raises(ValueError, match="latent shape"):
            rt.decode(wrong_shape)
        rt.teardown()


class TestPyTorchCompiledRuntimeAutocast:
    """Verify FP16 and BF16 autocast paths."""

    def _make_context(
        self,
        codec_config: CodecConfig,
        dtype: str = "float32",
    ) -> DecoderRuntimeContext:
        downsample = codec_config.encoder.downsample_factor
        return DecoderRuntimeContext(
            name="test_ctx",
            batch_size=1,
            latent_channels=codec_config.encoder.latent_channels,
            latent_height=16 // downsample,
            latent_width=16 // downsample,
            dtype=dtype,
            device="cpu",
            model_hash=codec_config.compute_hash(),
        )

    @_SKIP_NO_INDUCTOR
    def test_fp16_accepted(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(tiny_codec_config, dtype="float16")
        rt.prepare(ctx=ctx)
        assert rt._autocast_dtype == torch.float16
        assert rt.metadata is not None
        assert rt.metadata.extra_tags["autocast_dtype"] == "float16"
        # Decode should work under autocast.
        latent = torch.randn(
            ctx.batch_size,
            ctx.latent_channels,
            ctx.latent_height,
            ctx.latent_width,
        )
        output = rt.decode(latent)
        assert output.shape[0] == 1
        rt.teardown()

    def test_bf16_accepted(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(tiny_codec_config, dtype="bfloat16")
        rt.prepare(ctx=ctx)
        assert rt._autocast_dtype == torch.bfloat16
        assert rt.metadata is not None
        assert rt.metadata.extra_tags["autocast_dtype"] == "bfloat16"
        rt.teardown()

    def test_fp32_no_autocast(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
        )
        ctx = self._make_context(tiny_codec_config, dtype="float32")
        rt.prepare(ctx=ctx)
        assert rt._autocast_dtype is None
        rt.teardown()


class TestPyTorchCompiledRuntimeFullgraphFallback:
    """Verify graceful fallback when fullgraph=True compilation fails."""

    def test_fullgraph_fallback(self, tiny_codec_config) -> None:
        """When fullgraph=True fails, the runtime retries with False."""
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="default",
            fullgraph=True,
        )
        ctx = DecoderRuntimeContext(
            name="test_ctx",
            batch_size=1,
            latent_channels=tiny_codec_config.encoder.latent_channels,
            latent_height=4,
            latent_width=4,
            dtype="float32",
            device="cpu",
            model_hash=tiny_codec_config.compute_hash(),
        )

        # Patch torch.compile to fail on first call (fullgraph=True),
        # succeed on second call (fullgraph=False).
        call_count = [0]
        original_compile = torch.compile

        def mock_compile(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and kwargs.get("fullgraph", False):
                raise RuntimeError("Simulated fullgraph failure")
            return original_compile(*args, **kwargs)

        compile_target = "src.video_compression.runtime.pytorch_compiled.torch.compile"
        with patch(compile_target, side_effect=mock_compile):
            rt.prepare(ctx=ctx)
        # Should have been called twice: once with fullgraph=True (fail),
        # once with fullgraph=False (succeed).
        assert call_count[0] == 2
        assert rt.metadata is not None
        rt.teardown()


class TestPyTorchCompiledRuntimeMetadataTags:
    """Verify that metadata extra_tags capture compile settings."""

    def test_extra_tags_contain_compile_mode(self, tiny_codec_config) -> None:
        rt = PyTorchCompiledRuntime(
            codec_config=tiny_codec_config,
            compile_mode="max-autotune",
        )
        ctx = DecoderRuntimeContext(
            name="test_ctx",
            batch_size=1,
            latent_channels=tiny_codec_config.encoder.latent_channels,
            latent_height=4,
            latent_width=4,
            dtype="float32",
            device="cpu",
            model_hash=tiny_codec_config.compute_hash(),
        )
        rt.prepare(ctx=ctx)
        assert rt.metadata is not None
        assert rt.metadata.extra_tags["compile_mode"] == "max-autotune"
        # `prepare()` may fall back to ``fullgraph=False`` if the initial
        # ``fullgraph=True`` compile raises (e.g. because of a graph break in
        # a particular PyTorch / inductor version). Either is a valid outcome;
        # the metadata must record whichever path was actually taken.
        assert rt.metadata.extra_tags["fullgraph"] in ("True", "False")
        assert rt.metadata.build_time_s >= 0.0
        rt.teardown()
