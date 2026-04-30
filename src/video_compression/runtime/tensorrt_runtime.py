"""TensorRT decoder runtime — Phase 1 backend.

Uses ``torch_tensorrt`` to compile the decoder into a TensorRT engine
for maximum inference throughput on NVIDIA GPUs. This is the highest
performance backend but requires CUDA, an NVIDIA GPU, and the
``torch_tensorrt`` package.

Key design choices:

* Compilation via ``torch_tensorrt.compile`` uses the Dynamo frontend
  (``ir="dynamo"``) which integrates with ``torch.compile`` and
  supports FP16 natively through ``enabled_precisions``.
* The engine is shape-locked: re-calling ``prepare`` with different
  shapes rebuilds the TensorRT engine.
* FP16 and FP32 are supported via ``enabled_precisions``. BF16 is not
  natively supported by TensorRT on most consumer GPUs, so we map
  BF16 requests to FP16 with a warning.
* If ``torch_tensorrt`` is not installed, the module imports cleanly
  but ``prepare()`` raises a clear error.
* On CPU contexts, the runtime raises immediately — TensorRT is
  CUDA-only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import structlog
import torch

from src.video_compression.codec.codec import VideoCodec
from src.video_compression.config import CodecConfig
from src.video_compression.perf.device import device_label, resolve_device
from src.video_compression.runtime.metadata import CompiledArtifactMetadata
from src.video_compression.runtime.protocol import DecoderRuntimeContext
from src.video_compression.runtime.registry import (
    BaseDecoderRuntime,
    register_runtime,
)

logger = structlog.get_logger(__name__)

# Stable runtime name at module scope.
TENSORRT_RUNTIME_NAME = "tensorrt"

# Supported optimization levels for the TRT engine build.
TRTOptimizationLevel = Literal[1, 2, 3, 4, 5]

# Default optimization level (3 = balanced build time vs performance).
DEFAULT_OPTIMIZATION_LEVEL: TRTOptimizationLevel = 3


def _tensorrt_available() -> bool:
    """Check if torch_tensorrt is importable."""
    try:
        import torch_tensorrt  # noqa: F401  # type: ignore[import-untyped]

        return True
    except ImportError:
        return False


@register_runtime(TENSORRT_RUNTIME_NAME)
class TensorRTRuntime(BaseDecoderRuntime):
    """TensorRT-backed decode path.

    Compiles the decoder using ``torch_tensorrt.compile`` with the
    Dynamo frontend. The compiled engine is cached for the lifetime
    of the ``prepare`` window; ``teardown`` invalidates it.

    Supports FP32 and FP16 precision. BF16 requests are mapped to
    FP16 with a warning since TensorRT does not natively support BF16
    on most consumer hardware.
    """

    runtime_name = TENSORRT_RUNTIME_NAME

    def __init__(
        self,
        codec_config: CodecConfig | None = None,
        *,
        optimization_level: TRTOptimizationLevel = DEFAULT_OPTIMIZATION_LEVEL,
    ) -> None:
        super().__init__()
        self._codec_config = codec_config or CodecConfig(
            name="runtime_tensorrt_default",
        )
        self._optimization_level = optimization_level
        self._compiled_decoder: Callable[..., Any] | None = None
        self._device: torch.device | None = None

    # ----------------------------------------------------------- lifecycle

    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
        if not _tensorrt_available():
            raise RuntimeError(
                f"{self.name} requires torch_tensorrt to be installed. "
                f"Install with: pip install torch-tensorrt "
                f"--extra-index-url https://download.pytorch.org/whl/cu126",
            )

        device = resolve_device(ctx.device, context=f"runtime.{self.name}")
        if device.type != "cuda":
            raise ValueError(
                f"{self.name} requires a CUDA device, got {device.type!r}. "
                f"TensorRT only supports NVIDIA GPUs.",
            )
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # Validate latent channels.
        if ctx.latent_channels != self._codec_config.encoder.latent_channels:
            raise ValueError(
                f"context latent_channels={ctx.latent_channels} does not "
                f"match codec config latent_channels="
                f"{self._codec_config.encoder.latent_channels}",
            )

        # Validate model hash.
        expected_hash = self._codec_config.compute_hash()
        if ctx.model_hash != expected_hash:
            raise ValueError(
                f"context model_hash={ctx.model_hash!r} does not match "
                f"this runtime's codec config hash {expected_hash!r}; "
                f"either pass codec_config=... when constructing the "
                f"runtime, or pass the matching codec hash in the "
                f"context.",
            )

        # Determine TensorRT precision.
        enabled_precisions: set[torch.dtype] = {torch.float32}
        precision_label = ctx.dtype
        if ctx.dtype in ("float16", "bfloat16"):
            enabled_precisions.add(torch.float16)
            if ctx.dtype == "bfloat16":
                logger.warning(
                    "runtime.tensorrt.bf16_mapped_to_fp16",
                    runtime=self.name,
                    msg="BF16 not natively supported by TensorRT; "
                    "using FP16 instead",
                )
                precision_label = "float16"

        # Build codec and extract decoder.
        codec = VideoCodec(
            config=self._codec_config,
            use_mcts_rate_control=False,
            device=str(device),
        ).to(device)
        codec.eval()

        # Create example input for tracing.
        example_input = torch.randn(
            ctx.batch_size,
            ctx.latent_channels,
            ctx.latent_height,
            ctx.latent_width,
            device=device,
        )

        # Compile with TensorRT via torch_tensorrt Dynamo frontend.
        import torch_tensorrt

        t_start = time.perf_counter()
        try:
            compiled = torch_tensorrt.compile(
                codec.decoder,
                ir="dynamo",
                inputs=[example_input],
                enabled_precisions=enabled_precisions,
                optimization_level=self._optimization_level,
                truncate_double=True,
            )
        except Exception as exc:
            # If dynamo IR fails, try torch_compile fallback.
            logger.warning(
                "runtime.tensorrt.dynamo_failed",
                runtime=self.name,
                error=str(exc),
                msg="Retrying with ir='torch_compile'",
            )
            compiled = torch_tensorrt.compile(
                codec.decoder,
                ir="torch_compile",
                inputs=[example_input],
                enabled_precisions=enabled_precisions,
            )
        build_time = time.perf_counter() - t_start

        self._compiled_decoder = compiled
        self._device = device
        self._prepared_ctx = ctx
        self._metadata = CompiledArtifactMetadata(
            name="tensorrt_runtime_metadata",
            runtime_name=self.name,
            backend="tensorrt",
            precision=precision_label,
            model_hash=ctx.model_hash,
            device_label=device_label(device),
            batch_size=ctx.batch_size,
            latent_channels=ctx.latent_channels,
            latent_height=ctx.latent_height,
            latent_width=ctx.latent_width,
            build_time_s=build_time,
            artifact_path=None,
            artifact_size_bytes=None,
            extra_tags={
                "optimization_level": str(self._optimization_level),
                "enabled_precisions": str(
                    sorted(str(p) for p in enabled_precisions)
                ),
                "ir": "dynamo",
            },
        )

        logger.debug(
            "runtime.tensorrt.prepared",
            runtime=self.name,
            optimization_level=self._optimization_level,
            device=str(device),
            build_time_s=f"{build_time:.3f}",
            precisions=sorted(str(p) for p in enabled_precisions),
            latent_shape=(
                ctx.batch_size,
                ctx.latent_channels,
                ctx.latent_height,
                ctx.latent_width,
            ),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self._compiled_decoder is None or self._prepared_ctx is None:
            raise RuntimeError(
                f"{self.name}.decode() called before prepare()",
            )

        # Shape validation.
        expected_shape = (
            self._prepared_ctx.batch_size,
            self._prepared_ctx.latent_channels,
            self._prepared_ctx.latent_height,
            self._prepared_ctx.latent_width,
        )
        actual_shape = tuple(latent.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"latent shape {actual_shape} does not match prepared "
                f"context {expected_shape}; caller must re-run "
                f"prepare() for a different latent shape",
            )

        with torch.no_grad():
            return self._compiled_decoder(latent)

    def teardown(self) -> None:
        device = self._device
        self._compiled_decoder = None
        self._device = None
        super().teardown()
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
