"""``torch.compile`` decoder runtime — Phase 1 backend.

Wraps the synthesis transform (decoder) in ``torch.compile`` to
produce a fused graph that can be significantly faster than eager
execution. This is the lowest-friction backend upgrade: same model
weights, same device, no external dependencies beyond PyTorch 2.x.

Key design choices:

* The ``compile_mode`` field controls the ``torch.compile`` mode.
  ``"reduce-overhead"`` uses CUDA graphs and is the recommended
  setting for fixed-shape inference workloads (like our benchmark).
  ``"max-autotune"`` runs more autotuning trials and is suitable
  for headline numbers. ``"default"`` is a balanced fallback.
* Compilation happens inside ``prepare()`` during warmup — the first
  invocation is slow, subsequent ones are fast. The benchmark's
  warmup phase absorbs this cost.
* ``fullgraph=True`` is the default because the decoder is a
  static graph (no data-dependent control flow). If compilation
  fails, the runtime falls back to ``fullgraph=False`` with a
  warning.
* The compiled decoder is shape-locked to the prepared context.
  Re-calling ``prepare`` with different shapes triggers a
  recompile (same as ONNX / TensorRT runtimes).
* AMP (``torch.autocast``) is layered on when ``ctx.dtype`` is
  ``"float16"`` or ``"bfloat16"``; the compile graph itself runs
  under autocast so the kernel fusion includes the cast ops.
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
PYTORCH_COMPILED_RUNTIME_NAME = "pytorch-compiled"

# Supported compile modes — mirrors ``RuntimeProfile.compile_mode``.
CompileMode = Literal["default", "reduce-overhead", "max-autotune"]

# Default compile mode for the compiled runtime.
DEFAULT_COMPILE_MODE: CompileMode = "reduce-overhead"

# Whether to attempt ``fullgraph=True`` by default.
DEFAULT_FULLGRAPH: bool = True


@register_runtime(PYTORCH_COMPILED_RUNTIME_NAME)
class PyTorchCompiledRuntime(BaseDecoderRuntime):
    """``torch.compile``-backed decode path.

    Compiles ``VideoCodec.decoder`` with ``torch.compile`` using the
    configured mode and precision. The compiled graph is cached for the
    lifetime of the ``prepare`` window; ``teardown`` invalidates it.

    Supports FP32, FP16, and BF16 via ``torch.autocast``. The autocast
    context is entered in ``decode()`` so the compiled graph includes
    the precision-cast kernels.
    """

    runtime_name = PYTORCH_COMPILED_RUNTIME_NAME

    def __init__(
        self,
        codec_config: CodecConfig | None = None,
        *,
        compile_mode: CompileMode = DEFAULT_COMPILE_MODE,
        fullgraph: bool = DEFAULT_FULLGRAPH,
    ) -> None:
        super().__init__()
        self._codec_config = codec_config or CodecConfig(
            name="runtime_compiled_default",
        )
        self._compile_mode = compile_mode
        self._fullgraph = fullgraph
        self._compiled_decoder: Callable[..., Any] | None = None
        self._device: torch.device | None = None
        self._autocast_dtype: torch.dtype | None = None

    # ----------------------------------------------------------- lifecycle

    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
        device = resolve_device(ctx.device, context=f"runtime.{self.name}")
        if device.type == "cuda" and device.index is None:
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

        # Determine autocast dtype.
        self._autocast_dtype = None
        if ctx.dtype == "float16":
            self._autocast_dtype = torch.float16
        elif ctx.dtype == "bfloat16":
            self._autocast_dtype = torch.bfloat16
        # fp32 = no autocast needed

        # Build codec and extract decoder.
        codec = VideoCodec(
            config=self._codec_config,
            use_mcts_rate_control=False,
            device=str(device),
        ).to(device)
        codec.eval()

        t_start = time.perf_counter()
        actual_fullgraph = self._fullgraph
        try:
            compiled = torch.compile(
                codec.decoder,
                mode=self._compile_mode,
                fullgraph=actual_fullgraph,
            )
        except Exception:
            # Fallback: retry without fullgraph.
            actual_fullgraph = False
            logger.warning(
                "runtime.compiled.fullgraph_failed",
                runtime=self.name,
                compile_mode=self._compile_mode,
                msg="Retrying with fullgraph=False",
            )
            compiled = torch.compile(
                codec.decoder,
                mode=self._compile_mode,
                fullgraph=actual_fullgraph,
            )
        build_time = time.perf_counter() - t_start

        self._compiled_decoder = compiled
        self._device = device
        self._prepared_ctx = ctx
        self._metadata = CompiledArtifactMetadata(
            name="pytorch_compiled_metadata",
            runtime_name=self.name,
            backend="compiled",
            precision=ctx.dtype,
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
                "compile_mode": self._compile_mode,
                "fullgraph": str(actual_fullgraph),
                "autocast_dtype": ctx.dtype,
            },
        )

        logger.debug(
            "runtime.compiled.prepared",
            runtime=self.name,
            compile_mode=self._compile_mode,
            device=str(device),
            build_time_s=f"{build_time:.3f}",
            autocast_dtype=ctx.dtype,
            latent_shape=(
                ctx.batch_size,
                ctx.latent_channels,
                ctx.latent_height,
                ctx.latent_width,
            ),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self._compiled_decoder is None or self._prepared_ctx is None:
            raise RuntimeError(f"{self.name}.decode() called before prepare()")
        if latent.device != self._device:
            raise ValueError(
                f"latent on {latent.device}, runtime prepared on "
                f"{self._device}; caller must move tensor before decode()",
            )
        # Shape validation — same as eager for consistency.
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
        if self._autocast_dtype is not None:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self._device.type if self._device else "cpu",
                    dtype=self._autocast_dtype,
                ),
            ):
                return self._compiled_decoder(latent)
        with torch.no_grad():
            return self._compiled_decoder(latent)

    def teardown(self) -> None:
        device = self._device
        self._compiled_decoder = None
        self._device = None
        self._autocast_dtype = None
        super().teardown()
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
