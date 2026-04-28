"""PyTorch eager decoder runtime — the equivalence baseline.

This is the source of truth that every other runtime is measured
against. It does the minimal thing: instantiate ``VideoCodec``,
move it to the requested device, and forward through ``codec.decoder``
on each ``decode()`` call. No entropy coding, no quantization round-
trip — those add variance that obscures the runtime's own throughput.

Key design choices:

* ``CodecConfig`` is the single piece of input not in
  ``DecoderRuntimeContext``. We keep it as a constructor arg so the
  benchmark passes the same codec config it builds the synthetic
  data against. Backwards-compatible: ``codec_config=None`` builds
  the default config.
* The decoder is built fresh in each ``prepare`` call. ``VideoCodec``
  is small relative to a 1080p tensor; rebuilding is cheaper than
  reasoning about lingering activations across cells. This matches
  ``CodecForwardSubject.prepare``.
* Per-cell metadata records the exact device label (including GPU
  model name where applicable) using
  ``src/video_compression/perf/device.py::device_label``.
"""

from __future__ import annotations

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

# Stable runtime name. Lives at module scope (not as a literal in the
# class body) so other modules can reference it without a circular
# import.
PYTORCH_EAGER_RUNTIME_NAME = "pytorch-eager"


@register_runtime(PYTORCH_EAGER_RUNTIME_NAME)
class PyTorchEagerRuntime(BaseDecoderRuntime):
    """Eager PyTorch decode path.

    Wraps ``VideoCodec.decoder`` and runs the forward pass under
    ``torch.no_grad``. Calling ``decode(latent)`` is equivalent to
    ``codec.decoder(latent)`` with the standard PyTorch overhead.
    """

    runtime_name = PYTORCH_EAGER_RUNTIME_NAME

    def __init__(self, codec_config: CodecConfig | None = None) -> None:
        super().__init__()
        # Decoupled so a caller can pin a specific model config (e.g.
        # the one matching a checkpoint they trained) while keeping
        # the same runtime class. None -> default.
        self._codec_config = codec_config or CodecConfig(name="runtime_eager_default")
        self._codec: VideoCodec | None = None
        self._device: torch.device | None = None

    # ----------------------------------------------------------- lifecycle

    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
        device = resolve_device(ctx.device, context=f"runtime.{self.name}")

        # Validate latent dims against codec's expectations early so we
        # fail with a useful message before allocating tensors.
        if ctx.latent_channels != self._codec_config.encoder.latent_channels:
            raise ValueError(
                f"context latent_channels={ctx.latent_channels} does not "
                f"match codec config latent_channels="
                f"{self._codec_config.encoder.latent_channels}",
            )

        codec = VideoCodec(
            config=self._codec_config,
            use_mcts_rate_control=False,
            device=str(device),
        ).to(device)
        codec.eval()

        self._codec = codec
        self._device = device
        self._prepared_ctx = ctx
        self._metadata = CompiledArtifactMetadata(
            name="pytorch_eager_metadata",
            runtime_name=self.name,
            backend="pytorch",
            precision=ctx.dtype,
            model_hash=ctx.model_hash,
            device_label=device_label(device),
            batch_size=ctx.batch_size,
            latent_channels=ctx.latent_channels,
            latent_height=ctx.latent_height,
            latent_width=ctx.latent_width,
            build_time_s=0.0,
            artifact_path=None,
            artifact_size_bytes=None,
            extra_tags={"requires_grad": "False"},
        )

        logger.debug(
            "runtime.eager.prepared",
            runtime=self.name,
            device=str(device),
            latent_shape=(
                ctx.batch_size,
                ctx.latent_channels,
                ctx.latent_height,
                ctx.latent_width,
            ),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self._codec is None or self._prepared_ctx is None:
            raise RuntimeError(f"{self.name}.decode() called before prepare()")
        if latent.device != self._device:
            raise ValueError(
                f"latent on {latent.device}, runtime prepared on "
                f"{self._device}; caller must move tensor before decode()",
            )
        with torch.no_grad():
            return self._codec.decoder(latent)

    def teardown(self) -> None:
        # Free GPU memory before clearing the reference.
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._codec = None
        self._device = None
        super().teardown()
