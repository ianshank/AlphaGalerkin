"""ONNX Runtime decoder backend — Phase 1.

Exports the decoder to ONNX format, then runs inference via
``onnxruntime.InferenceSession``. This backend enables deployment
on any platform that supports ONNX Runtime (CPU, CUDA, TensorRT EP,
DirectML, OpenVINO, etc.).

Key design choices:

* The ONNX export happens inside ``prepare()`` using ``torch.onnx.export``
  with a dynamic batch axis. The export is a one-time
  cost absorbed by the benchmark warmup.
* The ONNX model is kept in memory (not saved to disk) unless an
  ``artifact_path`` is provided in the ``DecoderRuntimeContext``.
* The execution provider is selected based on the device in the context:
  ``CUDAExecutionProvider`` for CUDA, ``CPUExecutionProvider`` for CPU.
* FP16 is supported by casting the model inputs; the ONNX graph itself
  runs in the precision of the original model weights.
* If ``onnxruntime`` is not installed, the module imports cleanly but
  ``prepare()`` raises a clear error.
"""

from __future__ import annotations

import io
import time
from typing import Any

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
ONNX_RUNTIME_NAME = "onnx-cuda"

# Default ONNX opset version. Opset 17 is well-supported by
# onnxruntime >= 1.14 and includes all ops used by our decoder.
DEFAULT_OPSET_VERSION: int = 17


def _onnx_available() -> bool:
    """Check if onnxruntime and onnxscript are importable."""
    try:
        import onnxruntime  # noqa: F401
        import onnxscript  # noqa: F401  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False


@register_runtime(ONNX_RUNTIME_NAME)
class ONNXDecoderRuntime(BaseDecoderRuntime):
    """ONNX Runtime-backed decode path.

    Exports the decoder to ONNX, then runs inference via
    ``onnxruntime.InferenceSession``. Supports both CPU and CUDA
    execution providers.
    """

    runtime_name = ONNX_RUNTIME_NAME

    def __init__(
        self,
        codec_config: CodecConfig | None = None,
        *,
        opset_version: int = DEFAULT_OPSET_VERSION,
    ) -> None:
        super().__init__()
        self._codec_config = codec_config or CodecConfig(
            name="runtime_onnx_default",
        )
        self._opset_version = opset_version
        self._session: Any | None = None
        self._input_name: str | None = None
        self._device: torch.device | None = None
        self._output_device: torch.device | None = None

    # ----------------------------------------------------------- lifecycle

    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
        if not _onnx_available():
            raise RuntimeError(
                f"{self.name} requires onnxruntime to be installed. "
                f"Install with: pip install onnxruntime-gpu (CUDA) "
                f"or pip install onnxruntime (CPU).",
            )
        import onnxruntime as ort

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

        if ctx.dtype != "float32":
            raise ValueError(
                f"{self.name} only supports torch.float32 decode inputs for "
                f"the exported ONNX graph, but prepare() was called with "
                f"dtype={ctx.dtype!r}. Re-run prepare() with "
                f"torch.float32 or export a graph that matches the requested "
                f"dtype.",
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

        # Build codec and extract decoder for export.
        codec = VideoCodec(
            config=self._codec_config,
            use_mcts_rate_control=False,
            device="cpu",  # Export on CPU, then load with target EP
        )
        codec.eval()

        # Export to ONNX in-memory.
        t_start = time.perf_counter()
        dummy_input = torch.randn(
            ctx.batch_size,
            ctx.latent_channels,
            ctx.latent_height,
            ctx.latent_width,
        )

        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            codec.decoder,
            (dummy_input,),
            onnx_buffer,  # type: ignore[arg-type]
            opset_version=self._opset_version,
            input_names=["latent"],
            output_names=["output"],
            dynamic_axes={
                "latent": {0: "batch"},
                "output": {0: "batch"},
            },
        )
        onnx_bytes = onnx_buffer.getvalue()

        # Save to disk if artifact_path was provided.
        if ctx.artifact_path:
            from pathlib import Path

            Path(ctx.artifact_path).parent.mkdir(parents=True, exist_ok=True)
            Path(ctx.artifact_path).write_bytes(onnx_bytes)
            logger.debug(
                "runtime.onnx.saved",
                path=ctx.artifact_path,
                size_bytes=len(onnx_bytes),
            )

        # Select execution provider.
        if device.type == "cuda":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"device_id": device.index or 0},
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        # Create inference session.
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = ort.InferenceSession(
            onnx_bytes,
            sess_options=sess_options,
            providers=providers,
        )
        build_time = time.perf_counter() - t_start

        self._input_name = self._session.get_inputs()[0].name
        self._device = device
        self._output_device = device
        self._prepared_ctx = ctx
        self._metadata = CompiledArtifactMetadata(
            name="onnx_runtime_metadata",
            runtime_name=self.name,
            backend="onnx",
            precision=ctx.dtype,
            model_hash=ctx.model_hash,
            device_label=device_label(device),
            batch_size=ctx.batch_size,
            latent_channels=ctx.latent_channels,
            latent_height=ctx.latent_height,
            latent_width=ctx.latent_width,
            build_time_s=build_time,
            artifact_path=ctx.artifact_path,
            artifact_size_bytes=len(onnx_bytes),
            extra_tags={
                "opset_version": str(self._opset_version),
                "providers": str(
                    self._session.get_providers()
                ),
            },
        )

        logger.debug(
            "runtime.onnx.prepared",
            runtime=self.name,
            opset_version=self._opset_version,
            device=str(device),
            build_time_s=f"{build_time:.3f}",
            onnx_size_bytes=len(onnx_bytes),
            providers=self._session.get_providers(),
            latent_shape=(
                ctx.batch_size,
                ctx.latent_channels,
                ctx.latent_height,
                ctx.latent_width,
            ),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self._session is None or self._prepared_ctx is None:
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

        # Convert to numpy for ONNX Runtime.
        #
        # This backend currently exports the ONNX graph with a float32 dummy
        # input, so the session input type is float32. Rejecting non-float32
        # runtime contexts explicitly ensures we don't silently override the
        # requested dtype here.
        import numpy as np
        latent_np = latent.detach().cpu().numpy().astype(np.float32, copy=False)

        # Run inference.
        outputs = self._session.run(
            None, {self._input_name: latent_np},
        )

        # Convert back to torch tensor on the target device.
        result = torch.from_numpy(outputs[0])
        if self._output_device is not None:
            result = result.to(self._output_device)
        return result

    def teardown(self) -> None:
        device = self._device
        self._session = None
        self._input_name = None
        self._device = None
        self._output_device = None
        super().teardown()
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
