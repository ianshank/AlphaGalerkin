"""Benchmark subjects.

A "subject" is the thing being timed. The benchmark loop calls
``prepare(...)`` once per cell to set up state for a given (resolution,
batch_size) and then calls ``step()`` exactly once per measured iteration.

This Protocol-driven design lets later phases (1: torch.compile / ONNX /
TensorRT runtimes) plug new subjects in without touching the benchmark
loop. Phase 1 wires ``BenchmarkPhase.DECODE`` to a runtime-registry
lookup; ``BenchmarkPhase.FORWARD`` retains the original full-pass
subject as the equivalence baseline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
import torch

from src.video_compression.codec.codec import VideoCodec
from src.video_compression.config import CodecConfig
from src.video_compression.data.synthetic import (
    SyntheticPattern,
    SyntheticVideoConfig,
    SyntheticVideoGenerator,
)
from src.video_compression.perf.config import BenchmarkPhase

# Default runtime used when ``BenchmarkPhase.DECODE`` is requested
# without a more specific selection. Surfaced as a module-level
# constant so the perf module and tests reference one source of
# truth. Phase 2 / 4 / 5 add more entries to the registry; the default
# here only changes when there's a measured reason to.
DEFAULT_DECODE_RUNTIME_NAME = "pytorch-eager"

logger = structlog.get_logger(__name__)


@runtime_checkable
class BenchmarkSubject(Protocol):
    """Object that can be measured by the benchmark loop.

    Implementations are expected to be stateful: ``prepare`` configures
    state for one cell, ``step`` runs a single timed iteration. A subject
    must be re-preparable across cells without leaking memory.
    """

    @property
    def name(self) -> str:
        """Identifier used in log events and reports."""
        ...

    def prepare(self, *, batch_size: int, height: int, width: int) -> None:
        """Allocate per-cell state (input tensors, compiled graph, ...)."""
        ...

    def step(self) -> None:
        """Run one measurement iteration."""
        ...

    def teardown(self) -> None:
        """Release per-cell state."""
        ...


class CodecForwardSubject:
    """Measures the neural-only forward pass of ``VideoCodec``.

    No entropy coding is exercised; this is the lowest-variance subject
    and the right default for headline FPS numbers. Use
    ``CodecEncodeSubject`` / ``CodecDecodeSubject`` for end-to-end paths
    once those land.
    """

    def __init__(
        self,
        codec_config: CodecConfig,
        *,
        device: torch.device,
        pattern: SyntheticPattern,
        seed: int,
    ) -> None:
        self._codec_config = codec_config
        self._device = device
        self._pattern = pattern
        self._seed = seed
        self._codec: VideoCodec | None = None
        self._x: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return f"codec-forward-{self._device.type}"

    def prepare(self, *, batch_size: int, height: int, width: int) -> None:
        downsample = self._codec_config.encoder.downsample_factor
        if height % downsample != 0 or width % downsample != 0:
            raise ValueError(
                f"resolution {height}x{width} not divisible by codec "
                f"downsample_factor={downsample}",
            )

        # Build a fresh codec per cell to keep VRAM accounting clean. The
        # codec is small relative to the input tensor; this is cheaper than
        # reasoning about lingering activations across cells.
        self._codec = VideoCodec(
            config=self._codec_config,
            use_mcts_rate_control=False,
            device=str(self._device),
        ).to(self._device)
        self._codec.eval()

        gen = SyntheticVideoGenerator(
            SyntheticVideoConfig(
                pattern=self._pattern,
                num_frames=batch_size,
                height=height,
                width=width,
                seed=self._seed,
            ),
        )
        # generate() returns (T, C, H, W); we treat T as batch.
        self._x = gen.generate().to(self._device)

        logger.debug(
            "subject.prepared",
            subject=self.name,
            batch_size=batch_size,
            height=height,
            width=width,
            device=str(self._device),
        )

    def step(self) -> None:
        if self._codec is None or self._x is None:
            raise RuntimeError("step() called before prepare()")
        with torch.no_grad():
            self._codec(self._x)

    def teardown(self) -> None:
        self._codec = None
        self._x = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()


class RuntimeBackedDecoderSubject:
    """Benchmark subject that delegates decode to a registered runtime.

    Implements the ``BenchmarkSubject`` Protocol. On ``prepare`` it:

    1. Encodes a synthetic frame *once* with a fresh ``VideoCodec`` to
       get a realistic latent shape (this is one-time setup outside
       the timed loop).
    2. Looks up a runtime in ``RuntimeRegistry`` by name and calls
       its ``prepare`` with the latent shape.

    On ``step`` it calls ``runtime.decode(latent)`` — the runtime is
    the only thing being timed. ``teardown`` releases both the
    encoder-side state and the runtime's per-cell state.

    The runtime-name argument defaults to ``DEFAULT_DECODE_RUNTIME_NAME``
    (the eager baseline) so a vanilla ``BenchmarkPhase.DECODE`` cell
    still produces meaningful numbers. Phase 2 / 4 / 5 will let
    ``RuntimeProfile`` plumb in alternative runtime names.
    """

    def __init__(
        self,
        codec_config: CodecConfig,
        *,
        device: torch.device,
        pattern: SyntheticPattern,
        seed: int,
        runtime_name: str = DEFAULT_DECODE_RUNTIME_NAME,
    ) -> None:
        # Local imports keep ``perf.subjects`` importable even on
        # systems where the runtime package isn't installed (relevant
        # to optional backends in later iterations). Eager runtime is
        # always present so this can never fail in practice today.
        from src.video_compression.runtime.registry import (
            BaseDecoderRuntime,
            create_runtime,
        )

        self._codec_config = codec_config
        self._device = device
        self._pattern = pattern
        self._seed = seed
        self._runtime_name = runtime_name
        self._create_runtime = create_runtime

        self._runtime: BaseDecoderRuntime | None = None
        self._latent: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return f"runtime-decode-{self._runtime_name}-{self._device.type}"

    def prepare(self, *, batch_size: int, height: int, width: int) -> None:
        from src.video_compression.runtime.protocol import DecoderRuntimeContext

        downsample = self._codec_config.encoder.downsample_factor
        if height % downsample != 0 or width % downsample != 0:
            raise ValueError(
                f"resolution {height}x{width} not divisible by codec "
                f"downsample_factor={downsample}",
            )

        # Build a temporary codec just to get a realistic latent. We
        # discard it before timing starts; the runtime under test
        # will build its own decoder.
        temp_codec = VideoCodec(
            config=self._codec_config,
            use_mcts_rate_control=False,
            device=str(self._device),
        ).to(self._device)
        temp_codec.eval()

        gen = SyntheticVideoGenerator(
            SyntheticVideoConfig(
                pattern=self._pattern,
                num_frames=batch_size,
                height=height,
                width=width,
                seed=self._seed,
            ),
        )
        x = gen.generate().to(self._device)
        with torch.no_grad():
            latent = temp_codec.encoder(x)
        # Drop encoder state immediately so the runtime under test
        # gets a clean device.
        del temp_codec
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        self._latent = latent

        # Build the runtime under test and prepare it for this shape.
        # Forward our codec_config so the runtime decodes against the
        # exact same model state we encoded with — without this the
        # eager runtime would default to a different latent_channels
        # value and refuse the prepared context.
        self._runtime = self._create_runtime(
            self._runtime_name,
            codec_config=self._codec_config,
        )
        ctx = DecoderRuntimeContext(
            name=f"decode_ctx_{self._runtime_name}",
            batch_size=batch_size,
            latent_channels=self._codec_config.encoder.latent_channels,
            latent_height=height // downsample,
            latent_width=width // downsample,
            dtype="float32",
            device=str(self._device),
            model_hash=self._codec_config.compute_hash(),
        )
        self._runtime.prepare(ctx=ctx)

        logger.debug(
            "subject.prepared",
            subject=self.name,
            runtime=self._runtime_name,
            latent_shape=tuple(latent.shape),
            device=str(self._device),
        )

    def step(self) -> None:
        if self._runtime is None or self._latent is None:
            raise RuntimeError("step() called before prepare()")
        self._runtime.decode(self._latent)

    def teardown(self) -> None:
        if self._runtime is not None:
            self._runtime.teardown()
        self._runtime = None
        self._latent = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()


def create_subject(
    phase: BenchmarkPhase,
    codec_config: CodecConfig,
    *,
    device: torch.device,
    pattern: SyntheticPattern,
    seed: int,
    runtime_name: str | None = None,
) -> BenchmarkSubject:
    """Factory for benchmark subjects.

    ``BenchmarkPhase.FORWARD`` returns the legacy ``CodecForwardSubject``
    (encoder + entropy + decoder, no entropy coding) — kept as the
    equivalence baseline.

    ``BenchmarkPhase.DECODE`` returns ``RuntimeBackedDecoderSubject``
    backed by a runtime registered in ``src.video_compression.runtime``.
    Pass ``runtime_name`` to select a specific backend; ``None`` uses
    ``DEFAULT_DECODE_RUNTIME_NAME``.

    ``BenchmarkPhase.ENCODE`` is still future work (Phase 4, FFmpeg
    bridge); raises ``NotImplementedError`` with a clear pointer.
    """
    if phase is BenchmarkPhase.FORWARD:
        return CodecForwardSubject(
            codec_config,
            device=device,
            pattern=pattern,
            seed=seed,
        )
    if phase is BenchmarkPhase.DECODE:
        return RuntimeBackedDecoderSubject(
            codec_config,
            device=device,
            pattern=pattern,
            seed=seed,
            runtime_name=runtime_name or DEFAULT_DECODE_RUNTIME_NAME,
        )
    if phase is BenchmarkPhase.ENCODE:
        raise NotImplementedError(
            f"benchmark phase {phase.value!r} requires the end-to-end "
            f"encode subject. That ships with Phase 4 (FFmpeg bridge); "
            f"for headline neural throughput use BenchmarkPhase.FORWARD "
            f"or BenchmarkPhase.DECODE.",
        )
    raise ValueError(f"unknown benchmark phase: {phase!r}")
