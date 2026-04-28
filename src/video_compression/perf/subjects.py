"""Benchmark subjects.

A "subject" is the thing being timed. The benchmark loop calls
``prepare(...)`` once per cell to set up state for a given (resolution,
batch_size) and then calls ``step()`` exactly once per measured iteration.

This Protocol-driven design lets later phases (1: torch.compile / ONNX /
TensorRT runtimes) plug new subjects in without touching the benchmark
loop.
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


def create_subject(
    phase: BenchmarkPhase,
    codec_config: CodecConfig,
    *,
    device: torch.device,
    pattern: SyntheticPattern,
    seed: int,
) -> BenchmarkSubject:
    """Factory for benchmark subjects.

    Currently only ``forward`` is implemented; ``encode`` and ``decode``
    raise ``NotImplementedError`` with a pointer to the phase that adds
    them. This is preferable to silently falling back — the user asked
    for a specific phase and deserves a clear error.
    """
    if phase is BenchmarkPhase.FORWARD:
        return CodecForwardSubject(
            codec_config,
            device=device,
            pattern=pattern,
            seed=seed,
        )
    if phase in (BenchmarkPhase.ENCODE, BenchmarkPhase.DECODE):
        raise NotImplementedError(
            f"benchmark phase {phase.value!r} requires the end-to-end "
            f"encode/decode subject. That ships with Phase 4 (FFmpeg "
            f"bridge); for headline neural throughput use "
            f"BenchmarkPhase.FORWARD.",
        )
    raise ValueError(f"unknown benchmark phase: {phase!r}")
