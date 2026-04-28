"""Decoder runtime Protocol and per-cell context.

Phase 1 introduces an indirection so the perf benchmark can sweep
across multiple decoder backends (eager PyTorch, torch.compile, AMP,
ONNX Runtime, TensorRT) via the same loop. The Protocol is the
contract every backend honours.

Patterns lifted directly from Phase 0:

* ``@runtime_checkable`` Protocol (mirrors ``BenchmarkSubject``) so
  third-party / non-subclassing implementations satisfy isinstance.
* Stateful object with explicit ``prepare`` / ``decode`` / ``teardown``
  lifecycle, so the benchmark loop owns timing and cleanup.
* ``BaseModuleConfig`` for the per-cell context with ``extra="forbid"``
  (transient — never persisted, so strictness catches typos).

The decoded tensor convention matches ``VideoCodec.decode_frame``:
input ``latent`` shape ``(B, latent_channels, H/down, W/down)``,
output ``(B, 3, H, W)`` in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, Field, model_validator

from src.templates.config import BaseModuleConfig
from src.video_compression.runtime.metadata import CompiledArtifactMetadata


class DecoderRuntimeContext(BaseModuleConfig):
    """Per-cell setup parameters for a decoder runtime.

    Constructed once per benchmark cell; passed to ``prepare`` so the
    runtime can build (or load) any per-shape artifact (compiled graph,
    ONNX session, TensorRT engine).

    Strict ``extra="forbid"`` — this object is transient and never
    persisted, so unknown fields are programmer errors.

    ``protected_namespaces=()`` overrides Pydantic's default warning
    on ``model_*`` field names; ``model_hash`` is the canonical name
    we want here and it's not actually a Pydantic-internal accessor.
    """

    # Inherit BaseModuleConfig's extra="forbid" while clearing the
    # protected ``model_`` prefix so model_hash doesn't trigger a
    # spurious warning at every import.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Input description.
    #
    # Upper bounds are intentionally generous (16K-class) so 8K
    # research video (7680x4320) and exploratory 16K work
    # (15360x8640) fit without bumping the schema. They still serve
    # as guardrails — typos like ``height=80000`` fail loud rather
    # than silently allocating a multi-GB tensor. Bump again only
    # when a real workload exceeds these.
    batch_size: int = Field(..., ge=1, le=8192)
    latent_channels: int = Field(..., ge=1, le=4096)
    latent_height: int = Field(..., ge=1, le=16384)
    latent_width: int = Field(..., ge=1, le=16384)

    # Numerical settings
    dtype: str = Field(
        default="float32",
        description=(
            "Target dtype for the runtime's input/output. The runtime is "
            "responsible for casting from this dtype if its internal "
            "precision differs (e.g. AMP runtime accepts fp32 input and "
            "casts internally)."
        ),
    )

    # Hardware
    device: str = Field(
        ...,
        description=(
            "Device string the runtime will execute on. Accepts 'cpu', "
            "'cuda', or 'cuda:N'. A plain 'cuda' value is valid and "
            "uses the default/current CUDA device — the benchmark does "
            "not require the preference to be resolved to an indexed "
            "ordinal before calling prepare(). Runtime implementations "
            "should normalize 'cuda' -> 'cuda:N' internally so any "
            "device-equality checks against decoded-tensor devices "
            "work correctly."
        ),
    )

    # Provenance
    model_hash: str = Field(
        ...,
        min_length=1,
        description=(
            "Stable identifier for the model state being decoded. "
            "Runtimes use this as a cache key — a mismatch must rebuild "
            "the artifact."
        ),
    )

    # Optional pre-built artifact
    artifact_path: str | None = Field(
        default=None,
        description=(
            "Filesystem path to a pre-built backend artifact (compiled "
            "graph cache, ONNX file, TensorRT engine). None means the "
            "runtime should build on the fly."
        ),
    )

    @model_validator(mode="after")
    def _validate_dtype(self) -> DecoderRuntimeContext:
        # Accept the dtypes torch knows about in our supported set.
        # Guarded here so individual runtime implementations don't have
        # to duplicate the check.
        allowed = {"float32", "float16", "bfloat16"}
        if self.dtype not in allowed:
            raise ValueError(
                f"dtype {self.dtype!r} not in supported set {sorted(allowed)}",
            )
        return self

    def torch_dtype(self) -> torch.dtype:
        """Resolve the string dtype to a ``torch.dtype``."""
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping[self.dtype]


@runtime_checkable
class DecoderRuntime(Protocol):
    """Decoder backend with explicit lifecycle.

    Implementations are *stateful* — ``prepare`` configures state for
    one (model, input_shape, dtype) cell, ``decode`` runs a single
    timed iteration, ``teardown`` releases state. A runtime must be
    re-preparable across cells without leaking memory.

    The benchmark loop owns timing and cleanup; implementations focus
    on the actual decode call and any per-cell setup.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in log events and reports."""
        ...

    @property
    def metadata(self) -> CompiledArtifactMetadata | None:
        """Provenance + perf-relevant info about the prepared artifact.

        Returns ``None`` outside the ``prepare`` / ``teardown`` window.
        Callers that need a definite metadata reference (e.g. the
        equivalence checker) must assert non-None *after* ``prepare``.

        Why None rather than raising: ``@runtime_checkable`` Protocol
        ``isinstance`` checks call ``getattr`` on every attribute,
        including this property. A raising property would make every
        ``isinstance(rt, DecoderRuntime)`` check fail before
        ``prepare`` — useless for callers that want to validate type
        ahead of lifecycle.
        """
        ...

    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
        """Allocate per-cell state.

        Must be safe to call repeatedly with the same context. A
        repeated ``prepare`` may reuse existing state or rebuild it,
        but must leave the runtime ready to ``decode`` for ``ctx``;
        a ``prepare`` after ``teardown`` is a fresh build. The
        bundled eager runtime always rebuilds; future compiled /
        ONNX / TensorRT runtimes are free to fast-path when the
        context matches the cached artifact's.
        """
        ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Single timed decode call.

        Args:
            latent: Quantised latent tensor matching
                ``ctx.latent_channels``, ``ctx.latent_height``,
                ``ctx.latent_width`` and ``ctx.batch_size``.

        Returns:
            Reconstructed RGB frame in ``[0, 1]``, shape
            ``(B, 3, H, W)`` where ``H = latent_height *
            upsample_factor`` and likewise for ``W``.

        """
        ...

    def teardown(self) -> None:
        """Release per-cell state.

        Safe to call without a prior ``prepare`` (no-op).
        """
        ...
