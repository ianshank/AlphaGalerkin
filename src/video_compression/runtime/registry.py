"""Registry of decoder runtime implementations.

The registry is the single dispatch point for the perf benchmark when
it sees a ``RuntimeProfile.backend`` value. Implementations register
themselves at import time via the ``@register_runtime("name")``
decorator; the perf harness looks them up by the same name.

Design choices, all carried over from Phase 0 patterns:

* Reuses ``src/templates/registry.py::create_registry``. Thread-safe
  via the existing ``BaseRegistry._lock``; no re-implementation.
* ``BaseDecoderRuntime`` is a minimal ABC matching the established
  pattern (``BaseAnalyzer``, ``BaseEngine``). It implements the
  ``DecoderRuntime`` Protocol's lifecycle in concrete form so
  implementations only override the methods they care about.
* No hardcoded backend names; all dispatch keys flow through the
  ``RuntimeBackend`` enum (``src.video_compression.perf.config``).
* ``create_runtime`` factory is the single public lookup so callers
  never reach into ``RuntimeRegistry()`` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog
import torch

from src.templates.registry import create_registry
from src.video_compression.runtime.metadata import CompiledArtifactMetadata
from src.video_compression.runtime.protocol import DecoderRuntimeContext

logger = structlog.get_logger(__name__)


class BaseDecoderRuntime(ABC):
    """ABC for decoder-runtime implementations.

    Concrete subclasses fill in ``prepare`` / ``decode`` / ``teardown``
    and surface a ``CompiledArtifactMetadata`` via ``metadata``. The
    ABC handles bookkeeping shared across implementations — name,
    metadata invalidation across prepare cycles, and the "called
    decode before prepare" guard.

    By satisfying the ``DecoderRuntime`` Protocol structurally,
    instances pass ``isinstance(rt, DecoderRuntime)`` checks even
    without inheriting from the Protocol.
    """

    # Subclasses set this; it becomes the registry key.
    runtime_name: str = ""

    def __init__(self) -> None:
        self._metadata: CompiledArtifactMetadata | None = None
        self._prepared_ctx: DecoderRuntimeContext | None = None

    @property
    def name(self) -> str:
        if not self.runtime_name:
            raise RuntimeError(
                f"{type(self).__name__}.runtime_name not set; subclasses "
                f"must set the class attribute before registration",
            )
        return self.runtime_name

    @property
    def metadata(self) -> CompiledArtifactMetadata | None:
        """``CompiledArtifactMetadata`` once prepared, otherwise ``None``.

        Subclasses populate ``self._metadata`` inside ``prepare``;
        ``teardown`` resets it. Returning ``None`` (rather than
        raising) keeps ``@runtime_checkable`` Protocol isinstance
        checks usable outside the prepare window.
        """
        return self._metadata

    @abstractmethod
    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
        """Build / load per-cell state. Must populate ``self._metadata``."""

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Run a single decode iteration. See Protocol docstring."""

    def teardown(self) -> None:
        """Default teardown: clear metadata and prepared context.

        Subclasses override to release backend-specific state (compiled
        graphs, ONNX sessions, TensorRT engines).
        """
        self._metadata = None
        self._prepared_ctx = None


# The registry. Thread-safe by virtue of ``BaseRegistry._lock``.
# Note: ``create_registry`` wants a concrete ``type[T]`` for its
# ``base_class`` arg; ``BaseDecoderRuntime`` is abstract by design.
# At runtime the registry only stores concrete subclasses, so this
# is a sound use of ``type-abstract`` suppression.
RuntimeRegistry, register_runtime = create_registry(
    "DecoderRuntime",
    BaseDecoderRuntime,  # type: ignore[type-abstract]
)


def create_runtime(name: str, **kwargs: object) -> BaseDecoderRuntime:
    """Look up a registered runtime by name and instantiate it.

    Single public entry point so callers never touch
    ``RuntimeRegistry()`` directly. Raises a clear ``KeyError`` listing
    available runtimes when the name is unknown — easier to debug than
    a silent ``None``.

    ``**kwargs`` are forwarded to the runtime's constructor. This lets
    a caller (e.g. the perf benchmark) pass a specific ``codec_config``
    so the runtime decodes against the same model state the benchmark
    encoded with. Runtime classes that don't accept the supplied
    kwargs raise their own ``TypeError`` — we don't filter.
    """
    cls = RuntimeRegistry().get(name)
    if cls is None:
        available = sorted(RuntimeRegistry().list_items())
        raise KeyError(
            f"runtime {name!r} not registered; available: {available}. "
            f"Implementations register at import time via "
            f"@register_runtime; ensure the relevant module has been "
            f"imported (see src/video_compression/runtime/__init__.py).",
        )
    # The registry stores concrete subclasses; instantiation is safe.
    instance = cls(**kwargs)
    logger.debug(
        "runtime.created",
        runtime=name,
        cls=f"{cls.__module__}.{cls.__name__}",
        kwargs=sorted(kwargs.keys()),
    )
    return instance
