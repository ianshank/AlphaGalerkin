"""Decoder runtime backends.

Phase 1 of the self-hosted transcoder roadmap. Provides a Protocol-
based dispatch layer between the perf benchmark and the actual decode
implementation, so future iterations can swap in ``torch.compile``,
ONNX Runtime, or TensorRT backends without touching the benchmark
loop.

Public surface:

- ``DecoderRuntime`` — Protocol every backend honours.
- ``DecoderRuntimeContext`` — per-cell setup parameters.
- ``BaseDecoderRuntime`` — ABC for implementations to subclass.
- ``CompiledArtifactMetadata`` — persisted metadata for cached
  artifacts (forward-compat schema).
- ``RuntimeRegistry`` / ``register_runtime`` / ``create_runtime`` —
  registration and lookup.

Importing this package registers the bundled runtimes
(currently just ``pytorch-eager``). Future iterations add their
modules to the import block below so registration runs at import time.
"""

from __future__ import annotations

from src.video_compression.runtime.metadata import (
    COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION,
    CompiledArtifactMetadata,
)
from src.video_compression.runtime.protocol import (
    DecoderRuntime,
    DecoderRuntimeContext,
)

# Side-effect import: registers PYTORCH_EAGER_RUNTIME_NAME.
# Iteration 2/4/5 add their modules here so registration runs at
# import time. Order is alphabetical to keep diffs minimal.
from src.video_compression.runtime.pytorch_compiled import (  # noqa: F401
    PYTORCH_COMPILED_RUNTIME_NAME,
    PyTorchCompiledRuntime,
)
from src.video_compression.runtime.pytorch_eager import (  # noqa: F401
    PYTORCH_EAGER_RUNTIME_NAME,
    PyTorchEagerRuntime,
)
from src.video_compression.runtime.registry import (
    BaseDecoderRuntime,
    RuntimeRegistry,
    create_runtime,
    register_runtime,
)

__all__ = [
    "BaseDecoderRuntime",
    "COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION",
    "CompiledArtifactMetadata",
    "DecoderRuntime",
    "DecoderRuntimeContext",
    "PYTORCH_COMPILED_RUNTIME_NAME",
    "PYTORCH_EAGER_RUNTIME_NAME",
    "PyTorchCompiledRuntime",
    "PyTorchEagerRuntime",
    "RuntimeRegistry",
    "create_runtime",
    "register_runtime",
]
