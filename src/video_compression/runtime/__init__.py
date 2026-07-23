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

# Side-effect import: registers PYTORCH_EAGER_RUNTIME_NAME.
# Iteration 2/4/5 add their modules here so registration runs at
# import time. Order is alphabetical to keep diffs minimal.
from src.video_compression.runtime.onnx_runtime import (  # noqa: F401
    ONNX_RUNTIME_NAME,
    ONNXDecoderRuntime,
)
from src.video_compression.runtime.protocol import (
    DecoderRuntime,
    DecoderRuntimeContext,
)
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
from src.video_compression.runtime.tensorrt_runtime import (  # noqa: F401
    TENSORRT_RUNTIME_NAME,
    TensorRTRuntime,
)

__all__ = [
    "BaseDecoderRuntime",
    "COMPILED_ARTIFACT_METADATA_SCHEMA_VERSION",
    "CompiledArtifactMetadata",
    "DecoderRuntime",
    "DecoderRuntimeContext",
    "ONNX_RUNTIME_NAME",
    "ONNXDecoderRuntime",
    "PYTORCH_COMPILED_RUNTIME_NAME",
    "PYTORCH_EAGER_RUNTIME_NAME",
    "PyTorchCompiledRuntime",
    "PyTorchEagerRuntime",
    "RuntimeRegistry",
    "TENSORRT_RUNTIME_NAME",
    "TensorRTRuntime",
    "create_runtime",
    "register_runtime",
]
