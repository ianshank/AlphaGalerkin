# AGENT.md - ONNX Deployment Module (`src/deployment/`)

## Persona

**Name**: Deployment Engineer
**Expertise**: Model export, ONNX format, quantization, inference optimization, runtime provider selection, model validation
**Mindset**: You bridge the gap between training and production. Models must be exported correctly, validated against PyTorch outputs, optionally quantized for edge deployment, and wrapped in a clean inference API with multi-provider support.

## Module Overview

This module handles the full deployment pipeline: exporting PyTorch models to ONNX format (with trace/script/dynamo strategies), applying quantization (dynamic/static/QAT), wrapping ONNX Runtime for inference with multi-provider fallback (CPU, CUDA, TensorRT, CoreML, OpenVINO), and validating exported models against PyTorch reference outputs.

## Design Patterns

### 1. Strategy Pattern (Export Method)
`ONNXExporter` supports three export strategies:
- **trace**: `torch.jit.trace` — fast, requires sample input
- **script**: `torch.jit.script` — handles control flow
- **dynamo**: PyTorch 2.0+ dynamo export with fallback to trace

### 2. Strategy Pattern (Quantization Mode)
`ModelQuantizer` dispatches to different quantization backends:
- **dynamic**: No calibration data, post-training weight quantization
- **static**: Requires calibration dataset for activation quantization
- **QAT**: Defined in `QuantizationMode` enum but **not yet implemented** in `ModelQuantizer.quantize()` (raises `ValueError`)

### 3. Adapter Pattern (Calibration Data Reader)
`CalibrationDataReader` adapts PyTorch datasets to ONNX Runtime's calibration interface. Implements `set_range()`, `rewind()`, and yields numpy arrays.

### 4. Facade Pattern (ONNXRuntime)
`ONNXRuntime` wraps ONNX Runtime's session API with a clean interface:
- Automatic provider fallback (CUDA → CPU)
- Dict/tensor input handling
- Batch inference splitting
- Performance benchmarking
- Context manager for resource cleanup

### 5. Configuration as Code (Pydantic)
- `ExportConfig`: ONNX opset, dynamic axes, optimization level
- `QuantizationConfig`: Mode, weight type, calibration method
- `RuntimeConfig`: Execution providers, threading, memory options
- `DeploymentConfig`: Composite of all three

### 6. Builder Pattern (Model Info)
`ONNXExporter.get_model_info()` extracts complete graph structure and shapes from exported models for introspection.

## Skills Required

- **ONNX format**: Operator sets, dynamic axes, graph optimization
- **PyTorch export**: `torch.onnx.export`, `torch.jit.trace`, `torch.jit.script`, dynamo
- **Quantization**: INT8/INT4, per-channel, calibration methods (MinMax, Entropy, Percentile)
- **ONNX Runtime**: Session options, execution providers, memory patterns
- **Model validation**: Output comparison with tolerances, shape verification
- **Performance benchmarking**: Latency measurement, throughput calculation

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Export Specialist** | `export_onnx.py` | Adding export methods, fixing operator support |
| **Quantization Specialist** | `quantize.py` | Tuning quantization, adding calibration methods |
| **Runtime Specialist** | `runtime.py` | Adding providers, optimizing inference |
| **Validation Specialist** | `validate.py` | Output comparison, accuracy testing |
| **Config Designer** | `config.py` | Adding provider options, export settings |

## Tools & Commands

```bash
# Run deployment tests
pytest tests/deployment/ -v

# Export model to ONNX
python -m src.deployment.export_onnx \
    --checkpoint path/to/model.pt \
    --output model.onnx

# Export with quantization
python -m src.deployment.export_onnx \
    --checkpoint path/to/model.pt \
    --output model_int8.onnx \
    --quantize dynamic

# Validate exported model
python -m src.deployment.validate \
    --pytorch path/to/model.pt \
    --onnx model.onnx
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `config.py` | Configuration schemas | `ExportConfig`, `QuantizationConfig`, `RuntimeConfig`, `DeploymentConfig`, `QuantizationMode`, `ExecutionProvider` |
| `export_onnx.py` | ONNX export | `ONNXExporter` |
| `quantize.py` | Model quantization | `ModelQuantizer`, `CalibrationDataReader` |
| `runtime.py` | Inference wrapper | `ONNXRuntime`, `InferenceResult`, `RuntimeMetrics` |
| `validate.py` | Export validation | `ModelValidator`, `ValidationResult` |

## Dependencies

**Internal**: `src.modeling` (model to export)
**External**: `torch`, `onnx`, `onnxruntime`, `pydantic`, `structlog`, `numpy`

## Conventions & Constraints

1. **Dynamic Axes**: Always export with dynamic batch and spatial dimensions for resolution independence.
2. **Validation Required**: Never deploy without running `ModelValidator.validate()` to compare against PyTorch outputs.
3. **Provider Fallback**: `ONNXRuntime` tries providers in order (e.g., CUDA → CPU). Configure provider list based on deployment target.
4. **Opset Version**: Default opset 17. Increase only if newer operators are needed.
5. **Metadata Embedding**: `ONNXExporter` embeds `model_name`, `model_version`, `export_method`, and `opset_version` in ONNX model metadata for traceability.
6. **Context Manager**: Always use `ONNXRuntime` as a context manager (`with ONNXRuntime(...) as rt:`) for proper resource cleanup.
7. **Quantization Calibration**: Static quantization requires representative calibration data. Use training data distribution.

## Export Flow

```
PyTorch Model
  → ONNXExporter.export(model, config)
      → torch.onnx.export (trace/script/dynamo)
      → onnx.optimizer (basic/extended/full)
      → Add metadata (version, config hash)
  → model.onnx

model.onnx
  → ModelQuantizer.quantize(model_path, config)
      → CalibrationDataReader (for static)
      → onnxruntime.quantization
  → model_int8.onnx

model_int8.onnx
  → ModelValidator.validate(pytorch_model, onnx_path)
      → Compare outputs at multiple test inputs
      → Report max_diff, speedup_ratio
  → ValidationResult (passed/failed)

model_int8.onnx
  → ONNXRuntime(model_path, config)
      → InferenceSession with provider fallback
      → .run(input) → InferenceResult
```
