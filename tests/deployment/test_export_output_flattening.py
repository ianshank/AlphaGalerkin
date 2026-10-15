"""Structured-output flattening in the ONNX export path (hygiene B45).

Defect class: a model whose forward returns a dataclass -- ``ModelOutput``
since PR #77, which made it a frozen dataclass instead of a ``NamedTuple`` --
reaches the exporter unflattened. A dataclass is not a pytree, so
``torch.export`` (behind ``torch.onnx.export`` since torch 2.9) and the legacy
``torch.jit.trace`` both reject it, and every export of the real model fails
while the tensor-returning toy models the passing suite exercised look fine.

Every test here runs on torch alone, with no onnx, onnxruntime or onnxscript:
export backends are replaced by recorders, ``ONNXRuntime`` by a stub, and the
tracer checks call ``torch.export.export`` / ``torch.jit.trace``, which ship
with torch. The real-ONNX journeys stay in
``tests/deployment/test_export_onnx_integration.py`` and
``tests/integration/test_onnx_e2e.py``.

Planted defects (harden-a-guard), 6/6 killed, each by the named test(s):

1. The wrapper removed from the export path (``export_target = model``), the
   historical state: ``TestExportDispatch::test_dataclass_output_reaches_every
   _backend_wrapped[trace|dynamo|script]``, ``...::test_non_tensor_field_fails
   _before_any_backend_runs`` and ``TestFieldOrder::test_export_logs_the
   _flattened_fields_and_output_names``.
2. The wrapper applied but not flattening (``forward`` returns
   ``self.model(x, *args)``), which leaves (1)'s tests green:
   ``TestTupleOutputWrapper::test_forward_returns_the_flattened_tuple`` and
   ``TestTracersAcceptTheWrapper::test_torch_export_accepts_the_wrapped_model``
   / ``...::test_legacy_jit_trace_accepts_the_wrapped_model``.
3. The wrapper applied to every model, which breaks the byte-for-byte
   promise for tensor and tuple outputs:
   ``TestExportDispatch::test_tensor_output_reaches_every_backend_unchanged``,
   ``...::test_tuple_output_reaches_every_backend_unchanged`` (all three
   methods) and ``...::test_tensor_output_trace_call_is_the_historical_one``.
4. Fields flattened in reverse declaration order:
   ``TestFieldOrder::test_flattened_order_matches_export_output_names`` and
   six more, starting with ``TestFlattenModelOutput::test_model_output
   _without_lbb_is_policy_then_value``.
5. ``ModelValidator`` back on its own ``hasattr(output, "policy_logits")``
   extraction instead of the shared helper:
   ``TestValidatorFlattensLikeTheExporter::test_any_dataclass_is_compared_positionally``.
6. The wrapper's training flag copied with ``self.train(model.training)``,
   which recurses and resets the caller's submodules:
   ``TestTupleOutputWrapper::test_wrapping_leaves_the_models_submodule_modes_alone``.

All killers run under ``-m "not gpu_required and not fem_required"``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from config.schemas import OperatorConfig
from src.deployment import runtime as runtime_module
from src.deployment.config import ExportConfig
from src.deployment.export_onnx import (
    ONNXExporter,
    TupleOutputWrapper,
    flatten_model_output,
)
from src.deployment.validate import ModelValidator
from src.modeling.model import AlphaGalerkinModel, ModelOutput

IN_FEATURES = 6
N_ACTIONS = 5
BATCH = 2
EXPORT_METHODS = ("trace", "dynamo", "script")

# A tiny AlphaGalerkinModel, the same shape the ONNX integration suite exports.
TINY_OPERATOR_CONFIG: dict[str, Any] = {
    "d_model": 16,
    "d_key": 8,
    "d_value": 8,
    "d_ffn": 32,
    "n_heads": 2,
    "n_galerkin_layers": 1,
    "n_softmax_layers": 1,
    "n_fourier_features": 8,
    "use_fnet_mixing": False,
    "input_channels": 4,
}
BOARD_SIZE = 9

# Which ModelOutput field each default ExportConfig output name carries.
FIELD_FOR_OUTPUT_NAME = {"policy": "policy_logits", "value": "value"}


def _inputs(batch: int = BATCH) -> Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, IN_FEATURES)


def _heads() -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(1)
    return torch.randn(BATCH, N_ACTIONS), torch.randn(BATCH, 1), torch.rand(BATCH)


class _TinyHeads(nn.Module):
    """Returns a ``ModelOutput``, like ``AlphaGalerkinModel``, at toy size."""

    def __init__(self, with_lbb: bool = False) -> None:
        super().__init__()
        self.policy = nn.Linear(IN_FEATURES, N_ACTIONS)
        self.value = nn.Linear(IN_FEATURES, 1)
        self.with_lbb = with_lbb

    def forward(self, x: Tensor) -> ModelOutput:
        lbb = x.abs().amin(dim=-1) if self.with_lbb else None
        return ModelOutput(
            policy_logits=self.policy(x),
            value=torch.tanh(self.value(x)),
            lbb_constant=lbb,
        )


class _TwoInputHeads(_TinyHeads):
    def forward(self, x: Tensor, bias: Tensor) -> ModelOutput:  # type: ignore[override]
        return super().forward(x + bias)


class _TensorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(IN_FEATURES, N_ACTIONS)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class _TupleModel(_TensorModel):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        out = self.linear(x)
        return out, out.sum(dim=-1, keepdim=True)


class _VectorFieldHeads(_TinyHeads):
    def forward(self, x: Tensor) -> ModelOutput:
        return super().forward(x).with_vector_fields({"velocity": x})


@dataclasses.dataclass(frozen=True)
class _Heads:
    """A dataclass output whose field names are not ``ModelOutput``'s."""

    logits: Tensor
    skipped: Tensor | None
    v: Tensor


class _HeadsModel(_TinyHeads):
    def forward(self, x: Tensor) -> _Heads:  # type: ignore[override]
        out = super().forward(x)
        return _Heads(logits=out.policy_logits, skipped=None, v=out.value)


class _RecordingLogger:
    """Stands in for the exporter's bound structlog logger."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _record(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    info = debug = warning = error = _record

    def bind(self, **_: Any) -> _RecordingLogger:
        return self


class _Backends:
    """Every export backend replaced by a recorder of the module it receives."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.modules: list[nn.Module] = []
        self.onnx_export_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        recorder = self

        def fake_onnx_export(model: nn.Module, args: Any, f: str, **kwargs: Any) -> None:
            recorder.modules.append(model)
            recorder.onnx_export_calls.append(((model, args, f), kwargs))
            Path(f).write_bytes(b"stub")

        def fake_jit_script(model: nn.Module) -> nn.Module:
            recorder.modules.append(model)
            return model

        class _Program:
            def save(self, path: str) -> None:
                Path(path).write_bytes(b"stub")

        def fake_dynamo_export(model: nn.Module, *_: Tensor) -> _Program:
            recorder.modules.append(model)
            return _Program()

        monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
        monkeypatch.setattr(torch.jit, "script", fake_jit_script)
        monkeypatch.setattr(torch.onnx, "dynamo_export", fake_dynamo_export, raising=False)
        monkeypatch.setattr(ONNXExporter, "_add_metadata", lambda self, path: None)


@pytest.fixture()
def backends(monkeypatch: pytest.MonkeyPatch) -> _Backends:
    return _Backends(monkeypatch)


def _export(model: nn.Module, tmp_path: Path, method: str = "trace") -> ONNXExporter:
    exporter = ONNXExporter(ExportConfig(export_method=method, optimization_level="none"))
    exporter._logger = _RecordingLogger()  # type: ignore[assignment]
    exporter.export(model.eval(), _inputs(), tmp_path / "model.onnx")
    return exporter


class TestFlattenModelOutput:
    """The shared helper on every output shape it accepts or refuses."""

    def test_model_output_without_lbb_is_policy_then_value(self) -> None:
        policy, value, _ = _heads()
        flat = flatten_model_output(ModelOutput(policy_logits=policy, value=value))
        assert len(flat) == 2
        assert flat[0] is policy
        assert flat[1] is value

    def test_model_output_with_lbb_appends_it_last(self) -> None:
        policy, value, lbb = _heads()
        flat = flatten_model_output(
            ModelOutput(policy_logits=policy, value=value, lbb_constant=lbb)
        )
        assert len(flat) == 3
        assert flat[0] is policy
        assert flat[1] is value
        assert flat[2] is lbb

    def test_tensor_becomes_a_one_element_tuple(self) -> None:
        policy, _, _ = _heads()
        flat = flatten_model_output(policy)
        assert isinstance(flat, tuple)
        assert len(flat) == 1
        assert flat[0] is policy

    def test_tuple_is_returned_unchanged(self) -> None:
        policy, value, _ = _heads()
        output = (policy, value)
        assert flatten_model_output(output) is output

    def test_list_becomes_a_tuple_of_the_same_tensors(self) -> None:
        policy, value, _ = _heads()
        flat = flatten_model_output([policy, value])
        assert isinstance(flat, tuple)
        assert flat[0] is policy
        assert flat[1] is value

    def test_any_dataclass_keeps_tensor_fields_in_declaration_order(self) -> None:
        policy, value, _ = _heads()
        flat = flatten_model_output(_Heads(logits=policy, skipped=None, v=value))
        assert len(flat) == 2
        assert flat[0] is policy
        assert flat[1] is value

    def test_non_tensor_field_raises_naming_it(self) -> None:
        policy, value, _ = _heads()
        output = ModelOutput(policy_logits=policy, value=value).with_vector_fields(
            {"velocity": policy}
        )
        with pytest.raises(TypeError, match=r"ModelOutput\.vector_fields"):
            flatten_model_output(output)

    def test_unsupported_type_raises(self) -> None:
        policy, _, _ = _heads()
        with pytest.raises(TypeError, match="dict"):
            flatten_model_output({"policy": policy})

    def test_a_dataclass_type_is_not_an_output(self) -> None:
        with pytest.raises(TypeError, match="of type type"):
            flatten_model_output(ModelOutput)


class TestFieldOrder:
    """Position i of the flattened output is what ``output_names[i]`` names."""

    def test_flattened_order_matches_export_output_names(self) -> None:
        model = AlphaGalerkinModel(OperatorConfig(**TINY_OPERATOR_CONFIG)).eval()
        torch.manual_seed(0)
        board = torch.randn(1, TINY_OPERATOR_CONFIG["input_channels"], BOARD_SIZE, BOARD_SIZE)
        with torch.no_grad():
            output = model(board)

        flat = flatten_model_output(output)
        output_names = ExportConfig().output_names
        assert len(flat) == len(output_names)
        for position, name in enumerate(output_names):
            assert flat[position] is getattr(output, FIELD_FOR_OUTPUT_NAME[name])

    def test_export_logs_the_flattened_fields_and_output_names(
        self, backends: _Backends, tmp_path: Path
    ) -> None:
        exporter = _export(_TinyHeads(), tmp_path)
        events = [f for e, f in exporter._logger.events if e == "model_output_flattened"]  # type: ignore[attr-defined]
        assert len(events) == 1
        [event] = events
        assert event["output_type"] == "ModelOutput"
        assert event["fields"] == ["policy_logits", "value"]
        assert event["output_names"] == exporter.config.output_names
        named_fields = [FIELD_FOR_OUTPUT_NAME[name] for name in event["output_names"]]
        assert named_fields == event["fields"]

    def test_tensor_output_logs_no_flattening(self, backends: _Backends, tmp_path: Path) -> None:
        exporter = _export(_TensorModel(), tmp_path)
        events = [e for e, _ in exporter._logger.events]  # type: ignore[attr-defined]
        assert "model_output_flattened" not in events


class TestTupleOutputWrapper:
    def test_forward_returns_the_flattened_tuple(self) -> None:
        model = _TinyHeads(with_lbb=True).eval()
        x = _inputs()
        out = TupleOutputWrapper(model)(x)
        expected = flatten_model_output(model(x))
        assert isinstance(out, tuple)
        assert len(out) == len(expected) == 3
        for got, want in zip(out, expected, strict=True):
            torch.testing.assert_close(got, want)

    def test_model_is_a_registered_submodule(self) -> None:
        model = _TinyHeads()
        wrapper = TupleOutputWrapper(model)
        assert wrapper.model is model
        assert set(dict(wrapper.named_parameters())) == {
            f"model.{name}" for name, _ in model.named_parameters()
        }

    @pytest.mark.parametrize("training", [True, False])
    def test_mirrors_the_wrapped_models_training_mode(self, training: bool) -> None:
        model = _TinyHeads().train(training)
        assert TupleOutputWrapper(model).training is training

    def test_wrapping_leaves_the_models_submodule_modes_alone(self) -> None:
        """``nn.Module.train`` recurses; copying the flag must not reset a submodule."""
        model = _TinyHeads().eval()
        model.value.train()
        TupleOutputWrapper(model)
        assert model.training is False
        assert model.value.training is True
        assert model.policy.training is False

    def test_extra_positional_inputs_are_passed_through(self) -> None:
        model = _TwoInputHeads().eval()
        x, bias = _inputs(), torch.ones(BATCH, IN_FEATURES)
        out = TupleOutputWrapper(model)(x, bias)
        torch.testing.assert_close(out[0], model(x, bias).policy_logits)


class TestTracersAcceptTheWrapper:
    def test_premise_torch_export_rejects_a_dataclass_output(self) -> None:
        """Why the wrapper exists.

        If this starts failing, torch learned to flatten dataclass outputs and
        the wrapper (and this module) can be retired.
        """
        with pytest.raises(RuntimeError, match="ModelOutput"):
            torch.export.export(_TinyHeads().eval(), (_inputs(),), strict=False)

    def test_torch_export_accepts_the_wrapped_model(self) -> None:
        model = _TinyHeads().eval()
        x = _inputs()
        program = torch.export.export(TupleOutputWrapper(model), (x,), strict=False)
        out = program.module()(x)
        expected = flatten_model_output(model(x))
        assert isinstance(out, tuple)
        assert len(out) == len(expected)
        for got, want in zip(out, expected, strict=True):
            torch.testing.assert_close(got, want)

    @pytest.mark.filterwarnings("ignore:`torch.jit.trace:FutureWarning")
    def test_legacy_jit_trace_accepts_the_wrapped_model(self) -> None:
        model = _TinyHeads().eval()
        x = _inputs()
        traced = torch.jit.trace(TupleOutputWrapper(model), (x,))
        out = traced(x)
        assert isinstance(out, tuple)
        torch.testing.assert_close(out[0], model(x).policy_logits)


class TestExportDispatch:
    """Which module each export method receives, with the backends recorded."""

    @pytest.mark.parametrize("method", EXPORT_METHODS)
    def test_dataclass_output_reaches_every_backend_wrapped(
        self, method: str, backends: _Backends, tmp_path: Path
    ) -> None:
        model = _TinyHeads()
        _export(model, tmp_path, method)
        assert backends.modules, "no export backend ran"
        for received in backends.modules:
            assert isinstance(received, TupleOutputWrapper)
            assert received.model is model

    @pytest.mark.parametrize("method", EXPORT_METHODS)
    def test_tensor_output_reaches_every_backend_unchanged(
        self, method: str, backends: _Backends, tmp_path: Path
    ) -> None:
        model = _TensorModel()
        _export(model, tmp_path, method)
        assert backends.modules, "no export backend ran"
        assert all(received is model for received in backends.modules)

    @pytest.mark.parametrize("method", EXPORT_METHODS)
    def test_tuple_output_reaches_every_backend_unchanged(
        self, method: str, backends: _Backends, tmp_path: Path
    ) -> None:
        model = _TupleModel()
        _export(model, tmp_path, method)
        assert backends.modules, "no export backend ran"
        assert all(received is model for received in backends.modules)

    def test_tensor_output_trace_call_is_the_historical_one(
        self, backends: _Backends, tmp_path: Path
    ) -> None:
        """The full ``torch.onnx.export`` call for an unwrapped model, argument by argument."""
        model = _TensorModel()
        exporter = _export(model, tmp_path)
        config = exporter.config
        [(args, kwargs)] = backends.onnx_export_calls
        assert args[0] is model
        torch.testing.assert_close(args[1][0], _inputs())
        assert args[2] == str(tmp_path / "model.onnx")
        assert kwargs == {
            "input_names": config.input_names,
            "output_names": config.output_names,
            "dynamic_axes": config.dynamic_axes,
            "opset_version": config.opset_version,
            "do_constant_folding": config.do_constant_folding,
            "export_params": config.export_params,
            "verbose": config.verbose,
        }

    def test_non_tensor_field_fails_before_any_backend_runs(
        self, backends: _Backends, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="vector_fields"):
            _export(_VectorFieldHeads(), tmp_path)
        assert backends.modules == []


class _ExactRuntime:
    """Stub ``ONNXRuntime``: returns the reference model's own flattened outputs."""

    reference: nn.Module
    perturbation: float = 0.0

    def __init__(self, path: Any) -> None:
        self.path = path

    def run(self, inputs: np.ndarray) -> SimpleNamespace:
        with torch.no_grad():
            flat = flatten_model_output(self.reference(torch.from_numpy(inputs)))
        return SimpleNamespace(
            policy=flat[0].numpy() + self.perturbation,
            value=flat[1].numpy(),
            inference_time_ms=1.0,
        )

    def close(self) -> None:
        pass


@pytest.fixture()
def exact_runtime(monkeypatch: pytest.MonkeyPatch) -> type[_ExactRuntime]:
    runtime_cls = type("_Runtime", (_ExactRuntime,), {})
    monkeypatch.setattr(runtime_module, "ONNXRuntime", runtime_cls)
    return runtime_cls


class TestValidatorFlattensLikeTheExporter:
    """``ModelValidator`` compares the PyTorch side through the shared helper."""

    @pytest.mark.parametrize("model_cls", [_TinyHeads, _TupleModel])
    def test_matching_outputs_pass(
        self, model_cls: type[nn.Module], exact_runtime: type[_ExactRuntime]
    ) -> None:
        exact_runtime.reference = model_cls().eval()
        result = ModelValidator().validate(exact_runtime.reference, "unused.onnx", _inputs())
        assert result.passed, result
        assert result.max_policy_diff == 0.0

    def test_any_dataclass_is_compared_positionally(
        self, exact_runtime: type[_ExactRuntime]
    ) -> None:
        """A dataclass without ``policy_logits`` is still compared, field by position."""
        exact_runtime.reference = _HeadsModel().eval()
        result = ModelValidator().validate(exact_runtime.reference, "unused.onnx", _inputs())
        assert result.failed_samples == 0
        assert result.max_value_diff == 0.0

    def test_a_mismatch_is_still_caught(self, exact_runtime: type[_ExactRuntime]) -> None:
        """Without this, a validator that passed everything would satisfy the tests above."""
        exact_runtime.reference = _TinyHeads().eval()
        exact_runtime.perturbation = 1.0
        result = ModelValidator().validate(exact_runtime.reference, "unused.onnx", _inputs())
        assert not result.passed
        assert result.max_policy_diff == pytest.approx(1.0)
