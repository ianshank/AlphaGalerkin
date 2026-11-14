"""Export adapters: structured outputs, size-1 dynamic axes, ``dynamic_shapes`` (B45, B50).

Two defect classes stopped the real ``AlphaGalerkinModel`` from exporting:

- A model whose forward returns a dataclass -- ``ModelOutput`` since PR #77,
  which made it a frozen dataclass instead of a ``NamedTuple`` -- reached the
  exporter unflattened. A dataclass is not a pytree, so ``torch.export``
  (behind ``torch.onnx.export`` since torch 2.9) and the legacy
  ``torch.jit.trace`` both reject it, while the tensor-returning toy models
  the passing suite exercised looked fine.
- A declared-dynamic axis seen at size 1 in the sample was pinned to 1 by
  ``torch.export``, so the exported graph failed at any other size. The
  exporter now traces a size-2 sample on such axes, and nowhere else.
- ``ExportConfig.dynamic_axes`` reached ``torch.onnx.export`` as
  ``dynamic_axes``, which its dynamo path converts through a function
  deprecated for removal (B50). Where torch defaults to the dynamo exporter
  (``ONNX_EXPORT_DEFAULTS_TO_DYNAMO``, torch 2.9+) the exporter now derives
  ``dynamic_shapes`` (``dynamic_shapes_from_axes``) and passes that, with
  ``dynamo=True``; older torch keeps the B45 call, since torch 2.6 rejects
  named axes in ``dynamic_shapes``. A caller's own ``dynamic_shapes``,
  ``dynamic_axes`` or ``dynamo`` takes precedence.

Every test here runs on torch alone, with no onnx, onnxruntime or onnxscript:
export backends are replaced by recorders, ``ONNXRuntime`` by a stub, and the
tracer checks call ``torch.export.export`` / ``torch.jit.trace``, which ship
with torch. The real-ONNX journeys are in
``tests/deployment/test_onnx_runtime_regressions.py``,
``tests/deployment/test_export_onnx_integration.py`` and
``tests/integration/test_onnx_e2e.py``.

Planted defects (harden-a-guard), 12/12 killed, each by the named test(s):

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
   methods) and ``...::test_tensor_output_trace_call_is_pinned``.
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
7. The enlargement removed from the export path (the pre-fix state):
   ``TestExportDispatch::test_size_one_dynamic_axis_reaches_every_backend
   _enlarged[trace|dynamo|script]`` and ``...::test_enlargement_is_logged``;
   with ONNX installed also ``test_onnx_runtime_regressions.py::
   TestSizeOneDynamicAxis::test_batch_dependent_model_runs_at_every_batch
   [2]`` and ``[4]``.
8. Every size-1 axis enlarged, declared dynamic or not:
   ``TestEnlargeSizeOneDynamicAxes::test_static_axis_at_size_one_is_left
   _alone`` and ``TestExportDispatch::test_tensor_output_trace_call_is
   _pinned[static-batch-at-1]``, among five.
9. Every declared-dynamic axis enlarged at size 1 or more, not only at 1:
   ``TestEnlargeSizeOneDynamicAxes::test_dynamic_axis_already_at_two_or
   _more_is_left_alone`` and ``TestExportDispatch::test_tensor_output_trace
   _call_is_pinned[dynamic-batch-already-2]``, among nine.
10. The repeat count left at 1, so nothing grows:
    ``TestEnlargeSizeOneDynamicAxes::test_size_one_dynamic_batch_is_repeated
    _to_two`` and the enlarged-dispatch tests of (7).
11. The ``sample_axis_enlarged`` event dropped:
    ``TestExportDispatch::test_enlargement_is_logged``.
12. Every input matched to the first name's axes:
    ``TestEnlargeSizeOneDynamicAxes::test_inputs_are_matched_to_names_by
    _position``.

B50, 8 of its 9 planted defects killed here (the ninth, a reworded torch
warning, by ``test_onnx_runtime_regressions.py``'s premise test). "Pinned"
is ``TestExportDispatch::test_tensor_output_trace_call_is_pinned``:

13. ``dynamic_axes`` passed again, without ``dynamic_shapes`` or ``dynamo``
    (the pre-fix call): pinned, all three cases; with ONNX installed also
    the no-warning tests in ``test_onnx_runtime_regressions.py`` and
    ``test_export_onnx_integration.py``.
14. ``dynamic_shapes`` and ``dynamic_axes`` both passed: pinned, all three.
    torch ignores ``dynamic_axes`` when given ``dynamic_shapes``, so this
    exports without either warning and only the pinned call catches it.
15. ``dynamo=True`` dropped: pinned, all three. Harmless on torch >= 2.9,
    where it is the default; on 2.6-2.8 the export would take the legacy
    exporter, which rejects ``dynamic_shapes``.
16. Every input passed as static: pinned ``[dynamic-batch-already-2]`` and
    ``[batch-at-1-another-axis-dynamic]``; with ONNX installed also the
    parity tests at untraced shapes.
17. Specs ordered by the mapping, not the inputs (torch's own conversion):
    ``test_dynamic_shapes.py::TestDynamicShapesFromAxes::test_axes_are_matched_to_inputs_by_position
    _not_mapping_order``.
18. Unknown names ignored instead of raising: ``test_dynamic_shapes.py::TestDynamicShapesFromAxes::
    test_axes_for_an_unknown_name_raise_naming_it`` and ``...::test_axes_for
    _a_named_input_that_was_not_passed_raise``.
19. Output axes given slots of their own: ``test_dynamic_shapes.py::TestDynamicShapesFromAxes::
    test_output_axes_are_left_to_torch`` and pinned, all three, among seven.
20. The config's own mappings returned, uncopied: ``test_dynamic_shapes.py::
    TestDynamicShapesFromAxes::test_specs_are_copies_of_the_configs_mappings``.

B50 review round (PR #165): 9 planted defects, 7 for B50 and 2 for B52 (the
quantization metric). 6 are killed here; the rest in
``test_onnx_runtime_regressions.py`` and ``test_export_onnx_integration.py``,
with ``python -m scripts.plant_mutations``:

21. The version gate ignored, ``dynamic_shapes`` sent on every torch (the
    first version of B50, which broke torch 2.6): ``TestExportDispatch::
    test_tensor_output_trace_call_before_a_dynamo_default_is_the_historical
    _one``, all three cases.
22. The detector reading the parameter's presence, not its default (torch
    2.6-2.8 have it, defaulting to False): ``test_dynamic_shapes.py::TestDynamoDefaultDetection::
    test_a_dynamo_default_of_false_is_not``.
23. A caller's own axes ignored: ``TestExportDispatch::test_a_callers_own_axes
    _are_passed_through_alone``, all three cases, ``None`` included.
24. A caller's ``dynamo`` passed twice: ``TestExportDispatch::test_a_callers
    _dynamo_true_is_passed_once``.
25. A multi-input spec left flat for ``TupleOutputWrapper``: ``TestExport
    Dispatch::test_a_two_input_dataclass_model_gets_its_extra_inputs_nested``;
    with ONNX installed the real two-input export fails in its fixture too.
26. Negative axes not counted from the end: ``test_dynamic_shapes.py::TestDynamicShapesFromAxes::
    test_negative_axes_are_counted_from_the_end``.

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
from src.deployment import export_onnx as export_onnx_module
from src.deployment import runtime as runtime_module
from src.deployment.config import ExportConfig
from src.deployment.export_adapters import (
    DYNAMIC_AXIS_SAMPLE_SIZE,
    SPECIALIZED_SAMPLE_SIZE,
    AxisEnlargement,
    TupleOutputWrapper,
    enlarge_size_one_dynamic_axes,
    flatten_model_output,
)
from src.deployment.export_onnx import ONNXExporter
from src.deployment.validate import ModelValidator
from src.modeling.model import AlphaGalerkinModel, ModelOutput
from tests.support.onnx_export import DEFAULT_BOARD_SHAPES

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

# The default ExportConfig's only input, whose axes 0, 2 and 3 it declares dynamic.
BOARD_INPUT = ExportConfig().input_names[0]


# Exports in which no axis may be enlarged: (sample shape, dynamic_axes, the
# dynamic_shapes torch.onnx.export must receive), where None keeps the default
# ExportConfig's axes.
UNENLARGED_EXPORTS: dict[
    str,
    tuple[tuple[int, ...], dict[str, dict[int, str]] | None, tuple[dict[int, str] | None, ...]],
] = {
    "dynamic-batch-already-2": ((BATCH, IN_FEATURES), None, (DEFAULT_BOARD_SHAPES,)),
    "static-batch-at-1": ((SPECIALIZED_SAMPLE_SIZE, IN_FEATURES), {}, (None,)),
    "batch-at-1-another-axis-dynamic": (
        (SPECIALIZED_SAMPLE_SIZE, IN_FEATURES),
        {BOARD_INPUT: {1: "features"}},
        ({1: "features"},),
    ),
}


def _inputs(batch: int = BATCH) -> Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, IN_FEATURES)


def _heads() -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(1)
    return torch.randn(BATCH, N_ACTIONS), torch.randn(BATCH, 1), torch.rand(BATCH)


def _board(batch: int = SPECIALIZED_SAMPLE_SIZE, side: int = BOARD_SIZE) -> Tensor:
    torch.manual_seed(2)
    return torch.randn(batch, TINY_OPERATOR_CONFIG["input_channels"], side, side)


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
    """Every export backend replaced by a recorder of the module and inputs it receives."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.modules: list[nn.Module] = []
        self.inputs: list[tuple[Tensor, ...]] = []
        self.onnx_export_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        recorder = self

        def fake_onnx_export(model: nn.Module, args: Any, f: str, **kwargs: Any) -> None:
            recorder.modules.append(model)
            recorder.inputs.append(tuple(args))
            recorder.onnx_export_calls.append(((model, args, f), kwargs))
            Path(f).write_bytes(b"stub")

        def fake_jit_script(model: nn.Module) -> nn.Module:
            recorder.modules.append(model)
            return model

        class _Program:
            def save(self, path: str) -> None:
                Path(path).write_bytes(b"stub")

        def fake_dynamo_export(model: nn.Module, *inputs: Tensor) -> _Program:
            recorder.modules.append(model)
            recorder.inputs.append(inputs)
            return _Program()

        monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
        monkeypatch.setattr(torch.jit, "script", fake_jit_script)
        monkeypatch.setattr(torch.onnx, "dynamo_export", fake_dynamo_export, raising=False)
        monkeypatch.setattr(ONNXExporter, "_add_metadata", lambda self, path: None)


@pytest.fixture()
def backends(monkeypatch: pytest.MonkeyPatch) -> _Backends:
    return _Backends(monkeypatch)


def _export(
    model: nn.Module,
    tmp_path: Path,
    method: str = "trace",
    sample: Tensor | None = None,
    **config: Any,
) -> ONNXExporter:
    exporter = ONNXExporter(ExportConfig(export_method=method, optimization_level="none", **config))
    exporter._logger = _RecordingLogger()  # type: ignore[assignment]
    sample = _inputs() if sample is None else sample
    exporter.export(model.eval(), sample, tmp_path / "model.onnx")
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

    def test_dynamic_shapes_bind_extra_inputs_as_its_varargs_tuple(self) -> None:
        """``forward(x, *args)`` binds as ``(x, args)``; a flat multi-input spec fails to export."""
        first, second, third = {0: "batch"}, {0: "batch"}, {1: "rows"}
        assert TupleOutputWrapper.dynamic_shapes_for((first,)) == (first,)
        assert TupleOutputWrapper.dynamic_shapes_for((first, second, third)) == (
            first,
            (second, third),
        )

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


class TestEnlargeSizeOneDynamicAxes:
    """The default ``ExportConfig`` declares ``board_state`` axes 0, 2 and 3 dynamic."""

    @staticmethod
    def _enlarge(
        *inputs: Tensor,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        names: list[str] | None = None,
    ) -> tuple[tuple[Tensor, ...], tuple[AxisEnlargement, ...]]:
        config = ExportConfig()
        return enlarge_size_one_dynamic_axes(
            inputs,
            config.input_names if names is None else names,
            config.dynamic_axes if dynamic_axes is None else dynamic_axes,
        )

    def test_size_one_dynamic_batch_is_repeated_to_two(self) -> None:
        sample = _board()
        (enlarged,), records = self._enlarge(sample)
        assert enlarged.shape == (DYNAMIC_AXIS_SAMPLE_SIZE, *sample.shape[1:])
        for row in enlarged:
            torch.testing.assert_close(row, sample[0])
        assert records == (
            AxisEnlargement(BOARD_INPUT, 0, SPECIALIZED_SAMPLE_SIZE, DYNAMIC_AXIS_SAMPLE_SIZE),
        )

    def test_every_size_one_dynamic_axis_is_enlarged_in_axis_order(self) -> None:
        sample = _board(side=SPECIALIZED_SAMPLE_SIZE)
        (enlarged,), records = self._enlarge(sample)
        size = DYNAMIC_AXIS_SAMPLE_SIZE
        assert enlarged.shape == (size, sample.shape[1], size, size)
        assert [record.axis for record in records] == [0, 2, 3]

    def test_static_axis_at_size_one_is_left_alone(self) -> None:
        sample = _board()
        spatial_only = {BOARD_INPUT: {2: "height", 3: "width"}}
        (same,), records = self._enlarge(sample, dynamic_axes=spatial_only)
        assert same is sample
        assert records == ()

    def test_dynamic_axis_already_at_two_or_more_is_left_alone(self) -> None:
        sample = _board(batch=DYNAMIC_AXIS_SAMPLE_SIZE)
        (same,), records = self._enlarge(sample)
        assert same is sample
        assert records == ()

    def test_size_zero_axis_is_left_alone(self) -> None:
        sample = _board(batch=0)
        (same,), records = self._enlarge(sample)
        assert same is sample
        assert records == ()

    def test_axis_outside_the_inputs_rank_is_left_alone(self) -> None:
        """Axes 2 and 3 do not exist on a 2-D input; torch.onnx reports that, as before."""
        sample = _inputs()
        (same,), records = self._enlarge(sample)
        assert same is sample
        assert records == ()

    def test_negative_axis_counts_from_the_end(self) -> None:
        sample = _board()
        (enlarged,), records = self._enlarge(sample, dynamic_axes={BOARD_INPUT: {-4: "batch"}})
        assert enlarged.shape[0] == DYNAMIC_AXIS_SAMPLE_SIZE
        assert [record.axis for record in records] == [-4]

    def test_inputs_are_matched_to_names_by_position(self) -> None:
        board, extra = _board(), torch.randn(SPECIALIZED_SAMPLE_SIZE, IN_FEATURES)
        tensors, records = self._enlarge(
            board,
            extra,
            names=[BOARD_INPUT, "extra"],
            dynamic_axes={BOARD_INPUT: {0: "batch"}, "extra": {1: "features"}},
        )
        assert tensors[0].shape[0] == DYNAMIC_AXIS_SAMPLE_SIZE
        assert tensors[1] is extra
        assert [record.input_name for record in records] == [BOARD_INPUT]

    def test_an_input_without_a_name_is_left_alone(self) -> None:
        board, unnamed = _board(), torch.randn(SPECIALIZED_SAMPLE_SIZE, IN_FEATURES)
        tensors, records = self._enlarge(board, unnamed)
        assert tensors[1] is unnamed
        assert [record.input_name for record in records] == [BOARD_INPUT]


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

    @pytest.mark.parametrize(
        ("shape", "dynamic_axes", "expected_shapes"),
        list(UNENLARGED_EXPORTS.values()),
        ids=list(UNENLARGED_EXPORTS),
    )
    def test_tensor_output_trace_call_is_pinned(
        self,
        shape: tuple[int, ...],
        dynamic_axes: dict[str, dict[int, str]] | None,
        expected_shapes: tuple[dict[int, str] | None, ...],
        backends: _Backends,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The full ``torch.onnx.export`` call, argument by argument, where nothing is enlarged.

        On a torch whose exporter defaults to dynamo (2.9+). B45 pinned this as the
        historical call; B50 changes one thing there: the input axes arrive as
        ``dynamic_shapes``, with the dynamo exporter requested explicitly, and
        ``dynamic_axes`` is no longer passed.
        """
        monkeypatch.setattr(export_onnx_module, "ONNX_EXPORT_DEFAULTS_TO_DYNAMO", True)
        model, exporter, sample = self._export_tensor_model(shape, dynamic_axes, tmp_path)
        self._assert_call(
            backends,
            (model, exporter, sample),
            tmp_path,
            {"dynamic_shapes": expected_shapes, "dynamo": True},
        )

    @pytest.mark.parametrize(
        ("shape", "dynamic_axes"),
        [entry[:2] for entry in UNENLARGED_EXPORTS.values()],
        ids=list(UNENLARGED_EXPORTS),
    )
    def test_tensor_output_trace_call_before_a_dynamo_default_is_the_historical_one(
        self,
        shape: tuple[int, ...],
        dynamic_axes: dict[str, dict[int, str]] | None,
        backends: _Backends,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Before torch 2.9 the B45 call goes out unchanged.

        There the default is the legacy exporter, which takes ``dynamic_axes``
        itself and never reaches the deprecated conversion. On torch 2.6,
        ``torch.export`` also rejects the axis names ``dynamic_shapes`` would
        carry, so sending them would fail every export.
        """
        monkeypatch.setattr(export_onnx_module, "ONNX_EXPORT_DEFAULTS_TO_DYNAMO", False)
        model, exporter, sample = self._export_tensor_model(shape, dynamic_axes, tmp_path)
        self._assert_call(
            backends,
            (model, exporter, sample),
            tmp_path,
            {"dynamic_axes": exporter.config.dynamic_axes},
        )

    @staticmethod
    def _export_tensor_model(
        shape: tuple[int, ...],
        dynamic_axes: dict[str, dict[int, str]] | None,
        tmp_path: Path,
    ) -> tuple[nn.Module, ONNXExporter, Tensor]:
        model = _TensorModel()
        torch.manual_seed(0)
        sample = torch.randn(*shape)
        overrides = {} if dynamic_axes is None else {"dynamic_axes": dynamic_axes}
        return model, _export(model, tmp_path, sample=sample, **overrides), sample

    @staticmethod
    def _assert_call(
        backends: _Backends,
        exported: tuple[nn.Module, ONNXExporter, Tensor],
        tmp_path: Path,
        axes_kwargs: dict[str, Any],
    ) -> None:
        model, exporter, sample = exported
        config = exporter.config
        [(args, kwargs)] = backends.onnx_export_calls
        assert args[0] is model
        assert len(args[1]) == 1
        assert args[1][0] is sample
        assert args[2] == str(tmp_path / "model.onnx")
        assert kwargs == {
            "input_names": config.input_names,
            "output_names": config.output_names,
            **axes_kwargs,
            "opset_version": config.opset_version,
            "do_constant_folding": config.do_constant_folding,
            "export_params": config.export_params,
            "verbose": config.verbose,
        }

    @staticmethod
    def _export_with_kwargs(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        export_kwargs: dict[str, Any],
        model: nn.Module | None = None,
        sample: Tensor | tuple[Tensor, ...] | None = None,
        **config: Any,
    ) -> ONNXExporter:
        monkeypatch.setattr(export_onnx_module, "ONNX_EXPORT_DEFAULTS_TO_DYNAMO", True)
        exporter = ONNXExporter(ExportConfig(optimization_level="none", **config))
        exporter._logger = _RecordingLogger()  # type: ignore[assignment]
        exporter.export(
            (model or _TensorModel()).eval(),
            _inputs() if sample is None else sample,
            tmp_path / "model.onnx",
            **export_kwargs,
        )
        return exporter

    def test_a_callers_dynamo_false_gets_the_configs_dynamic_axes(
        self, backends: _Backends, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy exporter rejects ``dynamic_shapes``; it gets the config's axes as before."""
        exporter = self._export_with_kwargs(monkeypatch, tmp_path, {"dynamo": False})
        [(_, kwargs)] = backends.onnx_export_calls
        assert kwargs["dynamo"] is False
        assert kwargs["dynamic_axes"] == exporter.config.dynamic_axes
        assert "dynamic_shapes" not in kwargs

    @pytest.mark.parametrize(
        ("argument", "theirs"),
        [
            ("dynamic_shapes", ({0: "n"},)),
            ("dynamic_shapes", None),
            ("dynamic_axes", {BOARD_INPUT: {0: "n"}}),
        ],
        ids=["dynamic_shapes", "dynamic_shapes-None", "dynamic_axes"],
    )
    def test_a_callers_own_axes_are_passed_through_alone(
        self,
        argument: str,
        theirs: Any,
        backends: _Backends,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nothing is derived when the caller passes axes, even ``None`` (an explicit "static")."""
        self._export_with_kwargs(monkeypatch, tmp_path, {argument: theirs})
        [(_, kwargs)] = backends.onnx_export_calls
        assert kwargs[argument] is theirs
        assert {"dynamic_shapes", "dynamic_axes", "dynamo"} & kwargs.keys() == {argument}

    def test_a_callers_dynamo_true_is_passed_once(
        self, backends: _Backends, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._export_with_kwargs(monkeypatch, tmp_path, {"dynamo": True})
        [(_, kwargs)] = backends.onnx_export_calls
        assert kwargs["dynamo"] is True
        assert kwargs["dynamic_shapes"] == (DEFAULT_BOARD_SHAPES,)

    def test_a_two_input_dataclass_model_gets_its_extra_inputs_nested(
        self, backends: _Backends, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        axes = {"x": {0: "batch"}, "bias": {0: "batch"}}
        self._export_with_kwargs(
            monkeypatch,
            tmp_path,
            {},
            model=_TwoInputHeads(),
            sample=(_inputs(), _inputs()),
            input_names=["x", "bias"],
            output_names=["policy", "value"],
            dynamic_axes=axes,
        )
        [(args, kwargs)] = backends.onnx_export_calls
        assert isinstance(args[0], TupleOutputWrapper)
        assert kwargs["dynamic_shapes"] == ({0: "batch"}, ({0: "batch"},))

    def test_non_tensor_field_fails_before_any_backend_runs(
        self, backends: _Backends, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="vector_fields"):
            _export(_VectorFieldHeads(), tmp_path)
        assert backends.modules == []

    @pytest.mark.parametrize("method", EXPORT_METHODS)
    def test_size_one_dynamic_axis_reaches_every_backend_enlarged(
        self, method: str, backends: _Backends, tmp_path: Path
    ) -> None:
        sample = _inputs(batch=SPECIALIZED_SAMPLE_SIZE)
        _export(_TensorModel(), tmp_path, method, sample=sample)
        assert backends.inputs, "no export backend received inputs"
        for (received,) in backends.inputs:
            assert received.shape == (DYNAMIC_AXIS_SAMPLE_SIZE, IN_FEATURES)
            for row in received:
                torch.testing.assert_close(row, sample[0])

    def test_enlargement_is_logged(self, backends: _Backends, tmp_path: Path) -> None:
        sample = _inputs(batch=SPECIALIZED_SAMPLE_SIZE)
        exporter = _export(_TensorModel(), tmp_path, sample=sample)
        events = [f for e, f in exporter._logger.events if e == "sample_axis_enlarged"]  # type: ignore[attr-defined]
        assert events == [
            {
                "input_name": BOARD_INPUT,
                "axis": 0,
                "from_size": SPECIALIZED_SAMPLE_SIZE,
                "to_size": DYNAMIC_AXIS_SAMPLE_SIZE,
            }
        ]

    @pytest.mark.parametrize(
        ("defaults_to_dynamo", "export_kwargs", "route"),
        [
            (True, {}, "dynamic_shapes"),
            (False, {}, "dynamic_axes"),
            (True, {"dynamo": False}, "dynamic_axes"),
            (True, {"dynamic_shapes": None}, "caller"),
            (False, {"dynamic_axes": {BOARD_INPUT: {0: "n"}}}, "caller"),
        ],
        ids=[
            "dynamo-default",
            "legacy-default",
            "caller-dynamo-false",
            "caller-shapes",
            "caller-axes",
        ],
    )
    def test_the_axes_route_is_logged(
        self,
        defaults_to_dynamo: bool,
        export_kwargs: dict[str, Any],
        route: str,
        backends: _Backends,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Which call did the exporter make? The first question about a version-gate problem."""
        monkeypatch.setattr(
            export_onnx_module, "ONNX_EXPORT_DEFAULTS_TO_DYNAMO", defaults_to_dynamo
        )
        exporter = ONNXExporter(ExportConfig(optimization_level="none"))
        exporter._logger = _RecordingLogger()  # type: ignore[assignment]
        exporter.export(_TensorModel().eval(), _inputs(), tmp_path / "m.onnx", **export_kwargs)
        events = [f for e, f in exporter._logger.events if e == "export_axes_route"]  # type: ignore[attr-defined]
        assert events == [{"route": route, "defaults_to_dynamo": defaults_to_dynamo}]


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
