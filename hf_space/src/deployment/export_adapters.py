"""Adapters between a model's Python inputs and outputs and an ONNX graph's.

Two mismatches stop a model from exporting the way it runs:

- **Outputs.** ``ModelOutput`` is a frozen dataclass, and a dataclass --
  unlike the ``NamedTuple`` it replaced -- is not a pytree, so
  ``torch.export`` (behind ``torch.onnx.export`` since torch 2.9) and
  ``torch.jit.trace`` both reject it as a graph output.
  :func:`flatten_model_output` and :class:`TupleOutputWrapper` turn it into
  a tuple of tensors.
- **Inputs.** ``torch.export`` specializes an input axis whose example size
  is 1 even when ``dynamic_axes`` declares it dynamic, once an op guards on
  it (``AlphaGalerkinModel``'s coordinate ``einsum`` does), and torch.onnx's
  fallback then exports a graph with that 1 baked in.
  :func:`enlarge_size_one_dynamic_axes` exports from a size-2 sample on
  such axes instead.

The axes themselves reach ``torch.onnx.export`` as ``dynamic_shapes``
(:func:`dynamic_shapes_from_axes`): handed ``dynamic_axes``, its dynamo path
converts them through a function deprecated for removal (hygiene B50).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Final

from torch import Tensor, nn

#: Example size at which ``torch.export`` specializes a declared-dynamic axis.
SPECIALIZED_SAMPLE_SIZE: Final[int] = 1

#: Smallest example size ``torch.export`` keeps symbolic on a dynamic axis.
DYNAMIC_AXIS_SAMPLE_SIZE: Final[int] = 2

#: How the axes reach ``torch.onnx.export`` (:func:`select_axes_route`): as the caller's own
#: arguments, as ``ExportConfig.dynamic_axes`` (legacy exporter), or as derived ``dynamic_shapes``.
AXES_ROUTE_CALLER: Final[str] = "caller"
AXES_ROUTE_DYNAMIC_AXES: Final[str] = "dynamic_axes"
AXES_ROUTE_DYNAMIC_SHAPES: Final[str] = "dynamic_shapes"
_CALLER_AXES_ARGUMENTS: Final[tuple[str, ...]] = ("dynamic_shapes", "dynamic_axes")


def is_structured_output(output: Any) -> bool:
    """True for a dataclass instance (``is_dataclass`` is also true of the class)."""
    return dataclasses.is_dataclass(output) and not isinstance(output, type)


def structured_output_fields(output: Any) -> tuple[tuple[str, Tensor], ...]:
    """``(name, tensor)`` per dataclass field in declaration order, ``None`` fields dropped.

    Any other value raises ``TypeError``: an ONNX graph output must be a tensor,
    and dropping the value silently would export a model that lost an output.
    """
    fields: list[tuple[str, Tensor]] = []
    for field in dataclasses.fields(output):
        value = getattr(output, field.name)
        if value is None:
            continue
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Cannot export {type(output).__name__}.{field.name}: it holds a "
                f"{type(value).__name__}, but an ONNX graph output must be a tensor."
            )
        fields.append((field.name, value))
    return tuple(fields)


def flatten_model_output(output: Any) -> tuple[Tensor, ...]:
    """Return a model output as the tuple of tensors its ONNX graph emits.

    Shared by the export path (inside :class:`TupleOutputWrapper`) and
    ``ModelValidator`` (on the PyTorch side), so both order outputs alike. A
    ``Tensor`` becomes ``(output,)``; a ``tuple`` is returned unchanged and a
    ``list`` as a tuple; a dataclass becomes its tensor fields in declaration
    order, ``None`` fields dropped. For ``ModelOutput`` that is
    ``policy_logits``, ``value``, then ``lbb_constant`` only when it is not
    ``None`` -- the order ``ExportConfig.output_names`` (``["policy",
    "value"]``) and ``ONNXRuntime.run`` assume.

    Raises:
        TypeError: Any other output type, or a non-tensor dataclass field
            (for example ``ModelOutput.vector_fields``).

    """
    if isinstance(output, Tensor):
        return (output,)
    if isinstance(output, tuple):
        return output
    if isinstance(output, list):
        return tuple(output)
    if is_structured_output(output):
        return tuple(tensor for _, tensor in structured_output_fields(output))
    raise TypeError(
        f"Cannot flatten a model output of type {type(output).__name__}: expected a "
        "Tensor, a tuple or list of tensors, or a dataclass of tensors."
    )


class TupleOutputWrapper(nn.Module):
    """Export adapter: the wrapped model's output through :func:`flatten_model_output`.

    ``ONNXExporter`` wraps a model only when it returns a dataclass. The
    wrapper copies the model's training flag onto itself only, leaving the
    model's submodules as they are.

    ``forward`` names its first input, because ``torch.export`` binds inputs
    by this signature: inputs arrive as ``(x, args)``. A per-input
    ``dynamic_shapes`` spec therefore has to be nested the same way
    (:meth:`dynamic_shapes_for`); a flat spec fails to export once there is a
    second input.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.training = model.training

    def forward(self, x: Tensor, *args: Tensor) -> tuple[Tensor, ...]:
        return flatten_model_output(self.model(x, *args))

    @staticmethod
    def dynamic_shapes_for(
        shapes: tuple[dict[int, str] | None, ...],
    ) -> tuple[Any, ...]:
        """Nest one spec per input as ``forward`` binds them: ``(first, (rest, ...))``."""
        if len(shapes) <= 1:
            return shapes
        return (shapes[0], tuple(shapes[1:]))


@dataclasses.dataclass(frozen=True)
class AxisEnlargement:
    """One sample axis repeated so ``torch.export`` keeps it dynamic."""

    input_name: str
    axis: int
    from_size: int
    to_size: int


def enlarge_size_one_dynamic_axes(
    input_tensors: tuple[Tensor, ...],
    input_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]],
) -> tuple[tuple[Tensor, ...], tuple[AxisEnlargement, ...]]:
    """Repeat each input to size 2 along every declared-dynamic axis where it has size 1.

    Inputs are matched to ``input_names`` by position. An axis that is static,
    already at size 2 or more, or out of the input's range is left alone, and
    an input with nothing to enlarge is returned as the same tensor object, so
    every other export reaches the backend exactly as before.

    Args:
        input_tensors: Sample inputs, in the model's positional order.
        input_names: ONNX input names, as in ``ExportConfig.input_names``.
        dynamic_axes: As in ``ExportConfig.dynamic_axes``.

    Returns:
        The inputs to export from, and one record per enlarged axis.

    """
    tensors = list(input_tensors)
    enlargements: list[AxisEnlargement] = []
    for index, name in enumerate(input_names[: len(tensors)]):
        for axis in sorted(dynamic_axes.get(name, {})):
            tensor = tensors[index]
            in_range = -tensor.ndim <= axis < tensor.ndim
            if not in_range or tensor.shape[axis] != SPECIALIZED_SAMPLE_SIZE:
                continue
            repeats = [1] * tensor.ndim
            repeats[axis] = DYNAMIC_AXIS_SAMPLE_SIZE
            tensors[index] = tensor.repeat(repeats)
            enlargements.append(
                AxisEnlargement(name, axis, SPECIALIZED_SAMPLE_SIZE, DYNAMIC_AXIS_SAMPLE_SIZE)
            )
    return tuple(tensors), tuple(enlargements)


def dynamic_shapes_from_axes(
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]],
    input_ranks: Sequence[int],
) -> tuple[dict[int, str] | None, ...]:
    """Return ``dynamic_axes`` as the ``dynamic_shapes`` the dynamo exporter takes.

    One spec per input, in positional order: a copy of the input's
    ``{axis: name}`` mapping, or ``None`` when it has no entry or no name.
    torch turns each name into ``torch.export.Dim.DYNAMIC`` and gives the
    graph's dimension that name, as its deprecated ``dynamic_axes``
    conversion does. Output axes have no slot; their dimensions follow from
    the inputs'. Inputs are matched to ``input_names`` by position, as in
    :func:`enlarge_size_one_dynamic_axes`, where torch's conversion used the
    mapping's order. A negative axis is counted from the end of its input,
    as there; torch would read it as no axis and export the dimension
    static. An axis outside the input's rank is passed on for torch to reject.

    Args:
        input_names: ONNX input names, as in ``ExportConfig.input_names``.
        output_names: ONNX output names, as in ``ExportConfig.output_names``.
        dynamic_axes: As in ``ExportConfig.dynamic_axes``.
        input_ranks: ``ndim`` of each tensor passed to the export, in order.

    Raises:
        ValueError: An entry names neither a passed, named input nor an
            output. torch's conversion failed on most of these with a pytree
            length error; where the mapping's order happened to line an entry
            up with an unnamed input, it used that input, which this does not.

    """
    passed = list(input_names[: len(input_ranks)])
    unknown = [name for name in dynamic_axes if name not in passed and name not in output_names]
    if unknown:
        raise ValueError(
            f"dynamic_axes has entries for {unknown}, which name neither an input passed "
            f"to the export ({passed}) nor an output ({list(output_names)}). Name every "
            "input in input_names, in the model's positional order."
        )
    specs: list[dict[int, str] | None] = []
    # Not strict: more tensors than names is allowed; those inputs are static.
    for name, rank in zip(passed, input_ranks, strict=False):
        axes = dynamic_axes.get(name)
        if axes is None:
            specs.append(None)
            continue
        specs.append(
            {(axis + rank if -rank <= axis < 0 else axis): dim for axis, dim in axes.items()}
        )
    return (*specs, *([None] * (len(input_ranks) - len(passed))))


def select_axes_route(overrides: Mapping[str, Any], defaults_to_dynamo: bool) -> str:
    """Which way the axes reach ``torch.onnx.export``, given the caller's own ``overrides``.

    Axes the caller passes win outright (``AXES_ROUTE_CALLER``): deriving more would
    leave torch two sources and drop one. Derived ``dynamic_shapes`` go out only where
    torch defaults to the dynamo exporter (``defaults_to_dynamo``, torch 2.9+) and the
    caller has not asked for the legacy one: those releases accept the string axis
    names, and only they reach the deprecated ``dynamic_axes`` conversion. Everywhere
    else (``AXES_ROUTE_DYNAMIC_AXES``) the call is the historical one, a caller's
    ``dynamo=True`` on torch 2.6-2.8 included: torch 2.6 rejects string axis names.
    """
    if any(name in overrides for name in _CALLER_AXES_ARGUMENTS):
        return AXES_ROUTE_CALLER
    if defaults_to_dynamo and overrides.get("dynamo", True):
        return AXES_ROUTE_DYNAMIC_SHAPES
    return AXES_ROUTE_DYNAMIC_AXES


__all__ = [
    "AXES_ROUTE_CALLER",
    "AXES_ROUTE_DYNAMIC_AXES",
    "AXES_ROUTE_DYNAMIC_SHAPES",
    "DYNAMIC_AXIS_SAMPLE_SIZE",
    "SPECIALIZED_SAMPLE_SIZE",
    "AxisEnlargement",
    "TupleOutputWrapper",
    "dynamic_shapes_from_axes",
    "enlarge_size_one_dynamic_axes",
    "flatten_model_output",
    "is_structured_output",
    "select_axes_route",
    "structured_output_fields",
]
