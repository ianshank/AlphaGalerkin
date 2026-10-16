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

    ``forward`` names its first input: ``torch.export`` binds inputs by this
    signature, and a bare ``*args`` would bind the sample as one tuple that
    ``torch.onnx.export``'s ``dynamic_axes`` conversion cannot match.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.training = model.training

    def forward(self, x: Tensor, *args: Tensor) -> tuple[Tensor, ...]:
        return flatten_model_output(self.model(x, *args))


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
    n_inputs: int,
) -> tuple[dict[int, str] | None, ...]:
    """Return ``dynamic_axes`` as the ``dynamic_shapes`` the dynamo exporter takes.

    One spec per input, in positional order: a copy of the input's
    ``{axis: name}`` mapping, or ``None`` when it has no entry or no name.
    torch turns each name into ``torch.export.Dim.DYNAMIC`` and gives the
    graph's dimension that name, as its deprecated ``dynamic_axes``
    conversion does. Output axes have no slot; their dimensions follow from
    the inputs'. Inputs are matched to ``input_names`` by position, as in
    :func:`enlarge_size_one_dynamic_axes`, where torch's conversion used the
    mapping's order.

    Args:
        input_names: ONNX input names, as in ``ExportConfig.input_names``.
        output_names: ONNX output names, as in ``ExportConfig.output_names``.
        dynamic_axes: As in ``ExportConfig.dynamic_axes``.
        n_inputs: Number of tensors passed to the export.

    Raises:
        ValueError: An entry names neither a passed input nor an output.
            torch's conversion failed on these too, with a pytree length error.

    """
    passed = list(input_names[:n_inputs])
    unknown = [name for name in dynamic_axes if name not in passed and name not in output_names]
    if unknown:
        raise ValueError(
            f"dynamic_axes has entries for {unknown}, which name neither an input passed "
            f"to the export ({passed}) nor an output ({list(output_names)})."
        )
    specs = [dict(dynamic_axes[name]) if name in dynamic_axes else None for name in passed]
    return (*specs, *([None] * (n_inputs - len(passed))))


__all__ = [
    "DYNAMIC_AXIS_SAMPLE_SIZE",
    "SPECIALIZED_SAMPLE_SIZE",
    "AxisEnlargement",
    "TupleOutputWrapper",
    "dynamic_shapes_from_axes",
    "enlarge_size_one_dynamic_axes",
    "flatten_model_output",
    "is_structured_output",
    "structured_output_fields",
]
