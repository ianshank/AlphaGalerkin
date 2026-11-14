"""``dynamic_shapes_from_axes`` and the dynamo-default detection (hygiene B50).

Split from ``test_export_adapters.py`` when it crossed the 1000-line test-module
budget (``tests/docs/test_module_size_budget.py``). Torch only: no onnx,
onnxruntime or onnxscript, no export runs. The dispatch tests that pin the
``torch.onnx.export`` call these helpers feed stay in ``test_export_adapters.py``,
whose docstring records the planted defects these tests kill (items 17-20, 22
and 26).
"""

from __future__ import annotations

import pytest
import torch
from packaging.version import Version
from torch import nn

from src.deployment import export_onnx as export_onnx_module
from src.deployment.config import ExportConfig
from src.deployment.export_adapters import (
    AXES_ROUTE_CALLER,
    AXES_ROUTE_DYNAMIC_AXES,
    AXES_ROUTE_DYNAMIC_SHAPES,
    dynamic_shapes_from_axes,
    select_axes_route,
)
from tests.support.onnx_export import DEFAULT_BOARD_SHAPES

# The default ExportConfig's only input, whose axes 0, 2 and 3 it declares dynamic.
BOARD_INPUT = ExportConfig().input_names[0]

# The torch release whose ``torch.onnx.export`` defaults to the dynamo exporter
# (its docstring: "versionchanged 2.9: dynamo is now True by default").
DYNAMO_DEFAULT_RELEASE = (2, 9)


class TestDynamicShapesFromAxes:
    """``ExportConfig.dynamic_axes`` as the positional ``dynamic_shapes`` torch.export binds."""

    def test_default_config_gives_the_board_axes(self) -> None:
        config = ExportConfig()
        shapes = dynamic_shapes_from_axes(
            config.input_names, config.output_names, config.dynamic_axes, input_ranks=(4,)
        )
        assert shapes == (DEFAULT_BOARD_SHAPES,)

    def test_axes_are_matched_to_inputs_by_position_not_mapping_order(self) -> None:
        """Torch's own conversion used the mapping's insertion order, not the inputs'."""
        axes = {"second": {1: "features"}, "first": {0: "batch"}}
        shapes = dynamic_shapes_from_axes(["first", "second"], [], axes, input_ranks=(2, 2))
        assert shapes == ({0: "batch"}, {1: "features"})

    def test_an_input_without_axes_is_static(self) -> None:
        shapes = dynamic_shapes_from_axes(["a", "b"], [], {"a": {0: "batch"}}, input_ranks=(2, 2))
        assert shapes == ({0: "batch"}, None)

    def test_an_input_without_a_name_is_static(self) -> None:
        shapes = dynamic_shapes_from_axes(["a"], [], {"a": {0: "batch"}}, input_ranks=(2, 2))
        assert shapes == ({0: "batch"}, None)

    def test_no_dynamic_axes_gives_one_static_spec_per_input(self) -> None:
        assert dynamic_shapes_from_axes(["a", "b"], ["y"], {}, input_ranks=(2, 2)) == (None, None)

    def test_output_axes_are_left_to_torch(self) -> None:
        """Output dims follow from the inputs' symbols: ``dynamic_shapes`` has no slot for them."""
        axes = {"a": {0: "batch"}, "y": {0: "batch"}, "z": {1: "rows"}}
        shapes = dynamic_shapes_from_axes(["a"], ["y", "z"], axes, input_ranks=(2,))
        assert shapes == ({0: "batch"},)

    def test_axes_for_an_unknown_name_raise_naming_it(self) -> None:
        """Torch's conversion failed on these too, with a pytree error; this names the key."""
        with pytest.raises(ValueError, match="'bogus'"):
            dynamic_shapes_from_axes(
                ["a"], ["y"], {"a": {0: "batch"}, "bogus": {0: "b"}}, input_ranks=(2,)
            )

    def test_axes_for_a_named_input_that_was_not_passed_raise(self) -> None:
        with pytest.raises(ValueError, match="'b'"):
            dynamic_shapes_from_axes(["a", "b"], [], {"b": {0: "batch"}}, input_ranks=(2,))

    def test_negative_axes_are_counted_from_the_end(self) -> None:
        """Torch reads a negative key as no axis at all, so the graph came out static."""
        axes = {"a": {-4: "batch", -1: "width"}}
        shapes = dynamic_shapes_from_axes(["a"], [], axes, input_ranks=(4,))
        assert shapes == ({0: "batch", 3: "width"},)

    def test_an_axis_outside_the_rank_is_passed_on_for_torch_to_reject(self) -> None:
        axes = {"a": {-5: "under", 4: "over"}}
        shapes = dynamic_shapes_from_axes(["a"], [], axes, input_ranks=(4,))
        assert shapes == ({-5: "under", 4: "over"},)

    def test_specs_are_copies_of_the_configs_mappings(self) -> None:
        config = ExportConfig()
        (spec,) = dynamic_shapes_from_axes(
            config.input_names, config.output_names, config.dynamic_axes, input_ranks=(4,)
        )
        assert spec is not None
        spec[1] = "channels"
        assert config.dynamic_axes[BOARD_INPUT] == DEFAULT_BOARD_SHAPES


class TestDynamoDefaultDetection:
    """``ONNX_EXPORT_DEFAULTS_TO_DYNAMO`` picks the call: read from the exporter's signature."""

    def test_a_dynamo_default_of_true_is_detected(self) -> None:
        def export(model: nn.Module, *, dynamo: bool = True) -> None:
            del model, dynamo

        assert export_onnx_module._defaults_to_dynamo(export) is True

    def test_a_dynamo_default_of_false_is_not(self) -> None:
        """Torch 2.6-2.8: the parameter exists, but the legacy exporter is the default."""

        def export(model: nn.Module, *, dynamo: bool = False) -> None:
            del model, dynamo

        assert export_onnx_module._defaults_to_dynamo(export) is False

    def test_no_dynamo_parameter_is_not(self) -> None:
        def export(model: nn.Module) -> None:
            del model

        assert export_onnx_module._defaults_to_dynamo(export) is False

    def test_the_installed_torch_agrees_with_its_release(self) -> None:
        """Derived two ways: from the signature in production, from the release number here."""
        expected = Version(torch.__version__).release[:2] >= DYNAMO_DEFAULT_RELEASE
        assert export_onnx_module.ONNX_EXPORT_DEFAULTS_TO_DYNAMO is expected


class TestSelectAxesRoute:
    """The exporter's routing decision, as a pure function of the caller's kwargs and torch."""

    @pytest.mark.parametrize(
        ("overrides", "defaults_to_dynamo", "route"),
        [
            ({}, True, AXES_ROUTE_DYNAMIC_SHAPES),
            ({}, False, AXES_ROUTE_DYNAMIC_AXES),
            ({"dynamo": True}, False, AXES_ROUTE_DYNAMIC_AXES),
            ({"dynamo": False}, True, AXES_ROUTE_DYNAMIC_AXES),
            ({"dynamic_shapes": None}, True, AXES_ROUTE_CALLER),
            ({"dynamic_axes": {}}, False, AXES_ROUTE_CALLER),
            ({"dynamic_shapes": ({0: "n"},), "dynamo": False}, True, AXES_ROUTE_CALLER),
            ({"opset_version": 18}, True, AXES_ROUTE_DYNAMIC_SHAPES),
        ],
        ids=[
            "torch-default-dynamo",
            "torch-default-legacy",
            "caller-dynamo-on-old-torch-keeps-the-historical-call",
            "caller-legacy-on-new-torch",
            "caller-shapes-even-none",
            "caller-axes",
            "caller-axes-beat-caller-dynamo",
            "unrelated-kwarg",
        ],
    )
    def test_route(self, overrides: dict, defaults_to_dynamo: bool, route: str) -> None:
        assert select_axes_route(overrides, defaults_to_dynamo) == route

    def test_the_routes_are_distinct(self) -> None:
        routes = {AXES_ROUTE_CALLER, AXES_ROUTE_DYNAMIC_AXES, AXES_ROUTE_DYNAMIC_SHAPES}
        assert len(routes) == 3
