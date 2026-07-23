"""Pin the upstream ``eval_harness`` contract so a breaking change fails AG CI.

The adapter depends on a non-PyPI git dependency pinned to a commit SHA. If that
pin is ever moved and the harness's public types/registries drift, these
assertions fail loudly rather than the adapter breaking mysteriously at runtime.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("eval_harness")

from eval_harness.core import interfaces, types  # noqa: E402
from eval_harness.plugins import DATASETS, SCORERS, SINKS, TARGETS  # noqa: E402


def _field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_core_dataclass_fields_are_stable() -> None:
    assert _field_names(types.EvalItem) == {"id", "inputs", "expected", "metadata"}
    assert _field_names(types.TargetOutput) == {"output", "latency_ms", "error", "metadata"}
    assert _field_names(types.ScoreResult) == {"name", "value", "passed", "comment", "metadata"}
    assert _field_names(types.ScoreAggregate) == {"count", "mean", "pass_rate"}
    assert _field_names(types.RunResult) == {
        "run_id",
        "config_name",
        "items",
        "aggregate",
        "started_at",
        "finished_at",
    }


def test_interface_methods_are_stable() -> None:
    assert hasattr(interfaces.Scorer, "score")
    assert hasattr(interfaces.DatasetSource, "load")
    assert hasattr(interfaces.ResultSink, "emit")


def test_registry_contract_is_stable() -> None:
    for registry in (SCORERS, DATASETS, SINKS, TARGETS):
        assert hasattr(registry, "register_class")
        assert hasattr(registry, "create")
        assert hasattr(registry, "__contains__")


def test_callable_target_passes_inputs_dict() -> None:
    # The adapter's run_basis_cell relies on the callable target handing it
    # item.inputs (a dict) and wrapping the raw return in a TargetOutput.
    import sys
    import types as pytypes

    from eval_harness.plugins import bootstrap

    bootstrap()  # load the built-in `callable` target
    module = pytypes.ModuleType("_ag_eval_contract_probe")
    module.echo_inputs = lambda inputs: {"seen": inputs}  # type: ignore[attr-defined]
    sys.modules["_ag_eval_contract_probe"] = module
    try:
        target = TARGETS.create("callable", {"path": "_ag_eval_contract_probe:echo_inputs"})
        out = target.run(types.EvalItem(id="1", inputs={"x": 1}))
        assert out.output == {"seen": {"x": 1}}
        assert out.error is None
        assert out.latency_ms is not None
    finally:
        del sys.modules["_ag_eval_contract_probe"]


def test_eval_config_requires_schema_version() -> None:
    from eval_harness.config.models import EvalConfig

    with pytest.raises(Exception):  # noqa: B017 - any validation error is acceptable
        EvalConfig(
            schema_version="0.0",
            dataset={"type": "inline", "params": {}},
            target={"type": "echo", "params": {}},
        )
