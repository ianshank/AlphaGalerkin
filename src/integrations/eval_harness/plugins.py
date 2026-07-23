"""Idempotent registration of AlphaGalerkin adapters into the harness registries.

:func:`register_all` wires the scorers, sink, and dataset into the harness
``SCORERS`` / ``SINKS`` / ``DATASETS`` registries. It is called explicitly by
:func:`~src.integrations.eval_harness.runner.run_eval` and (via
``_entrypoint``) by the harness's ``eval_harness.plugins`` entry-point discovery,
so it must be safe to call more than once — hence the membership guard.

Everything imported here is torch-free, so registration (including the global
entry-point path that fires on any ``eval_harness.bootstrap()``) triggers no
heavy imports. Importing this module has no side effects; registration happens
only when :func:`register_all` is invoked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def _maybe_register(
    registry: Any,
    name: str,
    cls: type,
    *,
    aliases: Iterable[str] = (),
) -> None:
    """Register ``cls`` under ``name`` unless the name (or an alias) already exists."""
    if name in registry:
        return
    registry.register_class(name, cls, aliases=tuple(aliases))


def register_all() -> None:
    """Register the AlphaGalerkin scorers, sink, and dataset (idempotent)."""
    from eval_harness.plugins import DATASETS, SCORERS, SINKS  # noqa: PLC0415

    from src.integrations.eval_harness.dataset import BasisOracleDataset  # noqa: PLC0415
    from src.integrations.eval_harness.scorers import (  # noqa: PLC0415
        FinalResidualScorer,
        PolicyTopKScorer,
    )
    from src.integrations.eval_harness.sink import ScenarioResultSink  # noqa: PLC0415

    _maybe_register(SCORERS, "final_residual", FinalResidualScorer)
    _maybe_register(SCORERS, "policy_topk", PolicyTopKScorer)
    _maybe_register(SINKS, "scenario_result", ScenarioResultSink)
    _maybe_register(DATASETS, "basis_oracle", BasisOracleDataset)


__all__ = ["register_all"]
