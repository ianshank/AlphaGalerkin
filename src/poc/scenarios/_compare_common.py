"""Shared lifecycle for the four ``*_compare`` PoC families.

Do **not** put CUDA ``empty_cache`` teardown on :class:`src.poc.registry.BaseScenario`.
That would flush the GPU cache after every PoC scenario, including ones that
are still holding tensors for a follow-up cell. These helpers live here, next
to :mod:`src.poc.scenarios._centaur_common`, and are imported only by the
compare families:

* ``lshape_amr_compare``
* ``transfer_baseline_compare``
* ``stochastic_galerkin_compare``
* ``mcts_classical_amr_arena``

Name-lock is a **function** in :mod:`src.poc.scenarios._compare_lock` (not a
config field: adding a field would change ``compute_hash()``). Config modules
import that torch-free sibling so ``load_config_from_dict`` stays light. This
module re-exports the lock helper.

Lazy ``from src.research...`` imports stay inside ``execute`` / builders
(the poc↔research cycle).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from src.poc.device import resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario
from src.poc.scenarios._compare_lock import _name_locked as _name_locked
from src.poc.scenarios._compare_lock import lock_scenario_name as lock_scenario_name


def empty_cuda_cache() -> None:
    """Release GPU memory (no-op on CPU).

    Byte-identical to the four compare-scenario ``teardown`` bodies this
    module replaced.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SupportsMetricsMapping(Protocol):
    """Stochastic comparison: ``comparison.metrics`` is a read-only mapping.

    Declared as a read-only ``@property`` member, not a bare attribute: a bare
    ``metrics: Mapping[str, float]`` is a *settable* member, which a class
    exposing ``metrics`` through ``@property`` (``MultiSeedStochasticComparison``)
    does not satisfy. A read-only member is satisfied by a property **and** by
    a plain attribute, and ``dict[str, float]`` is accepted covariantly.
    """

    @property
    def metrics(self) -> Mapping[str, float]:
        """Headline + spread metrics for the comparison."""
        ...


class SupportsMetricsMethod(Protocol):
    """L-shape / transfer / arena: ``comparison.metrics()`` returns a mapping."""

    def metrics(self) -> Mapping[str, float]:
        """Headline + spread metrics for the comparison."""
        ...


def comparison_metrics(
    comparison: SupportsMetricsMapping | SupportsMetricsMethod,
) -> Mapping[str, float]:
    """Read metrics from either a ``.metrics()`` method or a ``.metrics`` mapping.

    Unify the two comparison types without forcing one harness shape.

    Dispatch is ``callable``-based, not ``isinstance`` on these Protocols:
    ``@runtime_checkable`` treats a ``metrics`` mapping as a method member
    (the attribute exists), so ``comparison.metrics()`` would call a dict.
    """
    # Attribute access, not ``getattr(comparison, "metrics", None)``: the
    # abstraction audit (scripts/audit_abstractions.py) credits a Protocol
    # member with a reader only on the ``.metrics`` form, so the ``getattr``
    # spelling left ``SupportsMetricsMapping.metrics`` looking unread
    # (Copilot review, PR #151). Same semantics: a missing attribute raises.
    try:
        raw = comparison.metrics
    except AttributeError as exc:
        raise TypeError(f"{type(comparison).__name__} has no metrics attribute") from exc
    if raw is None:
        raise TypeError(f"{type(comparison).__name__} has no metrics attribute")
    mapping: object = raw() if callable(raw) else raw
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"{type(comparison).__name__}.metrics must be a mapping or a "
            f"method returning one, got {type(mapping).__name__}"
        )
    return mapping


class CompareScenarioBase(BaseScenario):
    """Shared setup / teardown / metric recording for compare scenarios.

    Subclasses keep their own ``execute``, builders, and arena sidecar write.
    """

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None

    def setup(self) -> None:
        """Resolve the device, build the logger, and install default thresholds."""
        device_str = getattr(self.config, "device", None)
        if not isinstance(device_str, str):
            raise TypeError(f"{type(self.config).__name__} must declare a string device field")
        self._device = resolve_device(device_str, context=self.name)
        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            run_id=self.config.compute_hash(),
            device=str(self._device),
        )
        if not self.config.thresholds:
            get_defaults = getattr(self.config, "get_default_thresholds", None)
            if get_defaults is None:
                raise TypeError(
                    f"{type(self.config).__name__} must implement get_default_thresholds"
                )
            self.config.thresholds = get_defaults()
        self._scenario_logger.info("setup_complete", **self._setup_log_fields())

    def teardown(self) -> None:
        """Release GPU memory (no-op on CPU)."""
        empty_cuda_cache()

    @abstractmethod
    def _setup_log_fields(self) -> dict[str, Any]:
        """Structured extras for the ``setup_complete`` log event."""

    def _record_metrics(
        self,
        comparison: SupportsMetricsMapping | SupportsMetricsMethod,
    ) -> Mapping[str, float]:
        """Record every metric from the comparison object.

        Returns the mapping so subclasses can log extra fields (arena's
        ``dof_convention`` is not a metric).
        """
        assert self._scenario_logger is not None
        metrics = comparison_metrics(comparison)
        for name, value in metrics.items():
            self.record_metric(name, value)
            self._scenario_logger.metric(name, value)
        self._log_metrics_recorded(comparison, metrics)
        return metrics

    def _log_metrics_recorded(
        self,
        comparison: object,
        metrics: Mapping[str, float],
    ) -> None:
        """Structured log after metrics are recorded. Override for extras."""
        del comparison
        assert self._scenario_logger is not None
        self._scenario_logger.info("comparison_recorded", **dict(metrics))

    def _artifact_base_str(self) -> str:
        """``output_dir / artifact_basename`` without a suffix.

        Concatenate extensions (do not use ``Path.with_suffix``): a custom
        basename with an internal dot would be truncated.
        """
        output_dir = getattr(self.config, "output_dir", None)
        basename = getattr(self.config, "artifact_basename", None)
        if not isinstance(output_dir, str) or not isinstance(basename, str):
            raise TypeError(
                f"{type(self.config).__name__} must declare output_dir and "
                "artifact_basename string fields"
            )
        return str(Path(output_dir) / basename)

    def _write_csv_png_artifacts(
        self,
        payload: object,
        export_csv: Callable[..., Any],
        export_plot: Callable[..., Any],
    ) -> None:
        """Write and register the committed CSV/PNG artifacts."""
        assert self._scenario_logger is not None
        base_str = self._artifact_base_str()
        csv_path = export_csv(payload, Path(f"{base_str}.csv"))
        self.record_artifact("csv", str(csv_path))
        png_path = export_plot(payload, Path(f"{base_str}.png"))
        if png_path is not None:
            self.record_artifact("png", str(png_path))
        else:
            self._scenario_logger.warning("artifact_png_skipped", reason="matplotlib unavailable")


__all__ = [
    "CompareScenarioBase",
    "SupportsMetricsMapping",
    "SupportsMetricsMethod",
    "comparison_metrics",
    "empty_cuda_cache",
    "lock_scenario_name",
    "_name_locked",
]
