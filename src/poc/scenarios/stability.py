"""LBB stability monitoring scenario.

This scenario validates the stability claim:
    "LBB constant β > 0 throughout training"

The Ladyzhenskaya-Babuska-Brezzi (LBB) condition ensures well-posedness
of the Galerkin discretization. Monitoring β during training validates
that the inf-sup condition is maintained.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch

from src.poc.config import (
    ScenarioResult,
    ScenarioStatus,
    StabilityScenarioConfig,
)
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@scenario("stability")
class StabilityScenario(BaseScenario):
    """LBB stability monitoring scenario.

    Validates that the LBB constant remains positive:
        - At initialization
        - Across different resolutions
        - Throughout training

    Success Criteria:
        - LBB constant > threshold at all monitored points
        - No more than max_lbb_violations during training
    """

    config_class = StabilityScenarioConfig

    def __init__(
        self,
        config: StabilityScenarioConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize scenario."""
        super().__init__(config, **kwargs)
        self.config: StabilityScenarioConfig  # Type hint

        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None

    def setup(self) -> None:
        """Initialize resources."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            config_hash=self.config.compute_hash(),
        )

    def teardown(self) -> None:
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Execute the LBB stability validation.

        Returns:
            ScenarioResult with stability metrics.

        """
        if self._device is None or self._scenario_logger is None:
            raise RuntimeError("setup() must be called before execute()")

        # Validate resolutions list is non-empty
        if not self.config.resolutions:
            raise ValueError("resolutions list cannot be empty")

        torch.manual_seed(self.config.seed)

        # Phase 1: Test stability at initialization across resolutions
        self._scenario_logger.info("testing_initialization_stability")
        init_results = self._test_initialization_stability()

        # Phase 2: Test stability during training
        self._scenario_logger.info("testing_training_stability")
        training_results = self._test_training_stability()

        # Record metrics
        for res, lbb_values in init_results.items():
            self.record_metric(f"lbb_init_mean_{res}x{res}", float(np.mean(lbb_values)))
            self.record_metric(f"lbb_init_min_{res}x{res}", float(np.min(lbb_values)))

        self.record_metric("lbb_training_mean", float(np.mean(training_results["lbb_values"])))
        self.record_metric("lbb_training_min", float(np.min(training_results["lbb_values"])))
        self.record_metric("lbb_violations", training_results["n_violations"])

        # Evaluate thresholds
        init_violations = sum(
            1 for values in init_results.values() for v in values if v < self.config.lbb_threshold
        )

        training_violations = training_results["n_violations"]

        threshold_results = {
            "init_stability": init_violations == 0,
            "training_stability": (training_violations <= self.config.max_lbb_violations),
        }

        all_passed = all(threshold_results.values())
        status = ScenarioStatus.PASSED if all_passed else ScenarioStatus.FAILED

        # Create result
        end_time = datetime.now()
        assert self._start_time is not None
        duration = (end_time - self._start_time).total_seconds()

        return ScenarioResult.model_validate(
            {
                "scenario_name": self.name,
                "config_hash": self.config.compute_hash(),
                "status": status,
                "passed": all_passed,
                "metrics": dict(self._metrics),
                "threshold_results": threshold_results,
                "artifacts": {k: str(v) for k, v in self._artifacts.items()},
                "start_time": self._start_time,
                "end_time": end_time,
                "duration_seconds": duration,
                "device": str(self._device),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                # Custom fields (allowed by extra="allow" in model_config)
                "init_violations": init_violations,
                "training_violations": training_violations,
            }
        )

    def _test_initialization_stability(self) -> dict[int, list[float]]:
        """Test LBB stability at initialization across resolutions.

        Returns:
            Dict mapping resolution to list of LBB values.

        """
        from src.math_kernel.integral import GalerkinProjection

        if self._device is None:
            raise RuntimeError("setup() must be called before testing stability")

        results: dict[int, list[float]] = {}

        for resolution in self.config.resolutions:
            n_tokens = resolution * resolution

            projection = GalerkinProjection(
                d_model=self.config.d_model,
                d_key=self.config.d_key,
                d_value=self.config.d_value,
            ).to(self._device)

            lbb_values = []

            for _ in range(self.config.n_forward_passes):
                x = torch.randn(
                    self.config.batch_size,
                    n_tokens,
                    self.config.d_model,
                    device=self._device,
                )

                with torch.no_grad():
                    lbb = projection.compute_lbb_constant(x)
                    lbb_values.extend(lbb.cpu().tolist())

            results[resolution] = lbb_values

            if self._scenario_logger:
                self._scenario_logger.debug(
                    "init_stability_tested",
                    resolution=resolution,
                    mean_lbb=float(np.mean(lbb_values)),
                    min_lbb=float(np.min(lbb_values)),
                )

        return results

    def _test_training_stability(self) -> dict[str, Any]:
        """Test LBB stability during training.

        Returns:
            Dict with lbb_values list and n_violations count.

        """
        from src.math_kernel.integral import GalerkinProjection

        if self._device is None or self._scenario_logger is None:
            raise RuntimeError("setup() must be called before testing stability")

        # Use middle resolution for training test (validated non-empty in execute())
        resolution = self.config.resolutions[len(self.config.resolutions) // 2]
        n_tokens = resolution * resolution

        projection = GalerkinProjection(
            d_model=self.config.d_model,
            d_key=self.config.d_key,
            d_value=self.config.d_value,
        ).to(self._device)

        # Simple optimizer to update weights
        optimizer = torch.optim.Adam(
            projection.parameters(),
            lr=self.config.learning_rate,
        )

        lbb_values = []
        n_violations = 0
        log_interval = max(1, self.config.n_training_steps // 20)

        for step in range(self.config.n_training_steps):
            # Generate random input
            x = torch.randn(
                self.config.batch_size,
                n_tokens,
                self.config.d_model,
                device=self._device,
            )

            # Forward pass and dummy loss
            output = projection(x)
            loss = output.mean()  # Dummy loss to create gradient

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Monitor LBB
            with torch.no_grad():
                lbb = projection.compute_lbb_constant(x)
                lbb_mean = lbb.mean().item()
                lbb_values.append(lbb_mean)

                if lbb_mean < self.config.lbb_threshold:
                    n_violations += 1

            # Log progress
            if (step + 1) % log_interval == 0:
                self._scenario_logger.progress(
                    step + 1,
                    self.config.n_training_steps,
                    operation="training_stability",
                )
                self._scenario_logger.metric(
                    "lbb_constant",
                    lbb_mean,
                    step=step + 1,
                )

        return {
            "lbb_values": lbb_values,
            "n_violations": n_violations,
        }
