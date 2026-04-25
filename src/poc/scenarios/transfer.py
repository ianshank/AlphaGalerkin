"""Zero-shot transfer scenario implementation.

This scenario validates the core claim:
    "Train on 9x9 -> Evaluate on 19x19 with MSE < 0.05"

It trains a PhysicsOperator on Poisson equation data at the training
resolution, then evaluates zero-shot transfer to unseen resolutions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.physics.poisson import PoissonSample
from src.poc.config import (
    ScenarioResult,
    ScenarioStatus,
    TransferScenarioConfig,
)
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


def _collate_poisson_samples(batch: list[PoissonSample]) -> list[PoissonSample]:
    """Custom collate function that keeps PoissonSample objects as-is.

    The default collator tries to stack/batch the dataclass fields automatically,
    which fails for numpy arrays in a dataclass. This function preserves the
    list of samples, allowing manual batching in the training loop.

    Args:
        batch: List of PoissonSample objects.

    Returns:
        Same list unchanged.

    """
    return batch


@scenario("transfer")
class TransferScenario(BaseScenario):
    """Zero-shot transfer validation scenario.

    Validates that a model trained on one resolution generalizes
    to different resolutions without retraining.

    Success Criteria:
        - MSE < threshold on primary_eval_resolution (default: 19x19)
        - All secondary resolutions also below threshold
    """

    config_class = TransferScenarioConfig

    def __init__(
        self,
        config: TransferScenarioConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize scenario."""
        super().__init__(config, **kwargs)
        self.config: TransferScenarioConfig  # Type hint

        # Will be set in setup()
        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._output_dir: Path | None = None
        self._scenario_logger: ScenarioLogger | None = None

    def setup(self) -> None:
        """Initialize resources."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._output_dir = Path("outputs/poc/transfer")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            config_hash=self.config.compute_hash(),
        )

        self._scenario_logger.info(
            "setup_complete",
            device=str(self._device),
            output_dir=str(self._output_dir),
        )

    def teardown(self) -> None:
        """Cleanup resources."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Execute the zero-shot transfer validation.

        Returns:
            ScenarioResult with transfer metrics.

        """
        import sys

        assert self._device is not None
        assert self._scenario_logger is not None

        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Phase 1: Train model
        self._scenario_logger.info(
            "training_start",
            resolution=self.config.train_resolution,
            n_samples=self.config.n_train_samples,
            n_epochs=self.config.n_epochs,
        )

        with self._scenario_logger.timed("training"):
            self._model = self._train_model()

        # Phase 2: Evaluate transfer
        transfer_results: dict[int, dict[str, float]] = {}

        for eval_res in self.config.eval_resolutions:
            self._scenario_logger.info(
                "evaluating_transfer",
                eval_resolution=eval_res,
            )

            with self._scenario_logger.timed(f"eval_{eval_res}x{eval_res}"):
                metrics = self._evaluate_at_resolution(eval_res)

            transfer_results[eval_res] = metrics

            # Record metrics
            for metric_name, value in metrics.items():
                self.record_metric(f"{metric_name}_{eval_res}x{eval_res}", value)

            self._scenario_logger.metric(
                f"mse_{eval_res}x{eval_res}",
                metrics["mse"],
                resolution=eval_res,
            )

        # Compute threshold results
        threshold_results = {}
        for eval_res in self.config.eval_resolutions:
            metric_name = f"mse_{eval_res}x{eval_res}"
            mse = transfer_results[eval_res]["mse"]
            passed = mse < self.config.mse_threshold
            threshold_results[metric_name] = passed

        # Primary result
        primary_res = self.config.primary_eval_resolution
        primary_mse = transfer_results[primary_res]["mse"]
        primary_passed = primary_mse < self.config.mse_threshold

        # Overall pass/fail
        all_passed = all(threshold_results.values())

        status = ScenarioStatus.PASSED if all_passed else ScenarioStatus.FAILED

        # Save model artifact
        if self._output_dir and self._model:
            model_path = self._output_dir / f"model_{self.config.compute_hash()}.pt"
            self._save_model(model_path)
            self.record_artifact("model", str(model_path))

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
                "primary_resolution": primary_res,
                "primary_mse": primary_mse,
                "primary_passed": primary_passed,
            }
        )

    def _train_model(self) -> nn.Module:
        """Train the physics operator model.

        Returns:
            Trained model.

        """
        from src.experiments.physics_model import PhysicsOperator
        from src.physics.poisson import PoissonDataset

        assert self._device is not None

        # Create model
        model = PhysicsOperator(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            n_fourier_features=self.config.n_fourier_features,
            fourier_scale=self.config.fourier_scale,
            use_fnet=self.config.use_fnet,
        ).to(self._device)

        # Create dataset
        dataset = PoissonDataset(
            grid_size=self.config.train_resolution,
            n_samples=self.config.n_train_samples,
            n_charges=self.config.n_charges,
            seed=self.config.seed,
        )

        # Create dataloader with custom collate function
        # The default collator fails on PoissonSample dataclass
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_poisson_samples,
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
        )

        # Training loop
        model.train()
        best_loss = float("inf")

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                coords = torch.tensor(
                    np.stack([s.coords for s in batch]),
                    device=self._device,
                    dtype=torch.float32,
                )
                charges = torch.tensor(
                    np.stack([s.charges for s in batch]),
                    device=self._device,
                    dtype=torch.float32,
                )
                targets = torch.tensor(
                    np.stack([s.potential for s in batch]),
                    device=self._device,
                    dtype=torch.float32,
                )

                optimizer.zero_grad()
                predictions = model(coords, charges)
                loss = torch.nn.functional.mse_loss(predictions, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Log progress
            if (epoch + 1) % 10 == 0 and self._scenario_logger:
                self._scenario_logger.progress(
                    epoch + 1,
                    self.config.n_epochs,
                    operation="training",
                )
                self._scenario_logger.metric(
                    "train_loss",
                    avg_loss,
                    epoch=epoch + 1,
                )

        self.record_metric("train_loss_final", best_loss)

        return model

    def _evaluate_at_resolution(self, resolution: int) -> dict[str, float]:
        """Evaluate model at a specific resolution.

        Args:
            resolution: Grid resolution to evaluate.

        Returns:
            Dict of metrics (mse, mae, rmse, max_error).

        """
        from src.physics.poisson import PoissonDataset

        assert self._model is not None
        assert self._device is not None

        # Use different seed for evaluation data
        eval_seed = self.config.seed + 50000 + resolution

        dataset = PoissonDataset(
            grid_size=resolution,
            n_samples=self.config.n_eval_samples,
            n_charges=self.config.n_charges,
            seed=eval_seed,
        )

        self._model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for i in range(0, len(dataset), self.config.batch_size):
                batch_indices = list(range(i, min(i + self.config.batch_size, len(dataset))))
                samples = [dataset[j] for j in batch_indices]

                coords = torch.tensor(
                    np.stack([s.coords for s in samples]),
                    device=self._device,
                    dtype=torch.float32,
                )
                charges = torch.tensor(
                    np.stack([s.charges for s in samples]),
                    device=self._device,
                    dtype=torch.float32,
                )
                targets = np.stack([s.potential for s in samples])

                predictions = self._model(coords, charges).cpu().numpy()

                all_predictions.append(predictions)
                all_targets.append(targets)

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        errors = predictions - targets
        mse = float(np.mean(errors**2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))
        max_error = float(np.max(np.abs(errors)))

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
        }

    def _save_model(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save model.

        """
        assert self._model is not None

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "n_fourier_features": self.config.n_fourier_features,
                "fourier_scale": self.config.fourier_scale,
                "use_fnet": self.config.use_fnet,
            },
            "scenario_config_hash": self.config.compute_hash(),
        }

        torch.save(checkpoint, path)

        if self._scenario_logger:
            self._scenario_logger.info("model_saved", path=str(path))
