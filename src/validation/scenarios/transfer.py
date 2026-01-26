"""Zero-shot transfer validation scenario.

Validates that models trained on one resolution generalize
to different resolutions without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.validation.config import (
    TransferValidationConfig,
    ValidationResult,
    ValidationStatus,
)
from src.validation.logging import DebugContext
from src.validation.scenarios.base import BaseValidator
from src.validation.tolerance import ToleranceChecker


class TransferValidator(BaseValidator):
    """Validates zero-shot resolution transfer.

    Tests the core claim that models trained on 9x9 grids
    generalize to 19x19 grids without retraining.
    """

    name = "transfer_validation"
    config_class = TransferValidationConfig

    def __init__(
        self,
        config: TransferValidationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize transfer validator.

        Args:
            config: Transfer validation configuration.
            **kwargs: Override config fields.
        """
        super().__init__(config, **kwargs)
        self.config: TransferValidationConfig = self.config  # Type hint
        self._model: Any = None
        self._tolerance_checker = ToleranceChecker(config=self.config.tolerance)

    def setup(self) -> None:
        """Setup model and data generators."""
        import torch

        # Set random seed
        torch.manual_seed(self.config.seed)

        self._logger.info(
            "setup_start",
            train_resolution=self.config.train_resolution,
            eval_resolutions=self.config.eval_resolutions,
            model_path=self.config.model_path,
        )

    def validate(self) -> ValidationResult:
        """Run transfer validation.

        Returns:
            ValidationResult with transfer metrics.
        """
        import torch

        # Try to import physics model
        try:
            from src.experiments.physics_model import PhysicsOperator
            from src.physics.poisson import PoissonSolver
        except ImportError as e:
            self._logger.warning("physics_import_failed", error=str(e))
            return self._create_mock_result()

        with DebugContext("transfer_validation", self._logger):
            # Load or train model
            if self.config.model_path and Path(self.config.model_path).exists():
                model = self._load_model(self.config.model_path)
            else:
                model = self._train_model()

            if model is None:
                return self._create_result(
                    ValidationStatus.ERROR,
                    passed=False,
                    error="Failed to load or train model",
                )

            self._model = model

            # Evaluate on each resolution
            mse_results: dict[str, float] = {}
            relative_errors: dict[str, float] = {}

            for resolution in self.config.eval_resolutions:
                self._logger.info(
                    "evaluating_resolution",
                    resolution=resolution,
                    n_samples=self.config.n_eval_samples,
                )

                mse, rel_error = self._evaluate_resolution(
                    model, resolution, self.config.n_eval_samples
                )

                key = f"{resolution}x{resolution}"
                mse_results[key] = mse
                relative_errors[key] = rel_error

                self.record_metric(f"mse_{key}", mse)
                self.record_metric(f"relative_error_{key}", rel_error)

                self._logger.info(
                    "resolution_evaluated",
                    resolution=resolution,
                    mse=mse,
                    relative_error=rel_error,
                )

            # Calculate transfer metrics
            train_key = f"{self.config.train_resolution}x{self.config.train_resolution}"
            primary_key = f"{self.config.primary_eval_resolution}x{self.config.primary_eval_resolution}"

            train_mse = mse_results.get(train_key, float("inf"))
            primary_mse = mse_results.get(primary_key, float("inf"))
            transfer_ratio = primary_mse / train_mse if train_mse > 0 else float("inf")

            self.record_metric("train_mse", train_mse)
            self.record_metric("primary_eval_mse", primary_mse)
            self.record_metric("transfer_ratio", transfer_ratio)

            # Check success criteria
            passed = True
            failure_reasons: list[str] = []

            # Check MSE threshold on primary resolution
            if primary_mse > self.config.mse_threshold:
                passed = False
                failure_reasons.append(
                    f"Primary MSE {primary_mse:.4f} > threshold {self.config.mse_threshold}"
                )

            # Check relative transfer quality
            if transfer_ratio > self.config.relative_mse_threshold:
                passed = False
                failure_reasons.append(
                    f"Transfer ratio {transfer_ratio:.2f} > threshold {self.config.relative_mse_threshold}"
                )

            if failure_reasons:
                self.record_detail("failure_reasons", failure_reasons)

            return self._create_result(
                ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
                passed=passed,
                mse_results=mse_results,
                relative_errors=relative_errors,
            )

    def _train_model(self) -> Any:
        """Train a new physics model.

        Returns:
            Trained model or None on failure.
        """
        import torch

        try:
            from src.experiments.physics_model import PhysicsOperator
            from src.physics.poisson import PoissonSolver
        except ImportError:
            return None

        self._logger.info(
            "training_model",
            resolution=self.config.train_resolution,
            n_samples=self.config.n_train_samples,
            n_epochs=self.config.n_epochs,
        )

        # Generate training data
        solver = PoissonSolver(resolution=self.config.train_resolution)
        train_data = []

        for _ in range(self.config.n_train_samples):
            charges, potentials = solver.generate_sample(n_charges=self.config.n_charges)
            train_data.append((charges, potentials))

        # Create model
        model = PhysicsOperator(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            resolution=self.config.train_resolution,
        )

        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_data), self.config.batch_size):
                batch_charges = torch.stack(
                    [d[0] for d in train_data[i : i + self.config.batch_size]]
                )
                batch_potentials = torch.stack(
                    [d[1] for d in train_data[i : i + self.config.batch_size]]
                )

                optimizer.zero_grad()
                pred = model(batch_charges)
                loss = loss_fn(pred, batch_potentials)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                self._logger.debug(
                    "training_epoch",
                    epoch=epoch + 1,
                    loss=avg_loss,
                )

        return model

    def _load_model(self, path: str) -> Any:
        """Load a pre-trained model.

        Args:
            path: Path to model checkpoint.

        Returns:
            Loaded model or None on failure.
        """
        import torch

        try:
            from src.experiments.physics_model import PhysicsOperator
        except ImportError:
            return None

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            model = PhysicsOperator(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                resolution=self.config.train_resolution,
            )

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            self._logger.info("model_loaded", path=path)
            return model

        except Exception as e:
            self._logger.error("model_load_failed", path=path, error=str(e))
            return None

    def _evaluate_resolution(
        self,
        model: Any,
        resolution: int,
        n_samples: int,
    ) -> tuple[float, float]:
        """Evaluate model on a specific resolution.

        Args:
            model: Trained model.
            resolution: Grid resolution.
            n_samples: Number of samples.

        Returns:
            Tuple of (MSE, relative error).
        """
        import torch

        try:
            from src.physics.poisson import PoissonSolver
        except ImportError:
            return float("inf"), float("inf")

        model.eval()
        solver = PoissonSolver(resolution=resolution)

        total_mse = 0.0
        total_rel_error = 0.0

        with torch.no_grad():
            for _ in range(n_samples):
                charges, ground_truth = solver.generate_sample(
                    n_charges=self.config.n_charges
                )

                # Resize if needed
                if charges.shape[-1] != model.resolution:
                    charges = torch.nn.functional.interpolate(
                        charges.unsqueeze(0).unsqueeze(0),
                        size=(resolution, resolution),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()

                pred = model(charges.unsqueeze(0)).squeeze()

                # Calculate metrics
                mse = torch.mean((pred - ground_truth) ** 2).item()
                rel_error = (
                    torch.norm(pred - ground_truth) / torch.norm(ground_truth)
                ).item()

                total_mse += mse
                total_rel_error += rel_error

        return total_mse / n_samples, total_rel_error / n_samples

    def _create_mock_result(self) -> ValidationResult:
        """Create mock result when imports fail.

        Returns:
            ValidationResult with skip status.
        """
        return self._create_result(
            ValidationStatus.SKIPPED,
            passed=False,
            reason="Physics model imports not available",
        )
