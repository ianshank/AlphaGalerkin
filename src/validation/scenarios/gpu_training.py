"""GPU training validation scenario.

Validates that training runs correctly on GPU with larger models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.validation.config import GPUTrainingConfig, ValidationResult, ValidationStatus
from src.validation.logging import DebugContext
from src.validation.scenarios.base import BaseValidator


class GPUTrainingValidator(BaseValidator):
    """Validates GPU training with larger models.

    Checks:
    1. GPU availability and configuration
    2. Model initialization on GPU
    3. Training loop stability
    4. Loss convergence
    5. LBB constant monitoring
    6. Gradient health
    """

    name = "gpu_training"
    config_class = GPUTrainingConfig

    def __init__(
        self,
        config: GPUTrainingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GPU training validator.

        Args:
            config: GPU training configuration.
            **kwargs: Override config fields.
        """
        super().__init__(config, **kwargs)
        self.config: GPUTrainingConfig = self.config  # Type hint
        self._device: str = "cpu"
        self._model: Any = None
        self._optimizer: Any = None
        self._output_dir: Path | None = None

    def setup(self) -> None:
        """Setup GPU and model."""
        import torch

        # Determine device
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        # Check GPU requirement
        if self.config.require_gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU required but not available")

        if torch.cuda.is_available():
            # Log GPU info
            self._logger.info(
                "gpu_detected",
                device_name=torch.cuda.get_device_name(),
                memory_total_gb=torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
            )

            # Set memory limit if specified
            if self.config.memory_limit_gb is not None:
                limit_bytes = int(self.config.memory_limit_gb * 1024**3)
                torch.cuda.set_per_process_memory_fraction(
                    limit_bytes / torch.cuda.get_device_properties(0).total_memory
                )

        # Set random seed
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # Create output directory
        self._output_dir = Path(f"outputs/validation/{self.name}")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "setup_complete",
            device=self._device,
            seed=self.config.seed,
            output_dir=str(self._output_dir),
        )

    def validate(self) -> ValidationResult:
        """Run GPU training validation.

        Returns:
            ValidationResult with training metrics.
        """
        import torch

        # Import model components
        try:
            from src.modeling.model import AlphaGalerkinModel
            from src.training.loss import AlphaGalerkinLoss
        except ImportError as e:
            self._logger.warning("model_import_failed", error=str(e))
            return self._create_mock_result()

        with DebugContext("gpu_training", self._logger):
            # Initialize model
            self._logger.info("initializing_model", d_model=self.config.d_model)

            model_config = {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_galerkin_layers": self.config.n_layers,
                "input_channels": 17,
            }

            try:
                model = AlphaGalerkinModel(**model_config)
                model = model.to(self._device)
                self._model = model
            except Exception as e:
                self._logger.error("model_init_failed", error=str(e))
                return self._create_result(
                    ValidationStatus.ERROR,
                    passed=False,
                    error="Model initialization failed",
                )

            # Initialize optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-4,
            )
            self._optimizer = optimizer

            # Initialize loss
            loss_fn = AlphaGalerkinLoss()

            # Initialize AMP scaler if using mixed precision
            scaler = None
            if self.config.mixed_precision and self._device != "cpu":
                scaler = torch.amp.GradScaler()

            # Training metrics
            losses: list[float] = []
            lbb_constants: list[float] = []
            gradient_norms: list[float] = []

            # Training loop
            self._logger.info(
                "training_start",
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
            )

            for step in range(self.config.n_steps):
                # Generate synthetic batch
                batch = self._generate_batch()

                # Forward pass with optional AMP
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        output = model(batch["features"])
                        loss_output = loss_fn(
                            output.policy_logits,
                            output.value,
                            batch["target_policy"],
                            batch["target_value"],
                        )
                    scaler.scale(loss_output.total).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_gradient_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch["features"])
                    loss_output = loss_fn(
                        output.policy_logits,
                        output.value,
                        batch["target_policy"],
                        batch["target_value"],
                    )
                    loss_output.total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_gradient_norm
                    )
                    optimizer.step()

                # Record metrics
                loss_val = float(loss_output.total.item())
                losses.append(loss_val)
                gradient_norms.append(float(grad_norm))

                # Get LBB constant if available
                if hasattr(output, "lbb_constant") and output.lbb_constant is not None:
                    lbb_constants.append(float(output.lbb_constant))

                # Logging
                if (step + 1) % self.config.log_interval == 0:
                    self._logger.info(
                        "training_step",
                        step=step + 1,
                        loss=loss_val,
                        grad_norm=float(grad_norm),
                        lbb=lbb_constants[-1] if lbb_constants else None,
                    )

                # Checkpoint
                if (step + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(step + 1)

            # Final metrics
            final_loss = losses[-1] if losses else float("inf")
            initial_loss = losses[0] if losses else float("inf")
            loss_ratio = final_loss / initial_loss if initial_loss > 0 else 1.0
            loss_decrease = 1.0 - loss_ratio

            self.record_metric("final_loss", final_loss)
            self.record_metric("initial_loss", initial_loss)
            self.record_metric("loss_decrease_ratio", loss_decrease)
            self.record_metric("mean_loss", sum(losses) / len(losses) if losses else 0)
            self.record_metric(
                "mean_gradient_norm",
                sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0,
            )

            if lbb_constants:
                self.record_metric("min_lbb_constant", min(lbb_constants))
                self.record_metric("mean_lbb_constant", sum(lbb_constants) / len(lbb_constants))

            # GPU memory stats
            if torch.cuda.is_available():
                self.record_metric(
                    "peak_gpu_memory_gb",
                    torch.cuda.max_memory_allocated() / 1024**3,
                )

            # Check success criteria
            passed = True
            failure_reasons: list[str] = []

            if final_loss > self.config.max_loss_threshold:
                passed = False
                failure_reasons.append(
                    f"Final loss {final_loss:.4f} > threshold {self.config.max_loss_threshold}"
                )

            if loss_decrease < self.config.loss_decrease_threshold:
                passed = False
                failure_reasons.append(
                    f"Loss decrease {loss_decrease:.4f} < threshold {self.config.loss_decrease_threshold}"
                )

            if lbb_constants and min(lbb_constants) < self.config.min_lbb_constant:
                passed = False
                failure_reasons.append(
                    f"Min LBB {min(lbb_constants):.2e} < threshold {self.config.min_lbb_constant}"
                )

            max_grad = max(gradient_norms) if gradient_norms else 0
            if max_grad > self.config.max_gradient_norm * 2:
                passed = False
                failure_reasons.append(
                    f"Max gradient norm {max_grad:.4f} too high"
                )

            # Record failure reasons
            if failure_reasons:
                self.record_detail("failure_reasons", failure_reasons)

            # Save final model if passed
            if passed and self.config.save_best_model:
                self._save_checkpoint(self.config.n_steps, is_best=True)

            return self._create_result(
                ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
                passed=passed,
                n_steps=self.config.n_steps,
                losses=losses[-10:] if len(losses) > 10 else losses,  # Last 10 losses
            )

    def _generate_batch(self) -> dict[str, Any]:
        """Generate a synthetic training batch.

        Returns:
            Dictionary with batch tensors.
        """
        import torch

        board_size = self.config.board_sizes[0]  # Use first board size
        n_actions = board_size * board_size + 1  # +1 for pass

        # Synthetic input
        features = torch.randn(
            self.config.batch_size,
            17,  # Input channels
            board_size,
            board_size,
            device=self._device,
        )

        # Synthetic targets
        target_policy = torch.softmax(
            torch.randn(self.config.batch_size, n_actions, device=self._device),
            dim=-1,
        )
        target_value = torch.tanh(
            torch.randn(self.config.batch_size, 1, device=self._device)
        )

        return {
            "features": features,
            "target_policy": target_policy,
            "target_value": target_value,
        }

    def _save_checkpoint(self, step: int, is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            step: Current training step.
            is_best: Whether this is the best model.
        """
        import torch

        if self._output_dir is None or self._model is None:
            return

        filename = "best_model.pt" if is_best else f"checkpoint_{step:06d}.pt"
        path = self._output_dir / filename

        torch.save(
            {
                "step": step,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict()
                if self._optimizer
                else None,
                "config": self.config.model_dump(),
            },
            path,
        )

        self.record_artifact(
            "best_model" if is_best else f"checkpoint_{step}",
            str(path),
        )
        self._logger.info("checkpoint_saved", path=str(path), step=step, is_best=is_best)

    def _create_mock_result(self) -> ValidationResult:
        """Create a mock result when model imports fail.

        Returns:
            ValidationResult indicating skip.
        """
        return self._create_result(
            ValidationStatus.SKIPPED,
            passed=False,
            reason="Model imports not available",
        )

    def teardown(self) -> None:
        """Cleanup GPU resources."""
        import gc

        # Clear model and optimizer
        self._model = None
        self._optimizer = None

        # Force garbage collection
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
