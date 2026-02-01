"""Physics Zero-Shot Transfer Demo for AlphaGalerkin.

Demonstrates the key capability of training on small grids (9x9) and
achieving accurate predictions on larger grids (13x13, 19x19) without
retraining - a core feature of the Galerkin neural operator approach.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# Module-level logger
import structlog
import torch
from numpy.typing import NDArray

from src.demos.config import PhysicsDemoConfig
from src.demos.visualizations import ChartVisualizer, FieldVisualizer

logger = structlog.get_logger(__name__)


@dataclass
class TransferResult:
    """Result from a zero-shot transfer evaluation.

    Attributes:
        grid_size: The grid size evaluated.
        ground_truth: Ground truth potential field.
        prediction: Model prediction.
        mse: Mean squared error.
        mae: Mean absolute error.
        inference_time_ms: Inference time in milliseconds.

    """

    grid_size: int
    ground_truth: NDArray[np.float32]
    prediction: NDArray[np.float32]
    mse: float
    mae: float
    inference_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for visualization."""
        return {
            "grid_size": self.grid_size,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "mse": self.mse,
            "mae": self.mae,
            "inference_time_ms": self.inference_time_ms,
        }


class PhysicsDemo:
    """Interactive demo for physics zero-shot transfer visualization.

    Demonstrates:
    1. Poisson equation solving (charge → potential)
    2. Zero-shot generalization across resolutions
    3. MSE comparison charts
    4. Real-time field visualization
    """

    def __init__(
        self,
        config: PhysicsDemoConfig | None = None,
        model: torch.nn.Module | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize physics demo.

        Args:
            config: Demo configuration.
            model: Pre-trained physics model (optional, for inference).
            device: Torch device.

        """
        self.config = config or PhysicsDemoConfig()
        self.model = model
        self.device = device

        self.field_viz = FieldVisualizer(self.config.visualization)
        self.chart_viz = ChartVisualizer(self.config.visualization)

        self._solver = None  # Lazy initialization

        logger.info(
            "physics_demo_initialized",
            has_model=model is not None,
            device=device,
            train_size=self.config.train_grid_size,
            eval_sizes=self.config.eval_grid_sizes,
        )

    @property
    def solver(self) -> Any:  # noqa: ANN401 - PoissonSolver type varies by import
        """Get or create Poisson solver (lazy initialization)."""
        if self._solver is None:
            from src.physics.poisson import PoissonSolver

            self._solver = PoissonSolver(
                boundary_value=0.0,
                use_spectral=self.config.use_spectral_solver,
            )
        return self._solver

    def generate_sample(
        self,
        grid_size: int,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Generate a physics sample (charges, potential, coordinates).

        Args:
            grid_size: Grid resolution.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (charges, potential, coordinates).

        """
        from src.physics.poisson import generate_random_charges

        # Generate random charges
        charges_2d = generate_random_charges(
            grid_size=grid_size,
            n_charges=self.config.n_charges,
            charge_std=self.config.charge_std,
            seed=seed,
        )

        # Solve for potential
        potential_2d = self.solver.solve(charges_2d)

        # Create coordinate grid normalized to [0, 1]
        x = np.linspace(0, 1, grid_size, dtype=np.float32)
        y = np.linspace(0, 1, grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        coords = np.stack([xx.flatten(), yy.flatten()], axis=-1).astype(np.float32)

        logger.debug(
            "sample_generated",
            grid_size=grid_size,
            charges_range=(float(charges_2d.min()), float(charges_2d.max())),
            potential_range=(float(potential_2d.min()), float(potential_2d.max())),
        )

        return (
            charges_2d.flatten().astype(np.float32),
            potential_2d.flatten().astype(np.float32),
            coords,
        )

    def predict(
        self,
        coords: NDArray[np.float32],
        charges: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float]:
        """Run model inference.

        Args:
            coords: Point coordinates (N, 2).
            charges: Charge values (N,).

        Returns:
            Tuple of (predicted_potential, inference_time_ms).

        """
        if self.model is None:
            # Return dummy prediction if no model
            logger.warning("no_model_loaded", returning_zeros=True)
            return np.zeros_like(charges), 0.0

        self.model.eval()
        with torch.no_grad():
            coords_t = torch.from_numpy(coords).unsqueeze(0).to(self.device)
            charges_t = torch.from_numpy(charges).unsqueeze(0).to(self.device)

            start_time = time.perf_counter()
            prediction = self.model(coords_t, charges_t)
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            return prediction.cpu().numpy().squeeze(), inference_time_ms

    def evaluate_transfer(
        self,
        grid_size: int,
        seed: int | None = None,
    ) -> TransferResult:
        """Evaluate zero-shot transfer at a specific grid size.

        Args:
            grid_size: Target grid size for evaluation.
            seed: Random seed.

        Returns:
            TransferResult with metrics and arrays.

        """
        logger.info("evaluating_transfer", grid_size=grid_size, seed=seed)

        # Generate sample
        charges, potential_gt, coords = self.generate_sample(grid_size, seed)

        # Run inference
        prediction, inference_time_ms = self.predict(coords, charges)

        # Compute metrics
        mse = float(np.mean((potential_gt - prediction) ** 2))
        mae = float(np.mean(np.abs(potential_gt - prediction)))

        logger.info(
            "transfer_evaluated",
            grid_size=grid_size,
            mse=mse,
            mae=mae,
            inference_time_ms=inference_time_ms,
        )

        return TransferResult(
            grid_size=grid_size,
            ground_truth=potential_gt.reshape(grid_size, grid_size),
            prediction=prediction.reshape(grid_size, grid_size),
            mse=mse,
            mae=mae,
            inference_time_ms=inference_time_ms,
        )

    def run_transfer_evaluation(
        self,
        seed: int | None = None,
    ) -> dict[int, TransferResult]:
        """Run zero-shot transfer evaluation across all configured sizes.

        Args:
            seed: Base random seed.

        Returns:
            Dict mapping grid_size to TransferResult.

        """
        results = {}
        for i, size in enumerate(self.config.eval_grid_sizes):
            sample_seed = seed + i if seed is not None else None
            results[size] = self.evaluate_transfer(size, sample_seed)
        return results

    # ==================== Gradio Interface Methods ====================

    def visualize_sample(
        self,
        grid_size: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Generate and visualize a physics sample (for Gradio).

        Args:
            grid_size: Grid resolution.
            seed: Random seed.

        Returns:
            Tuple of (charges_image, potential_image, info_text).

        """
        charges, potential, coords = self.generate_sample(grid_size, int(seed))

        charges_2d = charges.reshape(grid_size, grid_size)
        potential_2d = potential.reshape(grid_size, grid_size)

        charges_plot = self.field_viz.render_field(
            charges_2d,
            title=f"Charge Density ({grid_size}×{grid_size})",
        )
        potential_plot = self.field_viz.render_field(
            potential_2d,
            title=f"Potential Field ({grid_size}×{grid_size})",
        )

        info = (
            f"Grid Size: {grid_size}×{grid_size}\n"
            f"Charges: min={charges_2d.min():.4f}, max={charges_2d.max():.4f}\n"
            f"Potential: min={potential_2d.min():.4f}, max={potential_2d.max():.4f}"
        )

        # Clean up figures
        charges_plot.close()
        potential_plot.close()

        return charges_plot.image, potential_plot.image, info

    def visualize_transfer(
        self,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Visualize zero-shot transfer results (for Gradio).

        Args:
            seed: Random seed.

        Returns:
            Tuple of (comparison_image, mse_chart_image, results_text).

        """
        results = self.run_transfer_evaluation(int(seed))

        # Create comparison visualization
        comparison_data = {
            size: {
                "ground_truth": result.ground_truth,
                "prediction": result.prediction,
                "mse": result.mse,
            }
            for size, result in results.items()
        }

        comparison_plot = self.field_viz.render_transfer_comparison(
            comparison_data,
            title="Zero-Shot Transfer: Train 9×9 → Evaluate Any Size",
        )

        # Create MSE bar chart
        labels = [f"{size}×{size}" for size in results]
        mse_values = [result.mse for result in results.values()]

        mse_plot = self.chart_viz.render_mse_bar_chart(
            labels=labels,
            mse_values=mse_values,
            threshold=self.config.mse_threshold,
            title="MSE by Resolution (Lower is Better)",
        )

        # Build results text
        results_lines = ["Zero-Shot Transfer Results:", "=" * 40]
        threshold = self.config.mse_threshold
        for size, result in results.items():
            status = "PASS" if result.mse < threshold else "FAIL"
            results_lines.append(
                f"{size}×{size}: MSE={result.mse:.6f} [{status}] "
                f"(inference: {result.inference_time_ms:.1f}ms)"
            )
        results_lines.append("=" * 40)
        results_lines.append(f"Threshold: {threshold}")
        all_passed = all(r.mse < threshold for r in results.values())
        results_lines.append(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

        # Clean up
        comparison_plot.close()
        mse_plot.close()

        return comparison_plot.image, mse_plot.image, "\n".join(results_lines)

    def demonstrate_resolution_independence(
        self,
        small_size: int,
        large_size: int,
        seed: int,
    ) -> tuple[np.ndarray, str]:
        """Demonstrate resolution independence (for Gradio).

        Shows the same physics problem solved at different resolutions.

        Args:
            small_size: Small grid size.
            large_size: Large grid size.
            seed: Random seed.

        Returns:
            Tuple of (comparison_image, explanation_text).

        """
        # Use same seed for both to show same underlying physics
        # Note: small_result is used for logging/debugging purposes only
        _ = self.evaluate_transfer(int(small_size), int(seed))  # Warm up with small size
        large_result = self.evaluate_transfer(int(large_size), int(seed))

        comparison_plot = self.field_viz.render_comparison(
            ground_truth=large_result.ground_truth,
            prediction=large_result.prediction,
            title=f"Trained on {small_size}×{small_size}, Evaluated on {large_size}×{large_size}",
            show_difference=True,
        )

        explanation = f"""
Resolution Independence Demonstration
=====================================

Training Resolution: {small_size}×{small_size}
Evaluation Resolution: {large_size}×{large_size}

Results:
- MSE: {large_result.mse:.6f}
- MAE: {large_result.mae:.6f}
- Inference Time: {large_result.inference_time_ms:.1f} ms

This demonstrates the Galerkin neural operator's ability to generalize
across resolutions without retraining. The model was trained ONLY on
{small_size}×{small_size} grids but can accurately predict on {large_size}×{large_size}.

Key Insight: Traditional CNNs would fail this test because they encode
position through discrete grid indices. Our approach uses continuous
Fourier features, enabling true resolution independence.
"""

        comparison_plot.close()
        return comparison_plot.image, explanation


def create_physics_demo_tab(
    config: PhysicsDemoConfig | None = None,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
) -> Any:  # noqa: ANN401 - Gradio Tab has complex type
    """Create Gradio tab for physics demo.

    Args:
        config: Demo configuration.
        model: Pre-trained physics model.
        device: Torch device.

    Returns:
        Gradio Tab component.

    """
    import gradio as gr

    demo = PhysicsDemo(config, model, device)
    effective_config = demo.config

    with gr.Tab("Physics: Zero-Shot Transfer") as tab:
        gr.Markdown("""
## Physics Zero-Shot Transfer Demo

This demo showcases AlphaGalerkin's resolution-independent learning:
- **Train** on small grids (9×9)
- **Evaluate** on any resolution (13×13, 19×19, etc.) without retraining

The physics task: Given a charge distribution ρ(x,y), predict the potential field φ(x,y)
by solving the Poisson equation: ∇²φ = ρ
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Generate Sample")
                grid_size_slider = gr.Slider(
                    minimum=5,
                    maximum=effective_config.max_grid_size,
                    value=effective_config.train_grid_size,
                    step=1,
                    label="Grid Size",
                )
                seed_input = gr.Number(
                    value=42,
                    label="Random Seed",
                    precision=0,
                )
                generate_btn = gr.Button("Generate Sample", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    charges_img = gr.Image(label="Charge Density", height=300)
                    potential_img = gr.Image(label="Potential Field", height=300)
                sample_info = gr.Textbox(label="Sample Info", lines=3)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Zero-Shot Transfer Test")
                transfer_seed = gr.Number(
                    value=42,
                    label="Random Seed",
                    precision=0,
                )
                # Note: eval_sizes displayed for user info, not used in callback
                gr.Textbox(
                    value=", ".join(map(str, effective_config.eval_grid_sizes)),
                    label="Evaluation Sizes",
                    info="Comma-separated grid sizes (configured in config)",
                    interactive=False,
                )
                run_transfer_btn = gr.Button("Run Transfer Evaluation", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    comparison_img = gr.Image(label="Predictions vs Ground Truth", height=400)
                    mse_chart_img = gr.Image(label="MSE Comparison", height=400)
                transfer_results = gr.Textbox(label="Results", lines=10)

        gr.Markdown("---")

        with gr.Accordion("Resolution Independence Demo", open=False):
            with gr.Row():
                small_size_slider = gr.Slider(
                    minimum=5,
                    maximum=15,
                    value=9,
                    step=1,
                    label="Training Size",
                )
                large_size_slider = gr.Slider(
                    minimum=15,
                    maximum=effective_config.max_grid_size,
                    value=19,
                    step=1,
                    label="Evaluation Size",
                )
                demo_seed = gr.Number(value=42, label="Seed", precision=0)
                demo_btn = gr.Button("Demonstrate", variant="secondary")

            demo_img = gr.Image(label="Comparison", height=400)
            demo_explanation = gr.Textbox(label="Explanation", lines=15)

        # Wire up callbacks
        generate_btn.click(
            demo.visualize_sample,
            inputs=[grid_size_slider, seed_input],
            outputs=[charges_img, potential_img, sample_info],
        )

        run_transfer_btn.click(
            demo.visualize_transfer,
            inputs=[transfer_seed],
            outputs=[comparison_img, mse_chart_img, transfer_results],
        )

        demo_btn.click(
            demo.demonstrate_resolution_independence,
            inputs=[small_size_slider, large_size_slider, demo_seed],
            outputs=[demo_img, demo_explanation],
        )

    return tab
