"""Architecture Visualization Demo for AlphaGalerkin.

Visualizes the key architectural components:
- Galerkin vs Softmax attention patterns
- Fourier feature embeddings
- LBB stability condition monitoring
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.demos.config import ArchitectureDemoConfig, ColorScheme, VisualizationConfig
from src.demos.visualizations import (
    AttentionVisualizer,
    ChartVisualizer,
    figure_to_image,
    get_colormap,
    PlotResult,
)

# Module-level logger
import structlog

logger = structlog.get_logger(__name__)


class ArchitectureDemo:
    """Interactive demo for architecture visualization.

    Provides:
    1. Attention pattern visualization (Galerkin vs Softmax)
    2. Fourier feature frequency spectrum
    3. LBB stability condition monitoring
    4. Resolution independence explanation
    """

    def __init__(
        self,
        config: ArchitectureDemoConfig | None = None,
        model: nn.Module | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize architecture demo.

        Args:
            config: Demo configuration.
            model: Pre-trained AlphaGalerkin model (optional).
            device: Torch device.

        """
        self.config = config or ArchitectureDemoConfig()
        self.model = model
        self.device = device

        self.attention_viz = AttentionVisualizer(self.config.visualization)
        self.chart_viz = ChartVisualizer(self.config.visualization)

        logger.info(
            "architecture_demo_initialized",
            has_model=model is not None,
            device=device,
            sample_board_size=self.config.sample_board_size,
        )

    def visualize_fourier_features(
        self,
        n_features: int = 64,
        scale: float = 1.0,
        grid_size: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Visualize Fourier feature embeddings.

        Args:
            n_features: Number of Fourier features.
            scale: Frequency scale.
            grid_size: Resolution for visualization.

        Returns:
            Tuple of (2d_embedding_plot, frequency_spectrum, explanation).

        """
        # Create coordinate grid
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (N, 2)

        # Generate Fourier features
        # Using simple random Fourier features for visualization
        torch.manual_seed(42)
        B = torch.randn(2, n_features) * scale * 2 * np.pi

        # Compute features: [sin(Bx), cos(Bx)]
        proj = coords @ B  # (N, n_features)
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*n_features)

        # Visualize first few feature dimensions as 2D fields
        fig, axes = plt.subplots(
            2, 4,
            figsize=(self.config.visualization.figure_width * 1.5,
                     self.config.visualization.figure_height),
            dpi=self.config.visualization.dpi,
        )

        cmap = get_colormap(self.config.visualization.color_scheme)

        for i, ax in enumerate(axes.flat):
            if i < features.shape[1]:
                field = features[:, i].reshape(grid_size, grid_size).numpy()
                ax.imshow(field, cmap=cmap, origin="lower")
                ax.set_title(f"Feature {i}", fontsize=8)
                ax.axis("off")

        plt.suptitle(f"Fourier Feature Visualization (scale={scale})", fontsize=12)
        plt.tight_layout()

        embedding_img = figure_to_image(fig)
        plt.close(fig)

        # Frequency spectrum
        frequencies = torch.sqrt((B ** 2).sum(dim=0)).numpy()

        spectrum_plot = self.chart_viz.render_fourier_spectrum(
            frequencies=frequencies,
            amplitudes=np.ones_like(frequencies),
            title=f"Fourier Feature Frequencies (n={n_features}, scale={scale})",
        )

        explanation = f"""
Fourier Feature Embeddings
==========================

Configuration:
- Number of features: {n_features}
- Frequency scale: {scale}
- Output dimension: {features.shape[1]}

Key Properties:
1. **High-frequency encoding**: Enables learning of sharp features
2. **Position-agnostic**: Same embedding for same normalized position
3. **Resolution independence**: Works for any grid size

The embedding maps (x, y) ∈ [0,1]² to a high-dimensional space:
γ(x, y) = [sin(2πBx), cos(2πBx), sin(2πBy), cos(2πBy), ...]

where B contains random frequencies sampled at scale {scale}.

This allows the model to learn patterns at multiple spatial scales
without being tied to discrete grid positions.
"""

        spectrum_plot.close()
        return embedding_img, spectrum_plot.image, explanation

    def visualize_attention_patterns(
        self,
        board_size: int = 9,
        n_heads: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Visualize Galerkin vs Softmax attention patterns.

        Args:
            board_size: Board size for visualization.
            n_heads: Number of attention heads.

        Returns:
            Tuple of (galerkin_plot, softmax_plot, comparison_text).

        """
        seq_length = board_size * board_size
        d_model = 64
        d_head = d_model // n_heads

        # Create random input (simulating board features)
        torch.manual_seed(42)
        x = torch.randn(1, seq_length, d_model)

        # Compute attention patterns
        # Galerkin: Q(K^T V) - no softmax
        q = x.view(1, seq_length, n_heads, d_head).transpose(1, 2)
        k = x.view(1, seq_length, n_heads, d_head).transpose(1, 2)

        # Galerkin attention (without softmax)
        galerkin_attn = (q @ k.transpose(-2, -1)) / seq_length
        galerkin_attn = galerkin_attn.abs()  # For visualization
        galerkin_attn = galerkin_attn / (galerkin_attn.max() + 1e-8)

        # Softmax attention
        softmax_attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(d_head), dim=-1)

        # Visualize
        comparison_plot = self.attention_viz.render_galerkin_vs_softmax(
            galerkin_attn=galerkin_attn.squeeze(0).numpy(),
            softmax_attn=softmax_attn.squeeze(0).numpy(),
            title=f"Attention Patterns ({board_size}×{board_size} board)",
        )

        # Individual head visualizations
        fig, axes = plt.subplots(
            2, n_heads,
            figsize=(self.config.visualization.figure_width * 1.5,
                     self.config.visualization.figure_height),
            dpi=self.config.visualization.dpi,
        )

        cmap = get_colormap(ColorScheme.VIRIDIS)

        for h in range(n_heads):
            # Galerkin (top row)
            axes[0, h].imshow(galerkin_attn[0, h].numpy(), cmap=cmap, aspect="auto")
            axes[0, h].set_title(f"Galerkin H{h}", fontsize=8)
            axes[0, h].axis("off")

            # Softmax (bottom row)
            axes[1, h].imshow(softmax_attn[0, h].numpy(), cmap=cmap, aspect="auto")
            axes[1, h].set_title(f"Softmax H{h}", fontsize=8)
            axes[1, h].axis("off")

        plt.suptitle("Per-Head Attention Patterns", fontsize=12)
        plt.tight_layout()

        heads_img = figure_to_image(fig)
        plt.close(fig)

        # Compute entropy metrics
        galerkin_entropy = float(
            -torch.sum(galerkin_attn * torch.log(galerkin_attn + 1e-10), dim=-1).mean()
        )
        softmax_entropy = float(
            -torch.sum(softmax_attn * torch.log(softmax_attn + 1e-10), dim=-1).mean()
        )

        comparison_text = f"""
Attention Pattern Analysis
==========================

Board Size: {board_size}×{board_size} (N={seq_length})
Number of Heads: {n_heads}

Galerkin Attention:
- Complexity: O(N) via Monte Carlo approximation
- Formula: Attention = Q(K^T V) / N
- Entropy: {galerkin_entropy:.4f}
- Pattern: More diffuse, captures global influence

Softmax Attention:
- Complexity: O(N²) from attention matrix
- Formula: Attention = softmax(QK^T / √d) V
- Entropy: {softmax_entropy:.4f}
- Pattern: More peaked, focuses on local patterns

Key Insight: Galerkin attention distributes attention more evenly,
acting as a global influence propagator (like convolution).
Softmax attention concentrates on specific positions,
better for precise local reading (tactical calculations).

AlphaGalerkin Strategy:
- Strategy Body: 6 Galerkin layers (global influence modeling)
- Tactical Head: 2 Softmax layers (precise reading)
"""

        comparison_plot.close()
        return heads_img, comparison_plot.image, comparison_text

    def visualize_lbb_stability(
        self,
        n_samples: int = 100,
    ) -> tuple[np.ndarray, str]:
        """Visualize LBB stability condition.

        The LBB (Ladyzhenskaya-Babuška-Brezzi) condition ensures the
        Galerkin projection is well-posed: sigma_min > beta > 0.

        Args:
            n_samples: Number of random matrices to sample.

        Returns:
            Tuple of (stability_plot, explanation).

        """
        # Simulate LBB stability over training
        torch.manual_seed(42)

        # Simulate singular value evolution during training
        steps = np.arange(n_samples)
        d_key, d_value = 64, 64

        # Simulate improving stability during training
        # Initially close to threshold, then stabilizing
        sigma_mins = []
        sigma_maxs = []

        for step in steps:
            # Simulated Key-Value projection matrix
            noise_scale = 1.0 / (1 + 0.1 * step)
            W = torch.randn(d_key, d_value) * noise_scale + torch.eye(d_key, d_value) * 0.5

            # Compute singular values
            svd = torch.linalg.svdvals(W)
            sigma_mins.append(float(svd.min()))
            sigma_maxs.append(float(svd.max()))

        # Visualization
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(self.config.visualization.figure_width * 1.5,
                     self.config.visualization.figure_height),
            dpi=self.config.visualization.dpi,
        )

        # Singular value evolution
        ax1.plot(steps, sigma_mins, label="σ_min", color="#e74c3c", linewidth=2)
        ax1.plot(steps, sigma_maxs, label="σ_max", color="#2ecc71", linewidth=2)
        ax1.axhline(y=1e-6, color="gray", linestyle="--", label="β threshold")
        ax1.fill_between(steps, 0, sigma_mins, alpha=0.3, color="#e74c3c")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Singular Value")
        ax1.set_title("LBB Stability: σ_min > β")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Condition number
        condition_numbers = [smax / (smin + 1e-10) for smax, smin in zip(sigma_maxs, sigma_mins)]
        ax2.semilogy(steps, condition_numbers, color="#3498db", linewidth=2)
        ax2.axhline(y=100, color="orange", linestyle="--", label="Good threshold")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Condition Number (log scale)")
        ax2.set_title("Matrix Conditioning")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        stability_img = figure_to_image(fig)
        plt.close(fig)

        final_sigma_min = sigma_mins[-1]
        final_condition = condition_numbers[-1]

        explanation = f"""
LBB Stability Condition
=======================

The Ladyzhenskaya-Babuška-Brezzi (LBB) condition ensures numerical
stability of the Galerkin projection:

    σ_min(K→V) > β > 0

where K→V is the Key-to-Value projection matrix.

Current Status:
- Minimum singular value: {final_sigma_min:.6f}
- Condition number: {final_condition:.2f}
- Stability threshold: 1e-6

LBB Constraint in AlphaGalerkin:
1. dim(Key) >= dim(Query) ensures inf-sup condition
2. Regularization term in loss: λ * max(0, β - σ_min)²
3. StabilityGuard monitors σ_min during training

Why It Matters:
- Without LBB satisfaction, Galerkin projection can become
  numerically unstable (spurious oscillations, divergence)
- Ensures the learned operators are well-posed PDEs
- Critical for physics-informed learning

Mathematical Background:
For the Petrov-Galerkin method, we solve: Find u ∈ V such that
    b(u, v) = f(v)  ∀v ∈ W

The inf-sup (LBB) condition:
    inf_(u∈V) sup_(v∈W) b(u,v) / (||u|| ||v||) >= β > 0

guarantees existence, uniqueness, and stability of the solution.
"""

        return stability_img, explanation

    def visualize_architecture_overview(self) -> tuple[np.ndarray, str]:
        """Create architecture overview diagram.

        Returns:
            Tuple of (architecture_diagram, description).

        """
        # Create a simple architecture diagram
        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure_width,
                     self.config.visualization.figure_height * 1.5),
            dpi=self.config.visualization.dpi,
        )

        # Hide axes
        ax.axis("off")

        # Draw architecture blocks
        blocks = [
            {"y": 0.9, "text": "Input: Board State", "color": "#3498db"},
            {"y": 0.8, "text": "Continuous Embedding\n(Fourier Features)", "color": "#9b59b6"},
            {"y": 0.65, "text": "Strategy Body\n(6× Galerkin Attention + FNet)\nO(N) complexity", "color": "#2ecc71"},
            {"y": 0.45, "text": "Tactical Head\n(2× Softmax Attention)\nO(N²) for precision", "color": "#e74c3c"},
            {"y": 0.25, "text": "Policy Head\n(Move Probabilities)", "color": "#f39c12"},
            {"y": 0.1, "text": "Value Head\n(Win Probability)", "color": "#f39c12"},
        ]

        for block in blocks:
            rect = plt.Rectangle(
                (0.2, block["y"] - 0.05), 0.6, 0.08,
                facecolor=block["color"],
                edgecolor="black",
                linewidth=2,
                alpha=0.7,
            )
            ax.add_patch(rect)
            ax.text(0.5, block["y"], block["text"], ha="center", va="center", fontsize=10)

        # Draw arrows
        for i in range(len(blocks) - 1):
            ax.annotate(
                "",
                xy=(0.5, blocks[i + 1]["y"] + 0.05),
                xytext=(0.5, blocks[i]["y"] - 0.05),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )

        # Add title
        ax.text(0.5, 0.98, "AlphaGalerkin Architecture", ha="center", va="top", fontsize=14, fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        architecture_img = figure_to_image(fig)
        plt.close(fig)

        description = """
AlphaGalerkin Architecture Overview
===================================

The architecture combines ideas from:
1. AlphaZero (MCTS + Neural Network)
2. Neural Operators (Galerkin attention)
3. FNet (FFT-based mixing)

Key Innovations:

1. **Resolution Independence**
   - Uses Fourier feature encoding instead of positional embedding
   - Coordinates normalized to [0,1]² domain
   - Same model works for any board size

2. **Dual Attention Strategy**
   - Strategy Body: Galerkin attention for global influence
   - Tactical Head: Softmax attention for precise reading
   - Balances efficiency with accuracy

3. **FNet Acceleration**
   - FFT-based mixing layers for O(N log N) complexity
   - Enables faster MCTS rollouts
   - Interleaved with Galerkin layers

4. **Mathematical Grounding**
   - Based on Petrov-Galerkin projection theory
   - LBB stability condition enforced during training
   - Monte Carlo integral normalization (1/N)

Training Objective:
L = L_policy + L_value + λ * L_LBB
"""

        return architecture_img, description


def create_architecture_demo_tab(
    config: ArchitectureDemoConfig | None = None,
    model: nn.Module | None = None,
    device: str = "cpu",
) -> Any:
    """Create Gradio tab for architecture demo.

    Args:
        config: Demo configuration.
        model: Pre-trained model.
        device: Torch device.

    Returns:
        Gradio Tab component.

    """
    import gradio as gr

    demo = ArchitectureDemo(config, model, device)

    with gr.Tab("Architecture Visualization") as tab:
        gr.Markdown("""
## Architecture Visualization

Explore the key components of AlphaGalerkin's neural operator architecture:
- **Fourier Features**: Resolution-independent positional encoding
- **Attention Patterns**: Galerkin vs Softmax comparison
- **LBB Stability**: Mathematical stability monitoring
        """)

        with gr.Accordion("Architecture Overview", open=True):
            overview_btn = gr.Button("Show Architecture Diagram", variant="primary")
            with gr.Row():
                arch_diagram = gr.Image(label="Architecture", height=500)
                arch_description = gr.Textbox(label="Description", lines=25)

        with gr.Accordion("Fourier Feature Embeddings", open=False):
            gr.Markdown("Visualize how Fourier features encode spatial positions.")
            with gr.Row():
                n_features_slider = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=64,
                    step=8,
                    label="Number of Features",
                )
                scale_slider = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    label="Frequency Scale",
                )
                fourier_btn = gr.Button("Visualize", variant="secondary")

            with gr.Row():
                fourier_2d_img = gr.Image(label="2D Feature Maps", height=300)
                fourier_spectrum_img = gr.Image(label="Frequency Spectrum", height=300)
            fourier_explanation = gr.Textbox(label="Explanation", lines=15)

        with gr.Accordion("Attention Patterns", open=False):
            gr.Markdown("Compare Galerkin (O(N)) and Softmax (O(N²)) attention patterns.")
            with gr.Row():
                attn_board_size = gr.Slider(
                    minimum=5,
                    maximum=13,
                    value=9,
                    step=1,
                    label="Board Size",
                )
                attn_n_heads = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Number of Heads",
                )
                attn_btn = gr.Button("Compare", variant="secondary")

            with gr.Row():
                heads_img = gr.Image(label="Per-Head Patterns", height=300)
                comparison_img = gr.Image(label="Galerkin vs Softmax", height=300)
            attn_text = gr.Textbox(label="Analysis", lines=20)

        with gr.Accordion("LBB Stability Condition", open=False):
            gr.Markdown("""
The LBB condition ensures numerical stability of Galerkin methods.
AlphaGalerkin monitors and enforces this during training.
            """)
            lbb_btn = gr.Button("Visualize Stability", variant="secondary")
            lbb_img = gr.Image(label="Stability Metrics", height=400)
            lbb_explanation = gr.Textbox(label="Explanation", lines=20)

        # Wire up callbacks
        overview_btn.click(
            demo.visualize_architecture_overview,
            inputs=[],
            outputs=[arch_diagram, arch_description],
        )

        fourier_btn.click(
            demo.visualize_fourier_features,
            inputs=[n_features_slider, scale_slider],
            outputs=[fourier_2d_img, fourier_spectrum_img, fourier_explanation],
        )

        attn_btn.click(
            demo.visualize_attention_patterns,
            inputs=[attn_board_size, attn_n_heads],
            outputs=[heads_img, comparison_img, attn_text],
        )

        lbb_btn.click(
            demo.visualize_lbb_stability,
            inputs=[],
            outputs=[lbb_img, lbb_explanation],
        )

    return tab
