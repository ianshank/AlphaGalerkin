"""FNet mixing blocks for high-speed MCTS rollouts.

FNet replaces attention with FFT-based mixing:
    FFT -> Mix in frequency domain -> iFFT

This provides O(N log N) complexity and can be significantly faster
than attention for MCTS rollout evaluation.

Reference: "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., 2021)
"""

from __future__ import annotations

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class FNetMixing(nn.Module):
    """FNet mixing layer using FFT.

    Applies 2D real FFT to mix information across spatial positions.
    Uses rfft2 for efficiency on real-valued inputs.
    """

    def __init__(
        self,
        use_2d: bool = True,
    ) -> None:
        """Initialize FNet mixing.

        Args:
            use_2d: Use 2D FFT (for board-like data) or 1D (for sequences).

        """
        super().__init__()
        self.use_2d = use_2d

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        board_size: int | None = None,
    ) -> Float[Tensor, "batch n d"]:
        """Apply FFT mixing.

        Args:
            x: Input tensor (batch, seq_len, features).
            board_size: Board size for 2D FFT reshaping.

        Returns:
            Mixed tensor.

        """
        if self.use_2d and board_size is not None:
            return self._mix_2d(x, board_size)
        else:
            return self._mix_1d(x)

    def _mix_1d(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Apply 1D FFT mixing along sequence dimension.

        Args:
            x: Input tensor.

        Returns:
            Mixed tensor.

        """
        # FFT along sequence dimension
        x_freq = torch.fft.rfft(x, dim=1)

        # Take real part (mix in frequency domain)
        # The real part corresponds to symmetric components
        x_mixed = x_freq.real

        # Pad or truncate to match original sequence length
        n = x.shape[1]
        n_freq = x_mixed.shape[1]

        if n_freq < n:
            # Pad with zeros
            padding = torch.zeros(
                x.shape[0], n - n_freq, x.shape[2],
                device=x.device, dtype=x.dtype
            )
            x_mixed = torch.cat([x_mixed, padding], dim=1)
        elif n_freq > n:
            # Truncate
            x_mixed = x_mixed[:, :n, :]

        return x_mixed

    def _mix_2d(
        self,
        x: Float[Tensor, "batch n d"],
        board_size: int,
    ) -> Float[Tensor, "batch n d"]:
        """Apply 2D FFT mixing for board-like data.

        Args:
            x: Input tensor.
            board_size: Board size (assumes square board).

        Returns:
            Mixed tensor.

        """
        batch, n, d = x.shape

        # Reshape to 2D spatial format
        x_2d = rearrange(x, "b (h w) d -> b d h w", h=board_size, w=board_size)

        # Apply 2D real FFT
        x_freq = torch.fft.rfft2(x_2d)

        # Take real part for mixing
        x_mixed_freq = x_freq.real

        # Inverse FFT to get back to spatial domain
        # Note: irfft2 needs the original spatial size
        x_mixed = torch.fft.irfft2(
            x_mixed_freq.to(torch.complex64),
            s=(board_size, board_size)
        )

        # Reshape back to sequence format
        x_out = rearrange(x_mixed, "b d h w -> b (h w) d")

        return x_out


class FNetBlock(nn.Module):
    """FNet Transformer block with FFT mixing.

    Structure:
        x -> LayerNorm -> FFT Mixing -> Residual
        x -> LayerNorm -> FFN -> Residual
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int | None = None,
        dropout: float = 0.1,
        use_2d_fft: bool = True,
    ) -> None:
        """Initialize FNet block.

        Args:
            d_model: Model dimension.
            d_ffn: Feed-forward network dimension (default: 4 * d_model).
            dropout: Dropout rate.
            use_2d_fft: Use 2D FFT for spatial mixing.

        """
        super().__init__()
        self.d_model = d_model
        d_ffn = d_ffn or 4 * d_model

        # FFT mixing
        self.fft_mixing = FNetMixing(use_2d=use_2d_fft)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        board_size: int | None = None,
    ) -> Float[Tensor, "batch n d"]:
        """Forward pass through FNet block.

        Args:
            x: Input tensor.
            board_size: Board size for 2D FFT.

        Returns:
            Output tensor.

        """
        # FFT mixing with residual
        x_norm = self.norm1(x)
        x_mixed = self.fft_mixing(x_norm, board_size)
        x = x + self.dropout(x_mixed)

        # FFN with residual
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn

        return x


class FNetStack(nn.Module):
    """Stack of FNet blocks for fast feature processing.

    Used in MCTS rollouts where speed is critical.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_ffn: int | None = None,
        dropout: float = 0.1,
        use_2d_fft: bool = True,
    ) -> None:
        """Initialize FNet stack.

        Args:
            d_model: Model dimension.
            n_layers: Number of FNet blocks.
            d_ffn: Feed-forward dimension.
            dropout: Dropout rate.
            use_2d_fft: Use 2D FFT for spatial mixing.

        """
        super().__init__()
        self.layers = nn.ModuleList([
            FNetBlock(d_model, d_ffn, dropout, use_2d_fft)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        board_size: int | None = None,
    ) -> Float[Tensor, "batch n d"]:
        """Forward pass through FNet stack.

        Args:
            x: Input tensor.
            board_size: Board size for 2D FFT.

        Returns:
            Output tensor.

        """
        for layer in self.layers:
            x = layer(x, board_size)
        return x


class GalerkinFNetHybrid(nn.Module):
    """Hybrid layer combining Galerkin attention with FNet mixing.

    Uses Galerkin attention for global operator approximation
    and FNet for fast local mixing. This provides:
    - Mathematical grounding (Galerkin)
    - Speed (FNet)
    - Resolution independence (both)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int | None = None,
        dropout: float = 0.1,
        fnet_ratio: float = 0.5,
    ) -> None:
        """Initialize hybrid layer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ffn: Feed-forward dimension.
            dropout: Dropout rate.
            fnet_ratio: Ratio of FNet to Galerkin output.

        """
        super().__init__()

        # Import here to avoid circular dependency
        from src.modeling.attention import GalerkinAttention

        self.galerkin = GalerkinAttention(d_model, n_heads, dropout=dropout)
        self.fnet = FNetBlock(d_model, d_ffn, dropout)

        # Learnable mixing ratio
        self.mix_ratio = nn.Parameter(torch.tensor(fnet_ratio))

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        board_size: int | None = None,
    ) -> Float[Tensor, "batch n d"]:
        """Forward pass through hybrid layer.

        Args:
            x: Input tensor.
            board_size: Board size for 2D FFT.

        Returns:
            Combined output.

        """
        # Apply both pathways
        galerkin_out = self.galerkin(x)
        fnet_out = self.fnet(x, board_size)

        # Mix outputs
        ratio = torch.sigmoid(self.mix_ratio)
        output = (1 - ratio) * galerkin_out + ratio * fnet_out

        return self.norm(output)
