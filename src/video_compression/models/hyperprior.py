"""Hyperprior entropy model for learned compression.

Implements the scale hyperprior from Ballé et al. (2018):
    p(y) = product_i p(y_i | sigma_i)

where sigma_i are predicted by a hyperprior:
    z = h_a(y)  # Hyper-analysis
    sigma = h_s(z)  # Hyper-synthesis

The hyperprior z is encoded with a factorized prior.

References:
- Ballé et al., "Variational image compression with a scale hyperprior" (2018)
- Minnen et al., "Joint autoregressive and hierarchical priors" (2018)

"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from src.video_compression.config import EntropyConfig, EntropyModelType


class EntropyOutput(NamedTuple):
    """Output from entropy model forward pass."""

    y_hat: Tensor  # Quantized latent
    z_hat: Tensor  # Quantized hyperprior (if applicable)
    y_likelihoods: Tensor  # Likelihoods for rate estimation
    z_likelihoods: Tensor  # Hyperprior likelihoods
    rate: Tensor  # Estimated rate in bits


class FactorizedPrior(nn.Module):
    """Factorized entropy model with learned CDFs.

    Models the marginal distribution of each channel as:
        p(x_i) = c_i(x_i + 0.5) - c_i(x_i - 0.5)

    where c_i is a learned CDF parameterized by a neural network.

    Reference: Ballé et al., "End-to-end optimized image compression" (2017)
    """

    def __init__(
        self,
        channels: int,
        num_filters: int = 3,
        init_scale: float = 10.0,
    ) -> None:
        """Initialize factorized prior.

        Args:
            channels: Number of latent channels.
            num_filters: Number of filters in CDF network.
            init_scale: Initial scale for parameter initialization.

        """
        super().__init__()
        self.channels = channels
        self.init_scale = init_scale

        # CDF is parameterized as cumulative of softplus activations
        # H, a, b are per-channel parameters
        self.H = nn.Parameter(torch.zeros(channels, num_filters + 1))
        self.a = nn.Parameter(torch.zeros(channels, num_filters))
        self.b = nn.Parameter(torch.zeros(channels, num_filters))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for approximately uniform distribution."""
        # Initialize for standard Gaussian-like prior
        nn.init.uniform_(self.H, -0.5, 0.5)
        nn.init.uniform_(self.a, 0.0, 1.0)
        nn.init.uniform_(self.b, -0.5, 0.5)

    def forward(
        self,
        x: Float[Tensor, "batch channels ..."],
    ) -> tuple[Float[Tensor, "batch channels ..."], Float[Tensor, batch]]:
        """Compute likelihoods for input.

        Args:
            x: Input tensor (quantized).

        Returns:
            Tuple of (quantized, likelihoods).

        """
        # Compute CDF at x ± 0.5
        lower = self._cdf(x - 0.5)
        upper = self._cdf(x + 0.5)

        # Likelihood = P(lower < X <= upper)
        likelihood = upper - lower

        # Clamp for numerical stability
        likelihood = torch.clamp(likelihood, min=1e-9)

        # Total rate in bits (negative log likelihood)
        rate = -torch.log2(likelihood).sum(dim=tuple(range(1, likelihood.ndim)))

        return x, rate

    def _cdf(self, x: Float[Tensor, "batch channels ..."]) -> Tensor:
        """Compute CDF at given points.

        Args:
            x: Points to evaluate CDF.

        Returns:
            CDF values.

        """
        # Move to (-inf, inf) range with appropriate scaling
        x_scaled = x / self.init_scale

        # Cumulative softplus transform (Ballé parameterization)
        # CDF = sigmoid(H_0 + sum_k softplus(a_k * x + b_k) * H_{k+1})
        batch_shape = x.shape

        # Reshape for broadcasting: (1, C, 1, ...) for parameters
        param_shape = (1, self.channels) + (1,) * (len(batch_shape) - 2)

        # Compute softplus terms
        logits = self.H[:, 0].view(param_shape)  # Initial term

        for k in range(self.a.shape[1]):
            a_k = self.a[:, k].view(param_shape)
            b_k = self.b[:, k].view(param_shape)
            h_k = self.H[:, k + 1].view(param_shape)
            logits = logits + F.softplus(a_k * x_scaled + b_k) * h_k

        return torch.sigmoid(logits)


class GaussianConditional(nn.Module):
    """Gaussian conditional entropy model.

    Models each element as Gaussian:
        p(y_i | sigma_i) = N(y_i; 0, sigma_i^2)

    For quantized values, integrates over bin:
        P(y_i | sigma_i) = Phi((y_i + 0.5) / sigma_i) - Phi((y_i - 0.5) / sigma_i)
    """

    def __init__(
        self,
        scale_bound: float = 0.11,
    ) -> None:
        """Initialize Gaussian conditional.

        Args:
            scale_bound: Minimum scale value for numerical stability.

        """
        super().__init__()
        self.scale_bound = scale_bound

    def forward(
        self,
        x: Float[Tensor, "batch channels ..."],
        scales: Float[Tensor, "batch channels ..."],
        means: Float[Tensor, "batch channels ..."] | None = None,
    ) -> tuple[Float[Tensor, "batch channels ..."], Float[Tensor, batch]]:
        """Compute likelihoods for Gaussian conditional.

        Args:
            x: Quantized values.
            scales: Predicted standard deviations.
            means: Predicted means (default 0).

        Returns:
            Tuple of (values, rate in bits).

        """
        if means is not None:
            x = x - means

        # Clamp scales for stability
        scales = torch.clamp(scales, min=self.scale_bound)

        # Compute CDF at x ± 0.5
        upper = self._standardized_cdf((x + 0.5) / scales)
        lower = self._standardized_cdf((x - 0.5) / scales)

        # Likelihood
        likelihood = upper - lower
        likelihood = torch.clamp(likelihood, min=1e-9)

        # Rate in bits
        rate = -torch.log2(likelihood).sum(dim=tuple(range(1, likelihood.ndim)))

        if means is not None:
            x = x + means

        return x, rate

    def _standardized_cdf(self, x: Tensor) -> Tensor:
        """Standard normal CDF using error function.

        Args:
            x: Input values.

        Returns:
            CDF values.

        """
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class HyperAnalysis(nn.Module):
    """Hyper-analysis transform for hyperprior.

    Extracts hyperprior z from latent y:
        z = h_a(|y|)

    Uses absolute value to model only the scale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 3,
    ) -> None:
        """Initialize hyper-analysis.

        Args:
            in_channels: Input latent channels.
            out_channels: Hyperprior channels.
            n_layers: Number of layers.

        """
        super().__init__()

        layers = []
        ch = in_channels

        for i in range(n_layers):
            out_ch = out_channels if i == n_layers - 1 else in_channels
            layers.extend(
                [
                    nn.Conv2d(ch, out_ch, 3, stride=2 if i < n_layers - 1 else 1, padding=1),
                    nn.LeakyReLU(0.2) if i < n_layers - 1 else nn.Identity(),
                ]
            )
            ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        y: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch hyper_ch h w"]:
        """Extract hyperprior.

        Args:
            y: Latent tensor.

        Returns:
            Hyperprior tensor.

        """
        # Use absolute value to model scale
        return self.net(torch.abs(y))


class HyperSynthesis(nn.Module):
    """Hyper-synthesis transform for hyperprior.

    Predicts scale parameters from hyperprior:
        sigma = h_s(z)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 3,
    ) -> None:
        """Initialize hyper-synthesis.

        Args:
            in_channels: Hyperprior channels.
            out_channels: Output channels (latent channels).
            n_layers: Number of layers.

        """
        super().__init__()

        layers = []
        ch = in_channels

        for i in range(n_layers):
            out_ch = out_channels if i == n_layers - 1 else in_channels * 2
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        ch,
                        out_ch,
                        3,
                        stride=2 if i < n_layers - 1 else 1,
                        padding=1,
                        output_padding=1 if i < n_layers - 1 else 0,
                    ),
                    nn.LeakyReLU(0.2) if i < n_layers - 1 else nn.Identity(),
                ]
            )
            ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: Float[Tensor, "batch hyper_ch h w"],
    ) -> Float[Tensor, "batch channels height width"]:
        """Predict scale parameters.

        Args:
            z: Hyperprior tensor.

        Returns:
            Scale tensor (same spatial size as latent).

        """
        # Output is log-scale, exp for positivity
        return torch.exp(self.net(z))


class HyperpriorEntropyModel(nn.Module):
    """Scale hyperprior entropy model.

    Architecture:
        y -> HyperAnalysis -> z -> Quantize -> HyperSynthesis -> sigma
        y ~ N(0, sigma^2)

    The hyperprior z is encoded with a factorized prior.
    The main latent y is encoded with a Gaussian conditional given sigma.
    """

    def __init__(self, config: EntropyConfig) -> None:
        """Initialize hyperprior model.

        Args:
            config: Entropy model configuration.

        """
        super().__init__()
        self.config = config

        # Hyper-analysis and synthesis transforms
        self.hyper_analysis = HyperAnalysis(
            in_channels=config.num_filters,
            out_channels=config.hyper_channels,
            n_layers=config.hyper_layers,
        )
        self.hyper_synthesis = HyperSynthesis(
            in_channels=config.hyper_channels,
            out_channels=config.num_filters,
            n_layers=config.hyper_layers,
        )

        # Entropy models
        self.hyperprior = FactorizedPrior(
            channels=config.hyper_channels,
            num_filters=3,
        )
        self.gaussian = GaussianConditional()

    def forward(
        self,
        y: Float[Tensor, "batch channels height width"],
        training: bool | None = None,
    ) -> EntropyOutput:
        """Compute entropy model outputs.

        Args:
            y: Latent tensor (before quantization).
            training: Override training mode.

        Returns:
            EntropyOutput with quantized values and rates.

        """
        is_training = training if training is not None else self.training

        # Hyper-analysis
        z = self.hyper_analysis(y)

        # Quantize hyperprior
        z_hat = z + torch.empty_like(z).uniform_(-0.5, 0.5) if is_training else torch.round(z)

        # Hyper-synthesis: predict scales
        scales = self.hyper_synthesis(z_hat)

        # Resize scales to match y if needed
        if scales.shape != y.shape:
            scales = F.interpolate(scales, size=y.shape[-2:], mode="bilinear", align_corners=False)

        # Quantize latent
        y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5) if is_training else torch.round(y)

        # Compute likelihoods
        _, y_rate = self.gaussian(y_hat, scales)
        _, z_rate = self.hyperprior(z_hat)

        # Total rate
        total_rate = y_rate + z_rate

        # Compute likelihoods for backwards compatibility
        y_likelihoods = torch.exp(-y_rate / (y.shape[1] * y.shape[2] * y.shape[3]))
        z_likelihoods = torch.exp(-z_rate / (z.shape[1] * z.shape[2] * z.shape[3]))

        return EntropyOutput(
            y_hat=y_hat,
            z_hat=z_hat,
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
            rate=total_rate,
        )

    def compress(
        self,
        y: Float[Tensor, "batch channels height width"],
    ) -> dict[str, Tensor]:
        """Compress latent to bitstream (returns quantized symbols).

        Args:
            y: Latent tensor.

        Returns:
            Dictionary with quantized symbols for entropy coding.

        """
        # Hyper-analysis and quantization
        z = self.hyper_analysis(y)
        z_hat = torch.round(z)

        # Predict scales
        scales = self.hyper_synthesis(z_hat)
        if scales.shape != y.shape:
            scales = F.interpolate(scales, size=y.shape[-2:], mode="bilinear", align_corners=False)

        # Quantize latent
        y_hat = torch.round(y)

        return {
            "y_symbols": y_hat.to(torch.int16),
            "z_symbols": z_hat.to(torch.int16),
            "scales": scales,
        }

    def decompress(
        self,
        y_symbols: Tensor,
        z_symbols: Tensor,
    ) -> Float[Tensor, "batch channels height width"]:
        """Decompress from quantized symbols.

        Args:
            y_symbols: Quantized latent symbols.
            z_symbols: Quantized hyperprior symbols.

        Returns:
            Reconstructed latent tensor.

        """
        return y_symbols.float()


def create_entropy_model(config: EntropyConfig) -> nn.Module:
    """Factory function to create entropy model from config.

    Args:
        config: Entropy model configuration.

    Returns:
        Configured entropy model instance.

    """
    match config.model_type:
        case EntropyModelType.FACTORIZED:
            return FactorizedPrior(
                channels=config.num_filters,
                num_filters=3,
            )
        case EntropyModelType.HYPERPRIOR:
            return HyperpriorEntropyModel(config)
        case EntropyModelType.AUTOREGRESSIVE:
            # Autoregressive is complex, return hyperprior as fallback
            return HyperpriorEntropyModel(config)
        case _:
            raise ValueError(f"Unknown entropy model type: {config.model_type}")
