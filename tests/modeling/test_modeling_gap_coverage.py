"""Targeted gap-coverage tests for modeling defensive branches.

Exercises defensive branches and new constructor parameters added
during the Galerkin Fusion Head preparation work so the modeling
modules keep passing the 85% coverage gate as the API surface grows.

The ADR re-export + signature contract is enforced in a separate file,
``tests/modeling/test_public_surface_contract.py``; the ``__all__``
parity check below is retained here for defence-in-depth.

No hardcoded shape/dtype assumptions: every dimension referenced is
derived from a parameter and could be re-parametrized without rewriting
the test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.modeling import (
    AdaptiveFourierFeatures,
    AlphaGalerkinModel,
    ContinuousEmbedding,
    FNetBlock,
    FNetMixing,
    FNetStack,
    FourierFeatures,
    FourierFeaturesConfig,
    GalerkinAttention,
    GalerkinFNetHybrid,
    HybridAttention,
    MultiScaleFourierFeatures,
    PositionalEncoding,
    ProgressiveFourierFeatures,
    SoftmaxAttention,
    SpatialPositionalEncoding,
    StabilityGuard,
    StableGalerkinInitializer,
)

# Submodule imports only for module-level constants (DEFAULT_FFN_EXPANSION,
# DEFAULT_SCALES) that are not part of the re-export class surface. All class
# imports use the stable top-level `src.modeling` path per the integration ADR.
from src.modeling import fnet as fnet_module
from src.modeling import multiscale_fourier as mf_module

# ---------------------------------------------------------------------------
# Re-export surface (ADR-mouse-droid-fusion-integration)
# ---------------------------------------------------------------------------


_PUBLIC_SURFACE: tuple[type, ...] = (
    GalerkinAttention,
    SoftmaxAttention,
    HybridAttention,
    FNetBlock,
    FNetMixing,
    FNetStack,
    GalerkinFNetHybrid,
    MultiScaleFourierFeatures,
    AdaptiveFourierFeatures,
    ProgressiveFourierFeatures,
    PositionalEncoding,
    SpatialPositionalEncoding,
    FourierFeaturesConfig,
    StabilityGuard,
    StableGalerkinInitializer,
    AlphaGalerkinModel,
    ContinuousEmbedding,
    FourierFeatures,
)


@pytest.mark.parametrize("cls", _PUBLIC_SURFACE, ids=lambda c: c.__name__)
def test_class_is_in_public_all(cls: type) -> None:
    """Every documented public class must appear in ``__all__``."""
    import src.modeling as mod

    assert cls.__name__ in mod.__all__, (
        f"{cls.__name__} must be re-exported via src.modeling.__all__ "
        "per docs/adr/0002-mouse-droid-fusion-integration.md"
    )


# ---------------------------------------------------------------------------
# MultiScaleFourierFeatures: config-driven init + defensive branches
# ---------------------------------------------------------------------------


class TestMultiScaleFourierFeaturesGapCoverage:
    """Closes coverage holes around config init, non-learnable mode, errors."""

    @pytest.fixture
    def input_dim(self) -> int:
        return 3

    def test_init_via_config_overrides_kwargs(self, input_dim: int) -> None:
        config = FourierFeaturesConfig(
            name="cfg",
            n_features=16,
            scales=[0.5, 2.0],
            learnable=False,
            include_input=False,
        )

        # kwargs that conflict with config must be silently overridden
        layer = MultiScaleFourierFeatures(
            input_dim=input_dim,
            config=config,
            n_features=99,
            scales=[1.0, 1.0, 1.0],
            learnable=True,
            include_input=True,
        )

        assert layer.n_features == config.n_features
        assert layer.scales == config.scales
        assert layer.frequency_matrices is None  # learnable=False
        assert layer.include_input is False
        assert layer.output_dim == 2 * config.n_features * len(config.scales)

    def test_default_scales_when_not_provided(self, input_dim: int) -> None:
        layer = MultiScaleFourierFeatures(input_dim=input_dim, n_features=4, scales=None)

        assert layer.scales == mf_module.DEFAULT_SCALES
        assert layer.n_scales == len(mf_module.DEFAULT_SCALES)

    def test_non_learnable_uses_buffers_in_forward(self, input_dim: int) -> None:
        layer = MultiScaleFourierFeatures(
            input_dim=input_dim, n_features=4, scales=[1.0, 2.0], learnable=False
        )
        for i in range(layer.n_scales):
            assert hasattr(layer, f"B_{i}")

        x = torch.randn(2, input_dim)
        out = layer(x)
        assert out.shape == (2, layer.output_dim)
        assert torch.isfinite(out).all()

    def test_forward_rejects_wrong_input_dim(self, input_dim: int) -> None:
        layer = MultiScaleFourierFeatures(input_dim=input_dim, n_features=4, scales=[1.0])
        with pytest.raises(ValueError, match=f"Expected input_dim={input_dim}"):
            layer(torch.randn(2, input_dim + 1))

    @pytest.mark.parametrize("bad_input_dim", [0, -1])
    def test_init_rejects_invalid_input_dim(self, bad_input_dim: int) -> None:
        with pytest.raises(ValueError, match="input_dim must be >= 1"):
            MultiScaleFourierFeatures(input_dim=bad_input_dim, n_features=4)

    def test_init_rejects_invalid_n_features(self) -> None:
        with pytest.raises(ValueError, match="n_features must be >= 1"):
            MultiScaleFourierFeatures(input_dim=2, n_features=0)

    def test_init_rejects_empty_scales(self) -> None:
        with pytest.raises(ValueError, match="scales must contain"):
            MultiScaleFourierFeatures(input_dim=2, n_features=4, scales=[])

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0])
    def test_init_rejects_non_positive_scale(self, bad_scale: float) -> None:
        with pytest.raises(ValueError, match="all scales must be > 0"):
            MultiScaleFourierFeatures(input_dim=2, n_features=4, scales=[1.0, bad_scale])


class TestAdaptiveFourierFeaturesGapCoverage:
    """Validation paths for AdaptiveFourierFeatures."""

    def test_init_rejects_inverted_scale_range(self) -> None:
        with pytest.raises(ValueError, match=r"scale_range\[0\] must be"):
            AdaptiveFourierFeatures(input_dim=2, n_features=8, scale_range=(10.0, 1.0))

    def test_init_rejects_non_positive_scale_range(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            AdaptiveFourierFeatures(input_dim=2, n_features=8, scale_range=(0.0, 1.0))

    def test_forward_rejects_wrong_input_dim(self) -> None:
        layer = AdaptiveFourierFeatures(input_dim=2, n_features=8, n_frequency_banks=2)
        with pytest.raises(ValueError, match="Expected input_dim=2"):
            layer(torch.randn(2, 3))


class TestProgressiveFourierFeaturesGapCoverage:
    """Non-learnable path + validation in ProgressiveFourierFeatures."""

    def test_non_learnable_uses_buffers_in_forward(self) -> None:
        layer = ProgressiveFourierFeatures(
            input_dim=2, n_features=4, scales=[1.0, 2.0], learnable=False
        )
        layer.set_progress(1.0)

        out = layer(torch.randn(2, 2))
        assert out.shape == (2, layer.output_dim)
        assert torch.isfinite(out).all()

    def test_forward_rejects_wrong_input_dim(self) -> None:
        layer = ProgressiveFourierFeatures(input_dim=2, n_features=4, scales=[1.0])
        with pytest.raises(ValueError, match="Expected input_dim=2"):
            layer(torch.randn(2, 5))


class TestPositionalEncodingsGapCoverage:
    """Defensive branches in PositionalEncoding & SpatialPositionalEncoding."""

    def test_positional_encoding_rejects_dim_mismatch(self) -> None:
        d_model = 8
        layer = PositionalEncoding(d_model=d_model, max_len=4)
        with pytest.raises(ValueError, match=f"Expected d_model={d_model}"):
            layer(torch.randn(1, 4, d_model + 1))

    def test_spatial_encoding_rejects_non_4d(self) -> None:
        layer = SpatialPositionalEncoding(d_model=8, max_size=4)
        with pytest.raises(ValueError, match="Expected 4D input"):
            layer(torch.randn(2, 4, 4))  # 3D

    def test_spatial_encoding_truncates_when_pe_too_wide(self) -> None:
        d_model = 8  # divisible by 4
        spatial_size = 4
        layer = SpatialPositionalEncoding(d_model=d_model, max_size=spatial_size)
        # Channel dim deliberately smaller than d_model -> exercise truncation branch
        smaller_channels = d_model // 2
        x = torch.zeros(1, smaller_channels, spatial_size, spatial_size)
        out = layer(x)
        assert out.shape == x.shape

    def test_spatial_encoding_pads_when_pe_too_narrow(self) -> None:
        d_model = 8
        spatial_size = 4
        layer = SpatialPositionalEncoding(d_model=d_model, max_size=spatial_size)
        # Channel dim deliberately larger than d_model -> exercise zero-padding branch
        larger_channels = d_model * 2
        x = torch.zeros(1, larger_channels, spatial_size, spatial_size)
        out = layer(x)
        assert out.shape == x.shape

    def test_spatial_encoding_rejects_indivisible_d_model(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            SpatialPositionalEncoding(d_model=10)


# ---------------------------------------------------------------------------
# FNet: validation + truncation branch in _mix_1d
# ---------------------------------------------------------------------------


class TestFNetGapCoverage:
    """Validation paths and rare branches in fnet.py."""

    def test_default_ffn_expansion_used_when_d_ffn_missing(self) -> None:
        d_model = 16
        block = FNetBlock(d_model=d_model)
        # First Linear layer in the FFN must reflect DEFAULT_FFN_EXPANSION
        assert block.ffn[0].out_features == fnet_module.DEFAULT_FFN_EXPANSION * d_model

    @pytest.mark.parametrize("bad_d_model", [0, -3])
    def test_init_rejects_invalid_d_model(self, bad_d_model: int) -> None:
        with pytest.raises(ValueError, match="d_model must be > 0"):
            FNetBlock(d_model=bad_d_model)

    def test_init_rejects_invalid_d_ffn(self) -> None:
        with pytest.raises(ValueError, match="d_ffn must be > 0"):
            FNetBlock(d_model=16, d_ffn=0)

    def test_init_rejects_invalid_dropout(self) -> None:
        with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
            FNetBlock(d_model=16, dropout=1.0)

    def test_mix_1d_truncates_when_freq_longer_than_input(self) -> None:
        """Cover the n_freq > n branch by injecting an oversized FFT result."""
        mixer = FNetMixing(use_2d=False)
        n, d = 4, 3

        class FakeFFTResult:
            def __init__(self, real: torch.Tensor) -> None:
                self.real = real

        oversized = torch.randn(2, n + 3, d)  # n_freq deliberately > n

        with patch("torch.fft.rfft", return_value=FakeFFTResult(oversized)):
            out = mixer(torch.randn(2, n, d))

        assert out.shape == (2, n, d)


# ---------------------------------------------------------------------------
# StabilityGuard: new params + SVD failure fallback
# ---------------------------------------------------------------------------


class TestStabilityGuardNewParams:
    """New ``margin_multiplier`` + validation + backwards-compat default."""

    def test_default_margin_multiplier_preserves_behavior(self) -> None:
        guard_default = StabilityGuard(beta_threshold=1e-3)
        guard_explicit = StabilityGuard(beta_threshold=1e-3, margin_multiplier=10.0)

        torch.manual_seed(0)
        keys = torch.randn(4, 8, 6) * 1e-3
        loss_default = guard_default.regularization_loss(keys.clone())
        loss_explicit = guard_explicit.regularization_loss(keys.clone())

        assert torch.allclose(loss_default, loss_explicit)

    def test_larger_margin_multiplier_gives_larger_loss(self) -> None:
        torch.manual_seed(1)
        keys = torch.randn(2, 10, 4) * 1e-4

        small = StabilityGuard(beta_threshold=1e-3, margin_multiplier=1.0)
        large = StabilityGuard(beta_threshold=1e-3, margin_multiplier=100.0)

        loss_small = small.regularization_loss(keys.clone())
        loss_large = large.regularization_loss(keys.clone())

        assert loss_large.item() >= loss_small.item()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"beta_threshold": 0.0}, "beta_threshold must be > 0"),
            ({"beta_threshold": -1.0}, "beta_threshold must be > 0"),
            ({"regularization_strength": -0.1}, "regularization_strength must be >= 0"),
            ({"log_interval": 0}, "log_interval must be >= 1"),
            ({"margin_multiplier": 0.0}, "margin_multiplier must be > 0"),
        ],
    )
    def test_init_rejects_invalid_args(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            StabilityGuard(**kwargs)


class TestStabilityGuardSVDFallback:
    """Cover the RuntimeError fallback in compute_lbb_constant."""

    def test_returns_zeros_when_svd_raises(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(3, 5, 4)

        with patch("torch.linalg.svdvals", side_effect=RuntimeError("synthetic")):
            beta = guard.compute_lbb_constant(keys)

        assert beta.shape == (keys.shape[0],)
        assert torch.equal(beta, torch.zeros(keys.shape[0]))


class TestStableGalerkinInitializerNewParams:
    """New tunable safety constants on StableGalerkinInitializer."""

    def test_defaults_match_class_constants(self) -> None:
        initializer = StableGalerkinInitializer()
        assert initializer.scale_epsilon == StableGalerkinInitializer.DEFAULT_SCALE_EPSILON
        assert initializer.scale_clamp == StableGalerkinInitializer.DEFAULT_SCALE_CLAMP
        assert (
            initializer.guard_threshold_ratio
            == StableGalerkinInitializer.DEFAULT_GUARD_THRESHOLD_RATIO
        )

    def test_custom_guard_threshold_ratio_propagates(self) -> None:
        beta_target = 0.2
        ratio = 4.0
        initializer = StableGalerkinInitializer(
            beta_target=beta_target, guard_threshold_ratio=ratio
        )
        assert initializer.stability_guard.beta_threshold == pytest.approx(beta_target / ratio)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"beta_target": 0.0}, "beta_target must be > 0"),
            ({"max_iterations": 0}, "max_iterations must be >= 1"),
            ({"scale_epsilon": 0.0}, "scale_epsilon must be > 0"),
            ({"guard_threshold_ratio": 0.0}, "guard_threshold_ratio must be > 0"),
            ({"scale_clamp": (0.0, 1.0)}, "scale_clamp values must be > 0"),
            ({"scale_clamp": (3.0, 1.0)}, "scale_clamp\\[0\\] must be <="),
            ({"scale_clamp": (1.0,)}, "scale_clamp must have exactly 2 elements"),
            ({"scale_clamp": (1.0, 2.0, 3.0)}, "scale_clamp must have exactly 2 elements"),
        ],
    )
    def test_init_rejects_invalid_args(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            StableGalerkinInitializer(**kwargs)


# ---------------------------------------------------------------------------
# Smoke: composing the operators that the fusion head will chain
# ---------------------------------------------------------------------------


class TestFusionHeadCompositionSmoke:
    """Composability smoke test mirroring the planned Galerkin Fusion Head.

    The Mouse-Droid-AGI fusion head will (a) embed heterogeneous coordinates
    via MultiScaleFourierFeatures, (b) project to a shared latent dim,
    (c) apply GalerkinAttention. This test exercises the chain end-to-end
    so any interface drift between the operators surfaces in CI before the
    fusion-head implementation begins.
    """

    def test_fourier_to_attention_chain(self) -> None:
        torch.manual_seed(7)
        input_dim, n_features, d_model, n_heads = 2, 8, 16, 2
        n_tokens = 5

        fourier = MultiScaleFourierFeatures(
            input_dim=input_dim, n_features=n_features, scales=[1.0, 2.0]
        )
        # Fourier output dim depends on n_scales, n_features, include_input
        proj = torch.nn.Linear(fourier.output_dim, d_model)
        attn = GalerkinAttention(d_model=d_model, n_heads=n_heads)

        coords = torch.rand(1, n_tokens, input_dim)
        embedded = fourier(coords)
        latent = proj(embedded)
        out, lbb = attn(latent, return_lbb=True)

        assert out.shape == (1, n_tokens, d_model)
        assert lbb.shape == (1,)
        assert torch.isfinite(out).all()
        assert torch.isfinite(lbb).all()

    def test_stability_guard_can_consume_attention_keys(self) -> None:
        d_key = 4
        guard = StabilityGuard(beta_threshold=1e-9)
        keys = torch.randn(2, 6, d_key)
        is_stable, beta = guard.check_stability(keys)
        assert isinstance(is_stable, bool)
        assert beta.shape == (keys.shape[0],)
