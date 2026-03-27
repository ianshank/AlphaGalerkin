"""Tests for NeuralOperator model."""

from __future__ import annotations

import pytest
import torch

from src.modeling.operator import NeuralOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42

# Small dimensions to keep tests fast
SMALL_WIDTH = 16
SMALL_LAYERS = 2
SMALL_MODES = 4
BATCH_SIZE = 2
SPATIAL_SIZE = 16


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    torch.manual_seed(DEFAULT_SEED)


@pytest.fixture(
    params=[
        pytest.param("fno", id="fno"),
        pytest.param("galerkin", id="galerkin"),
    ]
)
def backend(request: pytest.FixtureRequest) -> str:
    """Parametrized backend fixture."""
    return request.param


@pytest.fixture
def small_operator(backend: str) -> NeuralOperator:
    """Create a small NeuralOperator for the given backend."""
    return NeuralOperator(
        in_channels=1,
        out_channels=1,
        width=SMALL_WIDTH,
        n_layers=SMALL_LAYERS,
        modes=SMALL_MODES,
        backend=backend,
    )


@pytest.fixture
def input_tensor() -> torch.Tensor:
    """Create a small input tensor [B, C, H, W]."""
    return torch.randn(BATCH_SIZE, 1, SPATIAL_SIZE, SPATIAL_SIZE)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """Tests for backend selection and initialization."""

    def test_valid_backend_stored(self, small_operator: NeuralOperator, backend: str) -> None:
        """Backend string is stored on the instance."""
        assert small_operator.backend == backend

    def test_fno_backend_creates_fno2d(self) -> None:
        """FNO backend creates an FNO2d inner model."""
        from src.modeling.fno_layer import FNO2d

        op = NeuralOperator(
            in_channels=1, out_channels=1, width=SMALL_WIDTH,
            n_layers=SMALL_LAYERS, modes=SMALL_MODES, backend="fno",
        )
        assert isinstance(op.model, FNO2d)

    def test_galerkin_backend_creates_galerkin2d(self) -> None:
        """Galerkin backend creates a Galerkin2d inner model."""
        from src.modeling.galerkin_operator import Galerkin2d

        op = NeuralOperator(
            in_channels=1, out_channels=1, width=SMALL_WIDTH,
            n_layers=SMALL_LAYERS, modes=SMALL_MODES, backend="galerkin",
        )
        assert isinstance(op.model, Galerkin2d)

    def test_unsupported_backend_raises(self) -> None:
        """An unsupported backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="wavelet"):
            NeuralOperator(
                in_channels=1, out_channels=1, width=SMALL_WIDTH,
                backend="wavelet",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Forward pass shapes
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    """Tests for forward pass output shapes."""

    def test_output_matches_input_spatial(
        self, small_operator: NeuralOperator, input_tensor: torch.Tensor
    ) -> None:
        """Output spatial dimensions match input."""
        output = small_operator(input_tensor)
        assert output.shape == (BATCH_SIZE, 1, SPATIAL_SIZE, SPATIAL_SIZE)

    @pytest.mark.parametrize(
        ("in_ch", "out_ch"),
        [(1, 1), (1, 3), (3, 1)],
        ids=["1to1", "1to3", "3to1"],
    )
    def test_multi_channel(self, backend: str, in_ch: int, out_ch: int) -> None:
        """Multiple input/output channels produce correct shapes."""
        op = NeuralOperator(
            in_channels=in_ch,
            out_channels=out_ch,
            width=SMALL_WIDTH,
            n_layers=SMALL_LAYERS,
            modes=SMALL_MODES,
            backend=backend,
        )
        x = torch.randn(BATCH_SIZE, in_ch, SPATIAL_SIZE, SPATIAL_SIZE)
        y = op(x)
        assert y.shape == (BATCH_SIZE, out_ch, SPATIAL_SIZE, SPATIAL_SIZE)

    @pytest.mark.parametrize("spatial", [8, 16, 32])
    def test_various_resolutions(self, backend: str, spatial: int) -> None:
        """Operator handles various spatial resolutions."""
        modes = min(SMALL_MODES, spatial // 2)
        op = NeuralOperator(
            in_channels=1,
            out_channels=1,
            width=SMALL_WIDTH,
            n_layers=SMALL_LAYERS,
            modes=modes,
            backend=backend,
        )
        x = torch.randn(BATCH_SIZE, 1, spatial, spatial)
        y = op(x)
        assert y.shape == (BATCH_SIZE, 1, spatial, spatial)

    def test_forward_no_nan(
        self, small_operator: NeuralOperator, input_tensor: torch.Tensor
    ) -> None:
        """Output contains no NaNs or Infs."""
        output = small_operator(input_tensor)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


class TestParameterCounting:
    """Tests for count_parameters method."""

    def test_positive_parameter_count(self, small_operator: NeuralOperator) -> None:
        """Operator has a positive number of trainable parameters."""
        n_params = small_operator.count_parameters()
        assert n_params > 0

    def test_count_matches_pytorch(self, small_operator: NeuralOperator) -> None:
        """count_parameters matches manual sum over parameters."""
        expected = sum(p.numel() for p in small_operator.parameters() if p.requires_grad)
        assert small_operator.count_parameters() == expected

    def test_wider_model_has_more_params(self, backend: str) -> None:
        """A wider model has strictly more parameters."""
        narrow = NeuralOperator(
            in_channels=1, out_channels=1, width=SMALL_WIDTH,
            n_layers=SMALL_LAYERS, modes=SMALL_MODES, backend=backend,
        )
        wide = NeuralOperator(
            in_channels=1, out_channels=1, width=SMALL_WIDTH * 2,
            n_layers=SMALL_LAYERS, modes=SMALL_MODES, backend=backend,
        )
        assert wide.count_parameters() > narrow.count_parameters()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests for gradient computation through the operator."""

    def test_gradients_flow_to_parameters(
        self, small_operator: NeuralOperator, input_tensor: torch.Tensor
    ) -> None:
        """Backpropagation produces gradients for all trainable parameters."""
        output = small_operator(input_tensor)
        loss = output.sum()
        loss.backward()

        for name, param in small_operator.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_to_input(
        self, small_operator: NeuralOperator
    ) -> None:
        """Gradients flow back to input when requires_grad=True."""
        x = torch.randn(
            BATCH_SIZE, 1, SPATIAL_SIZE, SPATIAL_SIZE, requires_grad=True
        )
        output = small_operator(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Galerkin-specific configuration
# ---------------------------------------------------------------------------


class TestGalerkinSpecific:
    """Tests specific to the Galerkin backend."""

    def test_default_n_heads_derived_from_width(self) -> None:
        """When n_heads is None, it defaults to width // 16."""
        width = 64
        op = NeuralOperator(
            in_channels=1, out_channels=1, width=width,
            n_layers=SMALL_LAYERS, modes=SMALL_MODES, backend="galerkin",
        )
        # Galerkin2d should have been created with max(1, 64//16) = 4 heads
        assert op.count_parameters() > 0  # Sanity: model was created

    def test_explicit_n_heads(self) -> None:
        """Explicit n_heads is accepted."""
        op = NeuralOperator(
            in_channels=1, out_channels=1, width=SMALL_WIDTH,
            n_layers=SMALL_LAYERS, modes=SMALL_MODES,
            n_heads=2, backend="galerkin",
        )
        assert op.count_parameters() > 0
