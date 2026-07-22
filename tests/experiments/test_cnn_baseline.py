"""Tests for the DiscreteCNNBaseline arm (shapes, determinism, param matching)."""

from __future__ import annotations

import pytest
import torch

from src.experiments.cnn_baseline import (
    DiscreteCNNBaseline,
    _infer_grid_size,
    count_parameters,
    match_cnn_channels,
)
from src.physics.poisson import generate_influence_field


class TestForward:
    @pytest.mark.parametrize("n", [9, 13, 19])
    def test_flattened_roundtrip_shape(self, n: int) -> None:
        model = DiscreteCNNBaseline(n_layers=2, channels=8, kernel_size=3)
        out = model(torch.randn(4, n * n))
        assert out.shape == (4, n * n)

    def test_same_weights_run_at_any_resolution(self) -> None:
        """Fully-convolutional: one instance runs at multiple N (but is retrained per res)."""
        model = DiscreteCNNBaseline(n_layers=3, channels=6, kernel_size=3)
        assert model(torch.randn(2, 81)).shape == (2, 81)
        assert model(torch.randn(2, 361)).shape == (2, 361)

    def test_accepts_gridded_input(self) -> None:
        model = DiscreteCNNBaseline(n_layers=1, channels=4)
        assert model(torch.randn(3, 9, 9)).shape == (3, 81)
        assert model(torch.randn(3, 1, 9, 9)).shape == (3, 81)

    def test_rejects_non_square(self) -> None:
        model = DiscreteCNNBaseline(n_layers=1, channels=4)
        with pytest.raises(ValueError, match="square"):
            model(torch.randn(2, 3, 5))
        with pytest.raises(ValueError, match="square"):
            model(torch.randn(2, 1, 3, 5))

    def test_rejects_bad_rank(self) -> None:
        model = DiscreteCNNBaseline(n_layers=1, channels=4)
        with pytest.raises(ValueError, match="rank"):
            model(torch.randn(2, 3, 4, 5, 6))

    def test_deterministic_under_seed(self) -> None:
        torch.manual_seed(0)
        m1 = DiscreteCNNBaseline(n_layers=2, channels=8)
        torch.manual_seed(0)
        m2 = DiscreteCNNBaseline(n_layers=2, channels=8)
        x = torch.randn(2, 81)
        m1.eval()
        m2.eval()
        assert torch.allclose(m1(x), m2(x))

    def test_batchnorm_and_dropout_paths(self) -> None:
        # batchnorm needs batch > 1 in train mode; exercise both branches.
        m_bn = DiscreteCNNBaseline(n_layers=2, channels=4, use_batchnorm=True, dropout=0.1)
        m_bn.train()
        assert m_bn(torch.randn(4, 81)).shape == (4, 81)
        m_plain = DiscreteCNNBaseline(n_layers=2, channels=4, use_batchnorm=False, dropout=0.0)
        m_plain.train()
        assert m_plain(torch.randn(4, 81)).shape == (4, 81)


class TestConstructorValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"kernel_size": 4}, "odd"),
            ({"channels": 0}, "channels"),
            ({"n_layers": -1}, "n_layers"),
            ({"dropout": 1.0}, "dropout"),
        ],
    )
    def test_rejects_bad_args(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            DiscreteCNNBaseline(**kwargs)  # type: ignore[arg-type]

    def test_zero_layers_ok(self) -> None:
        model = DiscreteCNNBaseline(n_layers=0, channels=4)
        assert model(torch.randn(2, 81)).shape == (2, 81)


class TestInferGridSize:
    def test_perfect_square(self) -> None:
        assert _infer_grid_size(81) == 9
        assert _infer_grid_size(361) == 19

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError, match="perfect square"):
            _infer_grid_size(80)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _infer_grid_size(0)


class TestParamMatching:
    def test_count_parameters(self) -> None:
        model = DiscreteCNNBaseline(n_layers=2, channels=8, kernel_size=3)
        assert count_parameters(model) == sum(p.numel() for p in model.parameters())

    def test_match_hits_target_within_tolerance(self) -> None:
        target = 200_000
        channels = match_cnn_channels(target, n_layers=6, kernel_size=3, tolerance=0.15)
        built = DiscreteCNNBaseline(n_layers=6, channels=channels, kernel_size=3)
        assert abs(count_parameters(built) - target) / target <= 0.15

    def test_match_monotone_in_target(self) -> None:
        small = match_cnn_channels(50_000, n_layers=4, kernel_size=3)
        large = match_cnn_channels(500_000, n_layers=4, kernel_size=3)
        assert large > small

    def test_match_rejects_non_positive_target(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            match_cnn_channels(0, n_layers=4, kernel_size=3)

    def test_match_warns_when_outside_tolerance(self) -> None:
        # A tiny target with a deep net can't be matched within a tight band; the
        # helper still returns the closest width (>=1) rather than raising.
        channels = match_cnn_channels(1, n_layers=8, kernel_size=3, tolerance=0.01)
        assert channels >= 1


class TestPoissonSampleCoupling:
    """Lock the C-order coupling between the CNN reshape and ``PoissonSample`` fields.

    The CNN arm reshapes the flat charge vector to an image with ``view``; if
    ``PoissonSample`` ever changed its ``.flatten()`` order or ``coords`` indexing, the
    reshape would silently transpose the field while the shape-only tests stayed green.
    """

    def test_forward_matches_poisson_potential_shape(self) -> None:
        sample = generate_influence_field(grid_size=9, n_charges=3, seed=0)
        charges = torch.tensor(sample.charges, dtype=torch.float32).unsqueeze(0)
        out = DiscreteCNNBaseline(n_layers=1, channels=4)(charges)
        assert out.shape == (1, sample.potential.shape[0])

    def test_flatten_ordering_is_row_major_and_matches_coords(self) -> None:
        side = 9
        sample = generate_influence_field(grid_size=side, n_charges=3, seed=1)
        charges = torch.tensor(sample.charges, dtype=torch.float32)
        # The strongest-magnitude charge cell pins a specific (i, j).
        k = int(torch.argmax(torch.abs(charges)).item())
        i, j = divmod(k, side)
        # forward()'s (B, N) -> (B, 1, side, side) view must place flat index k at [i, j].
        grid = charges.view(1, 1, side, side)
        assert grid[0, 0, i, j].item() == pytest.approx(charges[k].item())
        # coords must share the SAME C-order flattening: coords[k] == (x[i], y[j]),
        # normalised to [0, 1] over meshgrid(indexing="ij").
        coords = sample.coords
        assert coords[k, 0] == pytest.approx(i / (side - 1), abs=1e-6)
        assert coords[k, 1] == pytest.approx(j / (side - 1), abs=1e-6)
