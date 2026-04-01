from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.modeling.fnet import FNetBlock, FNetMixing, FNetMixingLayer


def _grid_seq_len(grid_size: int) -> int:
    return grid_size * grid_size


def _make_fnet_block(d_model: int = 16) -> FNetBlock:
    torch.manual_seed(0)
    block = FNetBlock(d_model=d_model, dropout=0.0, use_2d_fft=False)
    block.eval()
    return block


def _make_fnet_mixing_layer(d_model: int = 16) -> FNetMixingLayer:
    torch.manual_seed(0)
    layer = FNetMixingLayer(d_model=d_model)
    layer.eval()
    return layer


class TestFFTRoundtrip:
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        grid_size=st.integers(min_value=3, max_value=8),
        d_model=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=20, deadline=None)
    def test_rfft2_irfft2_roundtrip(self, batch_size: int, grid_size: int, d_model: int) -> None:
        torch.manual_seed(batch_size * 11 + grid_size * 7 + d_model)
        x = torch.randn(batch_size, d_model, grid_size, grid_size)
        x_freq = torch.fft.rfft2(x)
        x_reconstructed = torch.fft.irfft2(x_freq, s=(grid_size, grid_size))
        assert torch.allclose(x, x_reconstructed, atol=1e-5)

    def test_rfft2_irfft2_roundtrip_ortho(self) -> None:
        torch.manual_seed(42)
        grid_size = 5
        norm_mode = "ortho"
        x = torch.randn(2, grid_size, grid_size, 8)
        x_freq = torch.fft.rfft2(x, dim=(1, 2), norm=norm_mode)
        x_reconstructed = torch.fft.irfft2(
            x_freq, s=(grid_size, grid_size), dim=(1, 2), norm=norm_mode
        )
        assert torch.allclose(x, x_reconstructed, atol=1e-5)


class TestFNetMixingLayerProperties:
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        grid_size=st.integers(min_value=3, max_value=8),
    )
    @settings(max_examples=20, deadline=None)
    def test_output_shape_matches_input(self, batch_size: int, grid_size: int) -> None:
        d_model = 8
        layer = _make_fnet_mixing_layer(d_model=d_model)
        seq_len = _grid_seq_len(grid_size)
        torch.manual_seed(batch_size * 13 + grid_size)
        x = torch.randn(batch_size, seq_len, d_model)
        with torch.no_grad():
            out = layer(x, grid_size=grid_size)
        assert out.shape == x.shape

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        grid_size=st.integers(min_value=3, max_value=6),
    )
    @settings(max_examples=15, deadline=None)
    def test_batch_independence(self, batch_size: int, grid_size: int) -> None:
        d_model = 8
        layer = _make_fnet_mixing_layer(d_model=d_model)
        seq_len = _grid_seq_len(grid_size)
        torch.manual_seed(batch_size * 17 + grid_size)
        x = torch.randn(batch_size, seq_len, d_model)
        with torch.no_grad():
            out_batch = layer(x, grid_size=grid_size)
            for i in range(batch_size):
                out_single = layer(x[i : i + 1], grid_size=grid_size)
                assert torch.allclose(out_batch[i : i + 1], out_single, atol=1e-5)

    def test_determinism(self) -> None:
        layer = _make_fnet_mixing_layer(d_model=8)
        torch.manual_seed(42)
        x = torch.randn(2, 9, 8)
        with torch.no_grad():
            out1 = layer(x, grid_size=3)
            out2 = layer(x, grid_size=3)
        assert torch.allclose(out1, out2, atol=0.0)

    def test_real_valued_output(self) -> None:
        layer = _make_fnet_mixing_layer(d_model=8)
        x = torch.randn(2, 9, 8)
        with torch.no_grad():
            out = layer(x, grid_size=3)
        assert not out.is_complex()
        assert out.dtype == x.dtype

    def test_output_is_finite(self) -> None:
        layer = _make_fnet_mixing_layer(d_model=8)
        x = torch.randn(2, 25, 8)
        with torch.no_grad():
            out = layer(x, grid_size=5)
        assert torch.isfinite(out).all()


class TestFNetBlockProperties:
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=20, deadline=None)
    def test_output_shape_matches_input(self, batch_size: int, seq_len: int) -> None:
        d_model = 16
        block = _make_fnet_block(d_model=d_model)
        torch.manual_seed(batch_size * 19 + seq_len)
        x = torch.randn(batch_size, seq_len, d_model)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=15, deadline=None)
    def test_batch_independence(self, batch_size: int, seq_len: int) -> None:
        d_model = 16
        block = _make_fnet_block(d_model=d_model)
        torch.manual_seed(batch_size * 23 + seq_len)
        x = torch.randn(batch_size, seq_len, d_model)
        with torch.no_grad():
            out_batch = block(x)
            for i in range(batch_size):
                out_single = block(x[i : i + 1])
                assert torch.allclose(out_batch[i : i + 1], out_single, atol=1e-5)

    def test_determinism_in_eval_mode(self) -> None:
        block = _make_fnet_block(d_model=16)
        x = torch.randn(2, 8, 16)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)
        assert torch.allclose(out1, out2, atol=0.0)

    def test_real_valued_output(self) -> None:
        block = _make_fnet_block(d_model=16)
        x = torch.randn(2, 8, 16)
        with torch.no_grad():
            out = block(x)
        assert not out.is_complex()
        assert out.dtype == x.dtype

    def test_output_is_finite(self) -> None:
        block = _make_fnet_block(d_model=16)
        x = torch.randn(2, 16, 16)
        with torch.no_grad():
            out = block(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("seq_len", [9, 25, 81, 169])
    def test_2d_mode_output_shape(self, seq_len: int) -> None:
        grid_size = int(math.isqrt(seq_len))
        assert grid_size * grid_size == seq_len
        d_model = 16
        torch.manual_seed(0)
        block = FNetBlock(d_model=d_model, dropout=0.0, use_2d_fft=True)
        block.eval()
        x = torch.randn(2, seq_len, d_model)
        with torch.no_grad():
            out = block(x, board_size=grid_size)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestFNetMixingProperties:
    def test_1d_output_shape_matches_input(self) -> None:
        mixing = FNetMixing(use_2d=False)
        mixing.eval()
        x = torch.randn(2, 12, 8)
        with torch.no_grad():
            out = mixing(x)
        assert out.shape == x.shape

    def test_2d_output_shape_matches_input(self) -> None:
        mixing = FNetMixing(use_2d=True)
        mixing.eval()
        grid_size = 4
        x = torch.randn(2, grid_size * grid_size, 8)
        with torch.no_grad():
            out = mixing(x, board_size=grid_size)
        assert out.shape == x.shape

    def test_1d_output_is_real(self) -> None:
        mixing = FNetMixing(use_2d=False)
        x = torch.randn(2, 10, 8)
        with torch.no_grad():
            out = mixing(x)
        assert not out.is_complex()

    def test_2d_output_is_real(self) -> None:
        mixing = FNetMixing(use_2d=True)
        x = torch.randn(2, 16, 8)
        with torch.no_grad():
            out = mixing(x, board_size=4)
        assert not out.is_complex()
