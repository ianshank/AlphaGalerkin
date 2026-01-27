import pytest
import torch
import torch.nn as nn
from src.modeling.fno_layer import SpectralConv2d, FNOBlock, FNO2d

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestSpectralConv2d:
    @pytest.mark.parametrize("in_channels, out_channels", [(1, 1), (16, 32)])
    @pytest.mark.parametrize("modes", [(4, 4), (8, 8)])
    def test_initialization(self, in_channels, out_channels, modes):
        modes1, modes2 = modes
        layer = SpectralConv2d(in_channels, out_channels, modes1=modes1, modes2=modes2)
        assert layer.in_channels == in_channels
        assert layer.out_channels == out_channels
        assert layer.weights1.shape == (in_channels, out_channels, modes1, modes2)
        assert layer.weights2.shape == (in_channels, out_channels, modes1, modes2)

    def test_forward_shape(self, device):
        bs, c_in, h, w = 2, 8, 32, 32
        c_out = 16
        layer = SpectralConv2d(c_in, c_out, modes1=8, modes2=8).to(device)
        x = torch.randn(bs, c_in, h, w).to(device)
        
        out = layer(x)
        assert out.shape == (bs, c_out, h, w)

    def test_gradients(self, device):
        layer = SpectralConv2d(4, 4, modes1=4, modes2=4).to(device)
        x = torch.randn(2, 4, 16, 16).to(device)
        x.requires_grad_(True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert layer.weights1.grad is not None
        assert layer.weights2.grad is not None
        assert x.grad is not None

class TestFNOBlock:
    @pytest.mark.parametrize("activation_name", ["gelu", "relu"])
    def test_activation_support(self, activation_name, device):
        block = FNOBlock(width=16, activation=activation_name).to(device)
        if activation_name == "gelu":
            assert isinstance(block.activation, nn.GELU)
        else:
            assert isinstance(block.activation, nn.ReLU)
            
        x = torch.randn(2, 16, 32, 32).to(device)
        out = block(x)
        assert out.shape == (2, 16, 32, 32)

class TestFNO2d:
    def test_end_to_end_shape(self, device):
        bs, c_in, c_out = 2, 1, 1
        h, w = 32, 32
        model = FNO2d(in_channels=c_in, out_channels=c_out, width=16, n_layers=2).to(device)
        x = torch.randn(bs, c_in, h, w).to(device)
        
        out = model(x)
        assert out.shape == (bs, c_out, h, w)

    def test_variable_resolution(self, device):
        # FNO should handle different resolutions at inference time
        model = FNO2d(in_channels=1, width=16, modes1=4, modes2=4).to(device)
        
        # Train resolution
        x1 = torch.randn(1, 1, 32, 32).to(device)
        out1 = model(x1)
        assert out1.shape == (1, 1, 32, 32)
        
        # Test resolution (e.g. higher res)
        x2 = torch.randn(1, 1, 64, 64).to(device)
        out2 = model(x2)
        assert out2.shape == (1, 1, 64, 64)

    def test_explicit_coords(self, device):
        bs, h, w = 2, 16, 16
        model = FNO2d(in_channels=1, width=16).to(device)
        x = torch.randn(bs, 1, h, w).to(device)
        
        # Create dummy coords
        coords = torch.randn(bs, h, w, 2).to(device)
        
        out = model(x, coords=coords)
        assert out.shape == (bs, 1, h, w)
