import pytest
import torch

from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel


@pytest.mark.e2e
def test_user_journey_resolution_transfer() -> None:
    """End-to-end journey for zero-shot resolution transfer."""
    # Step 1: Instantiate AlphaGalerkinModel initialized for 9x9 inputs
    config = OperatorConfig(
        d_model=32,
        d_key=32,
        d_value=32,
        d_ffn=64,
        n_heads=2,
        n_galerkin_layers=2,
        n_softmax_layers=1,
        input_channels=17,
        game_type="go",
    )
    model = AlphaGalerkinModel(config)
    model.eval()

    # Step 2: Call model.adapt_resolution(source_size=9, target_size=19)
    model.adapt_resolution(source_size=9, target_size=19)

    # Step 3: Run forward inference with a 19x19 input tensor (1, 17, 19, 19)
    batch_size = 1
    target_size = 19
    x = torch.randn(batch_size, 17, target_size, target_size)

    with torch.no_grad():
        output = model(x)

    # Step 4: Verify policy output shape adapts to (1, 362) (361 positions + 1 pass)
    # and output value is finite
    assert output.policy_logits.shape == (batch_size, target_size * target_size + 1)
    assert torch.isfinite(output.value).all()
