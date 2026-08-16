import pathlib

import pytest
import torch
import torch.nn as nn

from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel


@pytest.mark.e2e
def test_user_journey_go_training(tmp_path: pathlib.Path) -> None:
    """End-to-end journey for Go AI workflow."""
    # Step 1: Create OperatorConfig for Go
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

    # Step 2: Initialize AlphaGalerkinModel
    model = AlphaGalerkinModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Step 3: Lightweight mock training loop
    model.train()
    batch_size = 2
    board_size = 9
    for _ in range(2):
        # Random board state for Go (batch, channels, height, width)
        x = torch.randn(batch_size, 17, board_size, board_size)
        target_value = torch.rand(batch_size, 1) * 2 - 1.0  # [-1, 1]

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output.value, target_value)
        loss.backward()
        optimizer.step()

    # Step 4: Save model checkpoint
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Step 5: Load model and run inference
    loaded_model = AlphaGalerkinModel(config)
    # mypy requires types for loaded object, but torch.load returns Any
    loaded_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))  # type: ignore
    loaded_model.eval()

    with torch.no_grad():
        x_test = torch.randn(1, 17, board_size, board_size)
        output_test = loaded_model(x_test)

    # Step 6: Verify outputs
    assert output_test.policy_logits.shape == (1, board_size * board_size + 1)

    # Value is bounded around [-1, 1] given the tanh activation or initialized values
    # If linear without tanh, check shape and finiteness
    assert output_test.value.shape == (1, 1)
    assert torch.isfinite(output_test.value).all()
