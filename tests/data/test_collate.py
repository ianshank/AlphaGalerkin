"""Tests for collation functions.

Tests cover:
- TrainingBatch: Batched training data with padding
- VariableSizeCollator: Pads variable board sizes
- SameSizeCollator: Efficient same-size batching
- create_collator: Factory function
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from src.data.collate import (
    SameSizeCollator,
    TrainingBatch,
    VariableSizeCollator,
    create_collator,
)
from src.training.replay_buffer import Experience

# --- Fixtures ---


def create_experience(
    board_size: int = 9,
    n_channels: int = 17,
    seed: int | None = None,
) -> Experience:
    """Create a test experience with given parameters."""
    if seed is not None:
        torch.manual_seed(seed)

    n_actions = board_size**2 + 1
    return Experience(
        board_state=torch.randn(n_channels, board_size, board_size),
        board_size=board_size,
        target_policy=torch.softmax(torch.randn(n_actions), dim=0),
        target_value=random.uniform(-1.0, 1.0) if seed is None else 0.5,
        metadata={"seed": seed},
    )


@pytest.fixture
def variable_size_experiences() -> list[Experience]:
    """Create experiences with varying board sizes."""
    experiences = []
    for i, size in enumerate([9, 13, 19, 9, 13]):
        experiences.append(create_experience(board_size=size, seed=i))
    return experiences


@pytest.fixture
def same_size_experiences() -> list[Experience]:
    """Create experiences with same board size."""
    return [create_experience(board_size=9, seed=i) for i in range(5)]


@pytest.fixture
def single_experience() -> Experience:
    """Create a single experience."""
    return create_experience(board_size=9, seed=42)


# --- TrainingBatch Tests ---


class TestTrainingBatch:
    """Tests for TrainingBatch dataclass."""

    @pytest.fixture
    def sample_batch(self) -> TrainingBatch:
        """Create a sample batch for testing."""
        batch_size = 4
        max_size = 19
        n_channels = 17
        max_actions = max_size**2 + 1

        return TrainingBatch(
            board_states=torch.randn(batch_size, n_channels, max_size, max_size),
            board_sizes=torch.tensor([9, 13, 19, 9], dtype=torch.int64),
            target_policies=torch.rand(batch_size, max_actions),
            target_values=torch.rand(batch_size, 1),
            position_mask=torch.ones(batch_size, max_size, max_size, dtype=torch.bool),
            action_mask=torch.ones(batch_size, max_actions, dtype=torch.bool),
        )

    def test_batch_size_property(self, sample_batch: TrainingBatch):
        """Test batch_size property returns correct value."""
        assert sample_batch.batch_size == 4

    def test_to_device_cpu(self, sample_batch: TrainingBatch):
        """Test moving batch to CPU."""
        device = torch.device("cpu")
        moved = sample_batch.to(device)

        assert moved.board_states.device == device
        assert moved.board_sizes.device == device
        assert moved.target_policies.device == device
        assert moved.target_values.device == device
        assert moved.position_mask.device == device
        assert moved.action_mask.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self, sample_batch: TrainingBatch):
        """Test moving batch to CUDA."""
        device = torch.device("cuda")
        moved = sample_batch.to(device)

        assert moved.board_states.device.type == "cuda"
        assert moved.board_sizes.device.type == "cuda"

    def test_to_returns_new_batch(self, sample_batch: TrainingBatch):
        """Test to() returns a new TrainingBatch instance."""
        moved = sample_batch.to(torch.device("cpu"))
        assert moved is not sample_batch

    @pytest.mark.gpu_required
    def test_pin_memory(self, sample_batch: TrainingBatch):
        """Test pin_memory returns new batch with pinned tensors."""
        pinned = sample_batch.pin_memory()
        assert isinstance(pinned, TrainingBatch)
        # Verify shape is preserved
        assert pinned.board_states.shape == sample_batch.board_states.shape

    @pytest.mark.gpu_required
    def test_pin_memory_returns_new_batch(self, sample_batch: TrainingBatch):
        """Test pin_memory returns a new instance."""
        pinned = sample_batch.pin_memory()
        assert pinned is not sample_batch

    def test_all_attributes_present(self, sample_batch: TrainingBatch):
        """Test all expected attributes are present."""
        assert hasattr(sample_batch, "board_states")
        assert hasattr(sample_batch, "board_sizes")
        assert hasattr(sample_batch, "target_policies")
        assert hasattr(sample_batch, "target_values")
        assert hasattr(sample_batch, "position_mask")
        assert hasattr(sample_batch, "action_mask")


# --- VariableSizeCollator Tests ---


class TestVariableSizeCollator:
    """Tests for VariableSizeCollator class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        collator = VariableSizeCollator()
        assert collator.pad_value == 0.0
        assert collator.max_board_size is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        collator = VariableSizeCollator(pad_value=-1.0, max_board_size=19)
        assert collator.pad_value == -1.0
        assert collator.max_board_size == 19

    def test_call_returns_training_batch(self, variable_size_experiences: list[Experience]):
        """Test __call__ returns TrainingBatch."""
        collator = VariableSizeCollator()
        batch = collator(variable_size_experiences)
        assert isinstance(batch, TrainingBatch)

    def test_empty_list_raises(self):
        """Test ValueError on empty experience list."""
        collator = VariableSizeCollator()
        with pytest.raises(ValueError, match="Cannot collate empty list"):
            collator([])

    def test_batch_size_correct(self, variable_size_experiences: list[Experience]):
        """Test batch dimension matches input size."""
        collator = VariableSizeCollator()
        batch = collator(variable_size_experiences)
        assert batch.batch_size == len(variable_size_experiences)

    def test_board_states_padded_to_max(self, variable_size_experiences: list[Experience]):
        """Test board states are padded to maximum board size."""
        collator = VariableSizeCollator()
        batch = collator(variable_size_experiences)

        max_size = max(exp.board_size for exp in variable_size_experiences)
        assert batch.board_states.shape[2] == max_size
        assert batch.board_states.shape[3] == max_size

    def test_max_board_size_enforced(self, same_size_experiences: list[Experience]):
        """Test max_board_size forces padding to larger size."""
        collator = VariableSizeCollator(max_board_size=19)
        batch = collator(same_size_experiences)

        assert batch.board_states.shape[2] == 19
        assert batch.board_states.shape[3] == 19

    def test_padding_uses_pad_value(self):
        """Test padding regions use specified pad value."""
        # Create experience with small board
        experience = create_experience(board_size=3, n_channels=1, seed=42)
        # Set all values to 1 for easy detection
        experience = Experience(
            board_state=torch.ones(1, 3, 3),
            board_size=3,
            target_policy=torch.ones(10) / 10,  # 3^2 + 1 = 10
            target_value=0.5,
            metadata={},
        )

        collator = VariableSizeCollator(pad_value=-999.0, max_board_size=5)
        batch = collator([experience])

        # Check padding region has pad value
        padded_region = batch.board_states[0, 0, 3:, :]
        assert torch.all(padded_region == -999.0)

    def test_board_sizes_preserved(self, variable_size_experiences: list[Experience]):
        """Test original board sizes are recorded correctly."""
        collator = VariableSizeCollator()
        batch = collator(variable_size_experiences)

        expected_sizes = [exp.board_size for exp in variable_size_experiences]
        assert batch.board_sizes.tolist() == expected_sizes

    def test_position_mask_correct(self, variable_size_experiences: list[Experience]):
        """Test position mask marks valid positions."""
        collator = VariableSizeCollator()
        batch = collator(variable_size_experiences)

        for i, exp in enumerate(variable_size_experiences):
            size = exp.board_size
            # Valid region should be True
            assert batch.position_mask[i, :size, :size].all()
            # If there's padding, it should be False
            if size < batch.position_mask.shape[1]:
                assert not batch.position_mask[i, size:, :].any()

    def test_action_mask_correct(self, variable_size_experiences: list[Experience]):
        """Test action mask marks valid actions."""
        collator = VariableSizeCollator()
        batch = collator(variable_size_experiences)

        max_size = batch.board_states.shape[2]
        max_actions = max_size**2 + 1

        for i, exp in enumerate(variable_size_experiences):
            size = exp.board_size
            # Pass action (last one) should always be valid
            assert batch.action_mask[i, max_size**2]

            # Valid position actions
            for row in range(size):
                for col in range(size):
                    new_idx = row * max_size + col
                    assert batch.action_mask[i, new_idx]

    def test_policy_remapped_correctly(self):
        """Test policy is correctly remapped to padded action space."""
        # Create 3x3 experience with known policy
        board_size = 3
        policy = torch.zeros(10)  # 3^2 + 1 = 10
        policy[0] = 0.5  # (0,0)
        policy[4] = 0.3  # (1,1)
        policy[9] = 0.2  # pass

        experience = Experience(
            board_state=torch.randn(1, board_size, board_size),
            board_size=board_size,
            target_policy=policy,
            target_value=0.5,
            metadata={},
        )

        collator = VariableSizeCollator(max_board_size=5)
        batch = collator([experience])

        # In 5x5 space: (0,0)->0, (1,1)->6, pass->25
        max_size = 5
        assert batch.target_policies[0, 0] == 0.5  # (0,0)
        assert batch.target_policies[0, 1 * max_size + 1] == 0.3  # (1,1)
        assert batch.target_policies[0, max_size**2] == 0.2  # pass

    def test_values_preserved(self, variable_size_experiences: list[Experience]):
        """Test target values are preserved correctly."""
        collator = VariableSizeCollator()
        batch = collator(variable_size_experiences)

        expected_values = [exp.target_value for exp in variable_size_experiences]
        actual_values = batch.target_values.squeeze(-1).tolist()

        for expected, actual in zip(expected_values, actual_values):
            assert abs(expected - actual) < 1e-5

    def test_single_experience(self, single_experience: Experience):
        """Test collation of single experience."""
        collator = VariableSizeCollator()
        batch = collator([single_experience])

        assert batch.batch_size == 1
        assert batch.board_sizes[0] == single_experience.board_size


# --- SameSizeCollator Tests ---


class TestSameSizeCollator:
    """Tests for SameSizeCollator class."""

    def test_call_returns_training_batch(self, same_size_experiences: list[Experience]):
        """Test __call__ returns TrainingBatch."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)
        assert isinstance(batch, TrainingBatch)

    def test_empty_list_raises(self):
        """Test ValueError on empty experience list."""
        collator = SameSizeCollator()
        with pytest.raises(ValueError, match="Cannot collate empty list"):
            collator([])

    def test_different_sizes_raises(self, variable_size_experiences: list[Experience]):
        """Test ValueError when experiences have different board sizes."""
        collator = SameSizeCollator()
        with pytest.raises(ValueError, match="Expected all experiences to have board size"):
            collator(variable_size_experiences)

    def test_batch_size_correct(self, same_size_experiences: list[Experience]):
        """Test batch dimension matches input size."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)
        assert batch.batch_size == len(same_size_experiences)

    def test_no_padding_needed(self, same_size_experiences: list[Experience]):
        """Test no extra padding is added."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)

        expected_size = same_size_experiences[0].board_size
        assert batch.board_states.shape[2] == expected_size
        assert batch.board_states.shape[3] == expected_size

    def test_board_states_stacked(self, same_size_experiences: list[Experience]):
        """Test board states are stacked correctly."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)

        for i, exp in enumerate(same_size_experiences):
            assert torch.allclose(batch.board_states[i], exp.board_state)

    def test_policies_stacked(self, same_size_experiences: list[Experience]):
        """Test policies are stacked correctly."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)

        for i, exp in enumerate(same_size_experiences):
            assert torch.allclose(batch.target_policies[i], exp.target_policy)

    def test_all_positions_valid(self, same_size_experiences: list[Experience]):
        """Test all positions are marked valid."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)

        assert batch.position_mask.all()

    def test_all_actions_valid(self, same_size_experiences: list[Experience]):
        """Test all actions are marked valid."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)

        assert batch.action_mask.all()

    def test_board_sizes_filled(self, same_size_experiences: list[Experience]):
        """Test board_sizes tensor is correctly filled."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)

        expected_size = same_size_experiences[0].board_size
        assert torch.all(batch.board_sizes == expected_size)

    def test_single_experience(self, single_experience: Experience):
        """Test collation of single experience."""
        collator = SameSizeCollator()
        batch = collator([single_experience])

        assert batch.batch_size == 1
        assert torch.allclose(batch.board_states[0], single_experience.board_state)


# --- create_collator Factory Tests ---


class TestCreateCollator:
    """Tests for create_collator factory function."""

    def test_variable_size_true_returns_variable_collator(self):
        """Test variable_size=True returns VariableSizeCollator."""
        collator = create_collator(variable_size=True)
        assert isinstance(collator, VariableSizeCollator)

    def test_variable_size_false_returns_same_collator(self):
        """Test variable_size=False returns SameSizeCollator."""
        collator = create_collator(variable_size=False)
        assert isinstance(collator, SameSizeCollator)

    def test_pad_value_passed_to_variable_collator(self):
        """Test pad_value is passed to VariableSizeCollator."""
        collator = create_collator(variable_size=True, pad_value=-1.0)
        assert isinstance(collator, VariableSizeCollator)
        assert collator.pad_value == -1.0

    def test_max_board_size_passed_to_variable_collator(self):
        """Test max_board_size is passed to VariableSizeCollator."""
        collator = create_collator(variable_size=True, max_board_size=19)
        assert isinstance(collator, VariableSizeCollator)
        assert collator.max_board_size == 19

    def test_default_parameters(self):
        """Test default parameters create VariableSizeCollator."""
        collator = create_collator()
        assert isinstance(collator, VariableSizeCollator)
        assert collator.pad_value == 0.0
        assert collator.max_board_size is None


# --- Integration Tests ---


class TestCollateIntegration:
    """Integration tests for collation with DataLoader."""

    def test_variable_collator_with_dataloader(self, variable_size_experiences: list[Experience]):
        """Test VariableSizeCollator works with DataLoader."""
        from torch.utils.data import DataLoader

        from src.data.dataset import ExperienceListDataset

        dataset = ExperienceListDataset(variable_size_experiences)
        collator = VariableSizeCollator()
        loader = DataLoader(dataset, batch_size=3, collate_fn=collator)

        for batch in loader:
            assert isinstance(batch, TrainingBatch)
            assert batch.batch_size <= 3

    def test_same_size_collator_with_dataloader(self, same_size_experiences: list[Experience]):
        """Test SameSizeCollator works with DataLoader."""
        from torch.utils.data import DataLoader

        from src.data.dataset import ExperienceListDataset

        dataset = ExperienceListDataset(same_size_experiences)
        collator = SameSizeCollator()
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        for batch in loader:
            assert isinstance(batch, TrainingBatch)

    def test_batch_to_device_workflow(self, same_size_experiences: list[Experience]):
        """Test typical workflow: collate -> to device."""
        collator = SameSizeCollator()
        batch = collator(same_size_experiences)

        # Simulate moving to device
        device = torch.device("cpu")
        batch = batch.to(device)

        assert batch.board_states.device == device

    def test_mixed_workflow_variable_sizes(self, variable_size_experiences: list[Experience]):
        """Test complete workflow with variable sizes."""
        from torch.utils.data import DataLoader

        # Create dataset with augmentation
        from src.data.dataset import AugmentedExperience, ExperienceListDataset

        augment = AugmentedExperience()
        dataset = ExperienceListDataset(variable_size_experiences, transform=augment)

        # Create collator and dataloader
        collator = create_collator(variable_size=True, max_board_size=19)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        # Iterate and verify
        for batch in loader:
            assert isinstance(batch, TrainingBatch)
            assert batch.board_states.shape[2] == 19
            assert batch.board_states.shape[3] == 19
            # Move to device
            batch = batch.to(torch.device("cpu"))
            assert batch.board_states.device.type == "cpu"
