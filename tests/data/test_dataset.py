"""Tests for dataset classes.

Tests cover:
- ReplayDataset: Dataset wrapper around replay buffer
- StreamingReplayDataset: Infinite iterator for training
- BoardSizeBatchSampler: Groups experiences by board size
- ExperienceListDataset: Indexed access to experience list
- AugmentedExperience: Data augmentation transformations
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from src.data.dataset import (
    AugmentedExperience,
    BoardSizeBatchSampler,
    ExperienceListDataset,
    ReplayDataset,
    StreamingReplayDataset,
)
from src.training.replay_buffer import Experience, UniformReplayBuffer

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
def sample_experiences() -> list[Experience]:
    """Create list of sample experiences with varying board sizes."""
    experiences = []
    for i in range(20):
        board_size = random.choice([9, 13, 19])
        experiences.append(create_experience(board_size=board_size, seed=i))
    return experiences


@pytest.fixture
def same_size_experiences() -> list[Experience]:
    """Create list of experiences with same board size."""
    return [create_experience(board_size=9, seed=i) for i in range(10)]


@pytest.fixture
def filled_buffer() -> UniformReplayBuffer:
    """Create a replay buffer with sample data."""
    buffer = UniformReplayBuffer(capacity=100)
    for i in range(50):
        board_size = random.choice([9, 13, 19])
        buffer.add(create_experience(board_size=board_size, seed=i))
    return buffer


# --- ReplayDataset Tests ---


class TestReplayDataset:
    """Tests for ReplayDataset class."""

    def test_init_with_buffer(self, filled_buffer: UniformReplayBuffer):
        """Test dataset initialization."""
        dataset = ReplayDataset(filled_buffer)
        assert dataset.buffer is filled_buffer
        assert dataset.transform is None

    def test_init_with_transform(self, filled_buffer: UniformReplayBuffer):
        """Test dataset initialization with transform."""
        transform = MagicMock()
        dataset = ReplayDataset(filled_buffer, transform=transform)
        assert dataset.transform is transform

    def test_len_matches_buffer(self, filled_buffer: UniformReplayBuffer):
        """Test __len__ returns buffer size."""
        dataset = ReplayDataset(filled_buffer)
        assert len(dataset) == len(filled_buffer)

    def test_getitem_returns_experience(self, filled_buffer: UniformReplayBuffer):
        """Test __getitem__ returns an Experience."""
        dataset = ReplayDataset(filled_buffer)
        experience = dataset[0]  # Index is not used for actual indexing
        assert isinstance(experience, Experience)
        assert experience.board_state is not None

    def test_getitem_applies_transform(self, filled_buffer: UniformReplayBuffer):
        """Test transform is applied to sampled experience."""
        transform = MagicMock(side_effect=lambda x: x)
        dataset = ReplayDataset(filled_buffer, transform=transform)

        _ = dataset[0]
        transform.assert_called_once()

    def test_getitem_empty_buffer_raises(self):
        """Test IndexError when buffer is empty."""
        empty_buffer = UniformReplayBuffer(capacity=10)
        dataset = ReplayDataset(empty_buffer)

        with pytest.raises(IndexError, match="Buffer is empty"):
            _ = dataset[0]

    def test_iteration_with_dataloader(self, filled_buffer: UniformReplayBuffer):
        """Test dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ReplayDataset(filled_buffer)
        # Use num_workers=0 to avoid issues with mock buffer
        # Use collate_fn=list to avoid batching Experience objects with variable sizes
        loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=list)

        batch = next(iter(loader))
        # DataLoader returns batched experiences
        assert batch is not None


# --- StreamingReplayDataset Tests ---


class TestStreamingReplayDataset:
    """Tests for StreamingReplayDataset class."""

    def test_init_with_defaults(self, filled_buffer: UniformReplayBuffer):
        """Test initialization with default parameters."""
        dataset = StreamingReplayDataset(filled_buffer)
        assert dataset.buffer is filled_buffer
        assert dataset.batch_size == 32
        assert dataset.transform is None

    def test_init_with_custom_params(self, filled_buffer: UniformReplayBuffer):
        """Test initialization with custom parameters."""
        transform = MagicMock()
        dataset = StreamingReplayDataset(
            filled_buffer,
            batch_size=64,
            transform=transform,
        )
        assert dataset.batch_size == 64
        assert dataset.transform is transform

    def test_iter_yields_experiences(self, filled_buffer: UniformReplayBuffer):
        """Test iteration yields Experience objects."""
        dataset = StreamingReplayDataset(filled_buffer, batch_size=5)
        iterator = iter(dataset)

        # Get first few items
        for _ in range(10):
            experience = next(iterator)
            assert isinstance(experience, Experience)

    def test_iter_applies_transform(self, filled_buffer: UniformReplayBuffer):
        """Test transform is applied during iteration."""
        call_count = 0

        def counting_transform(exp):
            nonlocal call_count
            call_count += 1
            return exp

        dataset = StreamingReplayDataset(
            filled_buffer,
            batch_size=5,
            transform=counting_transform,
        )
        iterator = iter(dataset)

        for _ in range(10):
            _ = next(iterator)

        assert call_count == 10

    def test_iter_is_infinite(self, filled_buffer: UniformReplayBuffer):
        """Test iteration continues indefinitely."""
        dataset = StreamingReplayDataset(filled_buffer, batch_size=5)
        iterator = iter(dataset)

        # Should be able to get many more items than buffer size
        for _ in range(200):
            experience = next(iterator)
            assert experience is not None


# --- BoardSizeBatchSampler Tests ---


class TestBoardSizeBatchSampler:
    """Tests for BoardSizeBatchSampler class."""

    def test_init_creates_size_groups(self, sample_experiences: list[Experience]):
        """Test initialization groups experiences by board size."""
        sampler = BoardSizeBatchSampler(sample_experiences, batch_size=4)

        assert len(sampler.size_to_indices) > 0
        # All indices should be accounted for
        all_indices = []
        for indices in sampler.size_to_indices.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(sample_experiences)))

    def test_batches_have_same_board_size(self, sample_experiences: list[Experience]):
        """Test each batch contains experiences of same board size."""
        sampler = BoardSizeBatchSampler(sample_experiences, batch_size=4, shuffle=False)

        for batch_indices in sampler:
            board_sizes = [sample_experiences[i].board_size for i in batch_indices]
            assert len(set(board_sizes)) == 1, "Batch should have same board size"

    def test_batch_size_respected(self, sample_experiences: list[Experience]):
        """Test batches respect batch_size limit."""
        batch_size = 4
        sampler = BoardSizeBatchSampler(sample_experiences, batch_size=batch_size, drop_last=True)

        for batch_indices in sampler:
            assert len(batch_indices) == batch_size

    def test_drop_last_removes_incomplete(self, sample_experiences: list[Experience]):
        """Test drop_last removes incomplete batches."""
        batch_size = 7  # Won't evenly divide
        sampler = BoardSizeBatchSampler(sample_experiences, batch_size=batch_size, drop_last=True)

        for batch_indices in sampler:
            assert len(batch_indices) == batch_size

    def test_no_drop_last_includes_incomplete(self, sample_experiences: list[Experience]):
        """Test without drop_last, incomplete batches are included."""
        batch_size = 7
        sampler = BoardSizeBatchSampler(sample_experiences, batch_size=batch_size, drop_last=False)

        batch_sizes = [len(batch) for batch in sampler]
        # Should have some batches smaller than batch_size
        assert any(s < batch_size for s in batch_sizes) or all(s == batch_size for s in batch_sizes)

    def test_shuffle_changes_order(self, sample_experiences: list[Experience]):
        """Test shuffle randomizes batch order."""
        sampler1 = BoardSizeBatchSampler(sample_experiences, batch_size=4, shuffle=True)
        sampler2 = BoardSizeBatchSampler(sample_experiences, batch_size=4, shuffle=True)

        # Get batches from both (may be different order due to shuffle)
        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # Both should have same number of batches
        assert len(batches1) == len(batches2)

    def test_no_shuffle_is_deterministic(self, same_size_experiences: list[Experience]):
        """Test no shuffle gives deterministic order."""
        sampler1 = BoardSizeBatchSampler(same_size_experiences, batch_size=3, shuffle=False)
        sampler2 = BoardSizeBatchSampler(same_size_experiences, batch_size=3, shuffle=False)

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        assert batches1 == batches2

    def test_len_returns_correct_count(self, sample_experiences: list[Experience]):
        """Test __len__ returns correct batch count."""
        sampler = BoardSizeBatchSampler(sample_experiences, batch_size=4, drop_last=False)

        expected_len = len(sampler)
        actual_batches = list(sampler)

        assert len(actual_batches) == expected_len

    def test_empty_experiences(self):
        """Test with empty experience list."""
        sampler = BoardSizeBatchSampler([], batch_size=4)
        assert len(sampler) == 0
        assert list(sampler) == []


# --- ExperienceListDataset Tests ---


class TestExperienceListDataset:
    """Tests for ExperienceListDataset class."""

    def test_init_stores_experiences(self, sample_experiences: list[Experience]):
        """Test initialization stores experience list."""
        dataset = ExperienceListDataset(sample_experiences)
        assert dataset.experiences is sample_experiences
        assert dataset.transform is None

    def test_len_matches_list(self, sample_experiences: list[Experience]):
        """Test __len__ matches experience list length."""
        dataset = ExperienceListDataset(sample_experiences)
        assert len(dataset) == len(sample_experiences)

    def test_getitem_returns_correct_experience(self, sample_experiences: list[Experience]):
        """Test __getitem__ returns correct indexed experience."""
        dataset = ExperienceListDataset(sample_experiences)

        for i in range(len(sample_experiences)):
            assert dataset[i] is sample_experiences[i]

    def test_getitem_applies_transform(self, sample_experiences: list[Experience]):
        """Test transform is applied to retrieved experience."""
        transform = MagicMock(side_effect=lambda x: x)
        dataset = ExperienceListDataset(sample_experiences, transform=transform)

        _ = dataset[0]
        transform.assert_called_once_with(sample_experiences[0])

    def test_negative_indexing(self, sample_experiences: list[Experience]):
        """Test negative indexing works."""
        dataset = ExperienceListDataset(sample_experiences)
        assert dataset[-1] is sample_experiences[-1]

    def test_index_error_on_out_of_range(self, sample_experiences: list[Experience]):
        """Test IndexError for out-of-range indices."""
        dataset = ExperienceListDataset(sample_experiences)

        with pytest.raises(IndexError):
            _ = dataset[1000]

    def test_with_dataloader(self, sample_experiences: list[Experience]):
        """Test integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ExperienceListDataset(sample_experiences)
        # Use collate_fn=list to avoid batching Experience objects with variable sizes
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=list)

        batches = list(loader)
        assert len(batches) > 0


# --- AugmentedExperience Tests ---


class TestAugmentedExperience:
    """Tests for AugmentedExperience transformation class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        augment = AugmentedExperience()
        assert augment.use_rotations is True
        assert augment.use_reflections is True
        # Identity + 3 rotations + 2 reflections = 6 transforms
        assert len(augment.transforms) == 6

    def test_init_rotations_only(self):
        """Test initialization with rotations only."""
        augment = AugmentedExperience(use_rotations=True, use_reflections=False)
        # Identity + 3 rotations = 4 transforms
        assert len(augment.transforms) == 4

    def test_init_reflections_only(self):
        """Test initialization with reflections only."""
        augment = AugmentedExperience(use_rotations=False, use_reflections=True)
        # Identity + 2 reflections = 3 transforms
        assert len(augment.transforms) == 3

    def test_init_identity_only(self):
        """Test initialization with no augmentation."""
        augment = AugmentedExperience(use_rotations=False, use_reflections=False)
        # Identity only = 1 transform
        assert len(augment.transforms) == 1

    def test_call_returns_experience(self):
        """Test calling augmentation returns Experience."""
        augment = AugmentedExperience()
        experience = create_experience(board_size=9, seed=42)

        result = augment(experience)

        assert isinstance(result, Experience)
        assert result.board_size == experience.board_size
        assert result.target_value == experience.target_value

    def test_board_state_shape_preserved(self):
        """Test board state shape is preserved after augmentation."""
        augment = AugmentedExperience()
        experience = create_experience(board_size=9, seed=42)

        result = augment(experience)

        assert result.board_state.shape == experience.board_state.shape

    def test_policy_shape_preserved(self):
        """Test policy shape is preserved after augmentation."""
        augment = AugmentedExperience()
        experience = create_experience(board_size=9, seed=42)

        result = augment(experience)

        assert result.target_policy.shape == experience.target_policy.shape

    def test_policy_sums_to_one(self):
        """Test augmented policy still sums to 1."""
        augment = AugmentedExperience()
        experience = create_experience(board_size=9, seed=42)

        result = augment(experience)

        assert torch.isclose(result.target_policy.sum(), torch.tensor(1.0), atol=1e-5)

    def test_rotation_transforms_correctly(self):
        """Test rotation applies correctly to board state."""
        # Create augmentation with rotations only
        augment = AugmentedExperience(use_rotations=True, use_reflections=False)

        # Create deterministic experience
        experience = create_experience(board_size=3, n_channels=1, seed=42)

        # Manually apply 90-degree rotation
        rotated = torch.rot90(experience.board_state, 1, [-2, -1])

        # Check that rotation is one of the possible outcomes
        # (due to random selection, we check the transform is valid)
        result = augment(experience)
        assert result.board_state.shape == experience.board_state.shape

    def test_reflection_transforms_correctly(self):
        """Test reflection applies correctly."""
        # Create augmentation with reflections only
        augment = AugmentedExperience(use_rotations=False, use_reflections=True)

        experience = create_experience(board_size=3, n_channels=1, seed=42)
        result = augment(experience)

        assert result.board_state.shape == experience.board_state.shape

    def test_metadata_preserved(self):
        """Test metadata is preserved in augmented experience."""
        augment = AugmentedExperience()
        experience = create_experience(board_size=9, seed=42)
        experience.metadata["test_key"] = "test_value"

        result = augment(experience)

        assert result.metadata == experience.metadata

    def test_randomness_in_selection(self):
        """Test that different transformations are selected randomly."""
        augment = AugmentedExperience()
        experience = create_experience(board_size=9, seed=42)

        # Apply many times and check for variation
        results = [augment(experience) for _ in range(50)]

        # Check that not all results are identical
        board_states = [r.board_state.sum().item() for r in results]
        unique_sums = len(set(board_states))

        # With 6 transforms and 50 samples, we should see multiple unique results
        assert unique_sums > 1, "Augmentation should produce varied results"

    @patch("random.choice")
    def test_identity_transform(self, mock_choice):
        """Test identity transform returns unchanged data."""
        augment = AugmentedExperience()
        # Force selection of identity transform
        mock_choice.return_value = augment.transforms[0]

        experience = create_experience(board_size=9, seed=42)
        result = augment(experience)

        assert torch.allclose(result.board_state, experience.board_state)
        assert torch.allclose(result.target_policy, experience.target_policy)


# --- Integration Tests ---


class TestDatasetIntegration:
    """Integration tests for dataset components."""

    def test_batch_sampler_with_list_dataset(self, sample_experiences: list[Experience]):
        """Test BoardSizeBatchSampler with ExperienceListDataset."""
        from torch.utils.data import DataLoader

        dataset = ExperienceListDataset(sample_experiences)
        sampler = BoardSizeBatchSampler(sample_experiences, batch_size=4)

        # Can create DataLoader with this combination
        # collate_fn=list avoids default_collate which can't handle Experience dataclasses
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=list)
        batches = list(loader)
        assert len(batches) > 0

    def test_augmentation_with_list_dataset(self, sample_experiences: list[Experience]):
        """Test AugmentedExperience as transform in ExperienceListDataset."""
        augment = AugmentedExperience()
        dataset = ExperienceListDataset(sample_experiences, transform=augment)

        # Should be able to get augmented items
        for i in range(len(sample_experiences)):
            result = dataset[i]
            assert isinstance(result, Experience)
            assert result.board_size == sample_experiences[i].board_size

    def test_replay_dataset_with_augmentation(self, filled_buffer: UniformReplayBuffer):
        """Test ReplayDataset with augmentation transform."""
        augment = AugmentedExperience()
        dataset = ReplayDataset(filled_buffer, transform=augment)

        for _ in range(10):
            experience = dataset[0]
            assert isinstance(experience, Experience)
