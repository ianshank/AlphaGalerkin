"""Tests for dataset and collation modules."""

from __future__ import annotations

import pytest
import torch

from src.data.collate import SameSizeCollator, TrainingBatch, VariableSizeCollator
from src.data.dataset import AugmentedExperience, BoardSizeBatchSampler, ExperienceListDataset
from src.training.replay_buffer import Experience


def create_experience(board_size: int, value: float = 0.0) -> Experience:
    """Create a test experience."""
    n_channels = 17
    n_actions = board_size**2 + 1

    return Experience(
        board_state=torch.randn(n_channels, board_size, board_size),
        board_size=board_size,
        target_policy=torch.softmax(torch.randn(n_actions), dim=0),
        target_value=value,
    )


class TestVariableSizeCollator:
    """Tests for VariableSizeCollator."""

    def test_collate_same_size(self) -> None:
        """Test collating experiences with same board size."""
        collator = VariableSizeCollator()
        experiences = [create_experience(9) for _ in range(4)]

        batch = collator(experiences)

        assert isinstance(batch, TrainingBatch)
        assert batch.board_states.shape == (4, 17, 9, 9)
        assert batch.target_policies.shape == (4, 82)  # 9*9 + 1
        assert batch.target_values.shape == (4, 1)
        assert batch.position_mask.shape == (4, 9, 9)
        assert batch.action_mask.shape == (4, 82)

    def test_collate_different_sizes(self) -> None:
        """Test collating experiences with different board sizes."""
        collator = VariableSizeCollator()
        experiences = [
            create_experience(9),
            create_experience(13),
            create_experience(19),
        ]

        batch = collator(experiences)

        # Should pad to max size (19)
        assert batch.board_states.shape == (3, 17, 19, 19)
        max_actions = 19 * 19 + 1
        assert batch.target_policies.shape == (3, max_actions)
        assert batch.position_mask.shape == (3, 19, 19)

        # Check masks are correct
        assert batch.position_mask[0, :9, :9].all()  # 9x9 valid
        assert not batch.position_mask[0, 10, 10]  # Outside 9x9 is masked
        assert batch.position_mask[2, :19, :19].all()  # 19x19 all valid

    def test_collate_preserves_data(self) -> None:
        """Test that collation preserves original data."""
        collator = VariableSizeCollator()

        exp = create_experience(9, value=0.7)
        batch = collator([exp])

        # Check board state is preserved (first 9x9)
        assert torch.allclose(
            batch.board_states[0, :, :9, :9],
            exp.board_state,
            atol=1e-6,
        )

        # Check value is preserved
        assert batch.target_values[0, 0].item() == pytest.approx(0.7)

    def test_collate_empty_raises(self) -> None:
        """Test that empty list raises error."""
        collator = VariableSizeCollator()

        with pytest.raises(ValueError, match="Cannot collate empty"):
            collator([])

    def test_batch_to_device(self) -> None:
        """Test moving batch to device."""
        collator = VariableSizeCollator()
        experiences = [create_experience(9) for _ in range(2)]

        batch = collator(experiences)
        batch_moved = batch.to(torch.device("cpu"))

        assert batch_moved.board_states.device == torch.device("cpu")

    def test_max_board_size_parameter(self) -> None:
        """Test max_board_size parameter for fixed padding."""
        collator = VariableSizeCollator(max_board_size=19)
        experiences = [create_experience(9) for _ in range(2)]

        batch = collator(experiences)

        # Should pad to max_board_size even though all are 9x9
        assert batch.board_states.shape == (2, 17, 19, 19)


class TestSameSizeCollator:
    """Tests for SameSizeCollator."""

    def test_collate_same_size(self) -> None:
        """Test collating with same sizes."""
        collator = SameSizeCollator()
        experiences = [create_experience(9) for _ in range(4)]

        batch = collator(experiences)

        assert batch.board_states.shape == (4, 17, 9, 9)
        assert batch.position_mask.all()  # All positions valid

    def test_collate_different_sizes_raises(self) -> None:
        """Test that different sizes raise error."""
        collator = SameSizeCollator()
        experiences = [create_experience(9), create_experience(13)]

        with pytest.raises(ValueError, match="Expected all experiences"):
            collator(experiences)


class TestExperienceListDataset:
    """Tests for ExperienceListDataset."""

    def test_dataset_len(self) -> None:
        """Test dataset length."""
        experiences = [create_experience(9) for _ in range(10)]
        dataset = ExperienceListDataset(experiences)

        assert len(dataset) == 10

    def test_dataset_getitem(self) -> None:
        """Test indexed access."""
        experiences = [create_experience(9, value=float(i)) for i in range(5)]
        dataset = ExperienceListDataset(experiences)

        exp = dataset[2]
        assert exp.target_value == pytest.approx(2.0)

    def test_dataset_with_transform(self) -> None:
        """Test with transform function."""
        experiences = [create_experience(9) for _ in range(5)]

        def transform(exp: Experience) -> Experience:
            return Experience(
                board_state=exp.board_state * 2,
                board_size=exp.board_size,
                target_policy=exp.target_policy,
                target_value=exp.target_value * 2,
                metadata=exp.metadata,
            )

        dataset = ExperienceListDataset(experiences, transform=transform)
        exp = dataset[0]

        assert exp.target_value == pytest.approx(experiences[0].target_value * 2)


class TestBoardSizeBatchSampler:
    """Tests for BoardSizeBatchSampler."""

    def test_groups_by_size(self) -> None:
        """Test that batches contain same board size."""
        experiences = [
            create_experience(9),
            create_experience(9),
            create_experience(13),
            create_experience(13),
            create_experience(13),
        ]

        sampler = BoardSizeBatchSampler(experiences, batch_size=2, shuffle=False)
        batches = list(sampler)

        # Check each batch has same size
        for batch in batches:
            sizes = [experiences[i].board_size for i in batch]
            assert len(set(sizes)) == 1

    def test_batch_size_respected(self) -> None:
        """Test that batch size is respected."""
        experiences = [create_experience(9) for _ in range(10)]
        sampler = BoardSizeBatchSampler(experiences, batch_size=3, drop_last=True)

        batches = list(sampler)
        assert all(len(b) == 3 for b in batches)

    def test_drop_last(self) -> None:
        """Test drop_last parameter."""
        experiences = [create_experience(9) for _ in range(7)]

        sampler_drop = BoardSizeBatchSampler(experiences, batch_size=3, drop_last=True)
        sampler_keep = BoardSizeBatchSampler(experiences, batch_size=3, drop_last=False)

        batches_drop = list(sampler_drop)
        batches_keep = list(sampler_keep)

        assert len(batches_drop) == 2  # 6 items in full batches
        assert len(batches_keep) == 3  # Includes partial batch


class TestAugmentedExperience:
    """Tests for data augmentation."""

    def test_augmentation_preserves_size(self) -> None:
        """Test that augmentation preserves tensor sizes."""
        augmenter = AugmentedExperience()
        exp = create_experience(9)

        augmented = augmenter(exp)

        assert augmented.board_state.shape == exp.board_state.shape
        assert augmented.target_policy.shape == exp.target_policy.shape
        assert augmented.board_size == exp.board_size

    def test_augmentation_changes_data(self) -> None:
        """Test that augmentation can change data."""
        augmenter = AugmentedExperience(use_rotations=True, use_reflections=True)
        exp = create_experience(9)

        # Run many times - at least some should differ
        changed = False
        for _ in range(10):
            augmented = augmenter(exp)
            if not torch.allclose(augmented.board_state, exp.board_state):
                changed = True
                break

        assert changed, "Augmentation should sometimes change the data"

    def test_policy_consistency(self) -> None:
        """Test that policy sums are preserved after augmentation."""
        augmenter = AugmentedExperience()
        exp = create_experience(9)

        for _ in range(5):
            augmented = augmenter(exp)
            # Policy should still sum to ~1
            assert abs(augmented.target_policy.sum().item() - 1.0) < 0.01
