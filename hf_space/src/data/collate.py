"""Collation functions for variable-size board batching.

Handles batching of experiences with different board sizes through
padding and masking strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.training.replay_buffer import Experience


@dataclass
class TrainingBatch:
    """Batched training data with padding information.

    Attributes:
        board_states: Padded board states (batch, channels, max_h, max_w).
        board_sizes: Original board sizes per sample.
        target_policies: Padded target policies (batch, max_actions).
        target_values: Target values (batch, 1).
        position_mask: Mask for valid board positions.
        action_mask: Mask for valid actions in policy.

    """

    board_states: Tensor  # Float[batch, channels, max_h, max_w]
    board_sizes: Tensor  # Int[batch]
    target_policies: Tensor  # Float[batch, max_actions]
    target_values: Tensor  # Float[batch, 1]
    position_mask: Tensor  # Bool[batch, max_h, max_w]
    action_mask: Tensor  # Bool[batch, max_actions]

    def to(self, device: torch.device) -> TrainingBatch:
        """Move batch to device.

        Args:
            device: Target device.

        Returns:
            New TrainingBatch on device.

        """
        return TrainingBatch(
            board_states=self.board_states.to(device),
            board_sizes=self.board_sizes.to(device),
            target_policies=self.target_policies.to(device),
            target_values=self.target_values.to(device),
            position_mask=self.position_mask.to(device),
            action_mask=self.action_mask.to(device),
        )

    def pin_memory(self) -> TrainingBatch:
        """Pin batch memory for faster GPU transfer.

        Returns:
            New TrainingBatch with pinned memory.

        """
        return TrainingBatch(
            board_states=self.board_states.pin_memory(),
            board_sizes=self.board_sizes.pin_memory(),
            target_policies=self.target_policies.pin_memory(),
            target_values=self.target_values.pin_memory(),
            position_mask=self.position_mask.pin_memory(),
            action_mask=self.action_mask.pin_memory(),
        )

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.board_states.size(0)


class VariableSizeCollator:
    """Collator for variable-size board batching.

    Pads board states and policies to the maximum size in the batch,
    creating appropriate masks for the model to ignore padded regions.
    """

    def __init__(
        self,
        pad_value: float = 0.0,
        max_board_size: int | None = None,
    ) -> None:
        """Initialize collator.

        Args:
            pad_value: Value to use for padding.
            max_board_size: Maximum board size (for fixed-size padding).

        """
        self.pad_value = pad_value
        self.max_board_size = max_board_size

    def __call__(self, experiences: list[Experience]) -> TrainingBatch:
        """Collate experiences into a batch.

        Args:
            experiences: List of experiences to batch.

        Returns:
            TrainingBatch with padded tensors and masks.

        """
        if not experiences:
            raise ValueError("Cannot collate empty list of experiences")

        # Determine dimensions
        batch_size = len(experiences)
        n_channels = experiences[0].board_state.size(0)

        # Find max board size in batch
        board_sizes = [exp.board_size for exp in experiences]
        max_size = max(board_sizes)
        if self.max_board_size is not None:
            max_size = max(max_size, self.max_board_size)

        # Max actions = max_size^2 + 1 (pass)
        max_actions = max_size**2 + 1

        # Initialize tensors
        board_states = torch.full(
            (batch_size, n_channels, max_size, max_size),
            self.pad_value,
            dtype=torch.float32,
        )
        target_policies = torch.zeros(batch_size, max_actions, dtype=torch.float32)
        target_values = torch.zeros(batch_size, 1, dtype=torch.float32)
        position_mask = torch.zeros(batch_size, max_size, max_size, dtype=torch.bool)
        action_mask = torch.zeros(batch_size, max_actions, dtype=torch.bool)

        # Fill tensors
        for i, exp in enumerate(experiences):
            size = exp.board_size

            # Copy board state (with potential padding)
            board_states[i, :, :size, :size] = exp.board_state

            # Copy policy (remap to padded action space)
            # Original: [0..size^2-1, pass=size^2]
            # Padded: [0..max_size^2-1, pass=max_size^2]
            n_positions = size**2

            # Copy position policies
            for row in range(size):
                for col in range(size):
                    orig_idx = row * size + col
                    new_idx = row * max_size + col
                    target_policies[i, new_idx] = exp.target_policy[orig_idx]

            # Copy pass move probability
            target_policies[i, max_size**2] = exp.target_policy[n_positions]

            # Set value
            target_values[i, 0] = exp.target_value

            # Set masks
            position_mask[i, :size, :size] = True
            for row in range(size):
                for col in range(size):
                    new_idx = row * max_size + col
                    action_mask[i, new_idx] = True
            action_mask[i, max_size**2] = True  # Pass always valid

        return TrainingBatch(
            board_states=board_states,
            board_sizes=torch.tensor(board_sizes, dtype=torch.int64),
            target_policies=target_policies,
            target_values=target_values,
            position_mask=position_mask,
            action_mask=action_mask,
        )


class SameSizeCollator:
    """Collator for batches with same board size.

    More efficient than VariableSizeCollator when all experiences
    in a batch have the same board size (no padding needed).
    """

    def __call__(self, experiences: list[Experience]) -> TrainingBatch:
        """Collate experiences with same board size.

        Args:
            experiences: List of experiences (must have same board size).

        Returns:
            TrainingBatch without padding.

        Raises:
            ValueError: If experiences have different board sizes.

        """
        if not experiences:
            raise ValueError("Cannot collate empty list of experiences")

        # Verify same board size
        board_size = experiences[0].board_size
        for exp in experiences:
            if exp.board_size != board_size:
                raise ValueError(
                    f"Expected all experiences to have board size {board_size}, "
                    f"got {exp.board_size}"
                )

        batch_size = len(experiences)
        n_actions = board_size**2 + 1

        # Stack tensors directly
        board_states = torch.stack([exp.board_state for exp in experiences])
        target_policies = torch.stack([exp.target_policy for exp in experiences])
        target_values = torch.tensor(
            [[exp.target_value] for exp in experiences],
            dtype=torch.float32,
        )

        # All positions and actions are valid
        position_mask = torch.ones(batch_size, board_size, board_size, dtype=torch.bool)
        action_mask = torch.ones(batch_size, n_actions, dtype=torch.bool)

        return TrainingBatch(
            board_states=board_states,
            board_sizes=torch.full((batch_size,), board_size, dtype=torch.int64),
            target_policies=target_policies,
            target_values=target_values,
            position_mask=position_mask,
            action_mask=action_mask,
        )


def create_collator(
    variable_size: bool = True,
    pad_value: float = 0.0,
    max_board_size: int | None = None,
) -> VariableSizeCollator | SameSizeCollator:
    """Factory function to create appropriate collator.

    Args:
        variable_size: Whether to expect variable board sizes.
        pad_value: Padding value for variable-size collator.
        max_board_size: Maximum board size for padding.

    Returns:
        Configured collator.

    """
    if variable_size:
        return VariableSizeCollator(pad_value=pad_value, max_board_size=max_board_size)
    return SameSizeCollator()
