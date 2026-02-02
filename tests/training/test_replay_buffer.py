"""Tests for replay buffer implementations."""

from __future__ import annotations

import threading
import time

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.training.replay_buffer import (
    Experience,
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
    create_replay_buffer,
)


def create_test_experience(board_size: int = 9, value: float = 0.0) -> Experience:
    """Create a test experience."""
    n_channels = 17
    n_actions = board_size**2 + 1

    return Experience(
        board_state=torch.randn(n_channels, board_size, board_size),
        board_size=board_size,
        target_policy=torch.softmax(torch.randn(n_actions), dim=0),
        target_value=value,
        metadata={"board_size": board_size},
    )


class TestExperience:
    """Tests for Experience dataclass."""

    def test_experience_creation(self) -> None:
        """Test experience creation."""
        exp = create_test_experience(9, 0.5)

        assert exp.board_size == 9
        assert exp.board_state.shape == (17, 9, 9)
        assert exp.target_policy.shape == (82,)
        assert exp.target_value == 0.5

    def test_experience_to_device(self) -> None:
        """Test moving experience to device."""
        exp = create_test_experience()
        device = torch.device("cpu")

        exp_moved = exp.to(device)

        assert exp_moved.board_state.device == device
        assert exp_moved.target_policy.device == device


class TestUniformReplayBuffer:
    """Tests for UniformReplayBuffer."""

    def test_add_and_sample(self) -> None:
        """Test basic add and sample operations."""
        buffer = UniformReplayBuffer(capacity=100)

        # Add experiences
        for i in range(10):
            exp = create_test_experience(9, float(i) / 10)
            buffer.add(exp)

        assert len(buffer) == 10

        # Sample
        samples = buffer.sample(5)
        assert len(samples) == 5
        assert all(isinstance(s, Experience) for s in samples)

    def test_capacity_limit(self) -> None:
        """Test that buffer respects capacity."""
        capacity = 10
        buffer = UniformReplayBuffer(capacity=capacity)

        # Add more than capacity
        for i in range(20):
            buffer.add(create_test_experience())

        assert len(buffer) == capacity

    def test_add_batch(self) -> None:
        """Test batch addition."""
        buffer = UniformReplayBuffer(capacity=100)

        experiences = [create_test_experience() for _ in range(10)]
        buffer.add_batch(experiences)

        assert len(buffer) == 10

    def test_sample_empty_buffer(self) -> None:
        """Test sampling from empty buffer."""
        buffer = UniformReplayBuffer(capacity=100)
        samples = buffer.sample(5)
        assert samples == []

    def test_sample_more_than_buffer(self) -> None:
        """Test sampling more items than in buffer."""
        buffer = UniformReplayBuffer(capacity=100)
        buffer.add(create_test_experience())
        buffer.add(create_test_experience())

        samples = buffer.sample(10)
        assert len(samples) == 2

    def test_clear(self) -> None:
        """Test buffer clearing."""
        buffer = UniformReplayBuffer(capacity=100)
        buffer.add_batch([create_test_experience() for _ in range(10)])
        buffer.clear()
        assert len(buffer) == 0

    def test_get_stats(self) -> None:
        """Test buffer statistics."""
        buffer = UniformReplayBuffer(capacity=100)

        for i in range(5):
            buffer.add(create_test_experience(9, float(i) / 5))
        for i in range(3):
            buffer.add(create_test_experience(19, float(i) / 3))

        stats = buffer.get_stats()
        assert stats["size"] == 8
        assert stats["capacity"] == 100
        assert stats["board_sizes"]["unique"] == 2

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        buffer = UniformReplayBuffer(capacity=1000)
        n_threads = 4
        n_ops_per_thread = 100

        def writer() -> None:
            for _ in range(n_ops_per_thread):
                buffer.add(create_test_experience())
                time.sleep(0.001)

        def reader() -> None:
            for _ in range(n_ops_per_thread):
                buffer.sample(5)
                time.sleep(0.001)

        threads = []
        for _ in range(n_threads):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash and buffer should have items
        assert len(buffer) > 0

    @given(capacity=st.integers(10, 1000), n_items=st.integers(1, 500))
    @settings(max_examples=20)
    def test_buffer_never_exceeds_capacity(self, capacity: int, n_items: int) -> None:
        """Property: buffer size never exceeds capacity."""
        buffer = UniformReplayBuffer(capacity=capacity)

        for _ in range(n_items):
            buffer.add(create_test_experience())

        assert len(buffer) <= capacity


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    def test_add_and_sample(self) -> None:
        """Test basic add and sample with priorities."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        for i in range(10):
            buffer.add(create_test_experience(), priority=float(i + 1))

        assert len(buffer) == 10

        experiences, indices = buffer.sample(5)
        assert len(experiences) == 5
        assert len(indices) == 5

    def test_priority_sampling_bias(self) -> None:
        """Test that higher priority items are sampled more often."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)

        # Add with varying priorities
        for i in range(10):
            exp = create_test_experience(value=float(i))
            # Higher index = higher priority
            buffer.add(exp, priority=float(i + 1))

        # Sample many times and track values
        n_samples = 1000
        values = []
        for _ in range(n_samples):
            experiences, _ = buffer.sample(1)
            if experiences:
                values.append(experiences[0].target_value)

        # Higher priority items should be sampled more
        # Average value should be above midpoint
        avg_value = sum(values) / len(values)
        assert avg_value > 4.5, "Priority sampling should bias toward high-priority items"

    def test_importance_weights(self) -> None:
        """Test importance sampling weights."""
        buffer = PrioritizedReplayBuffer(capacity=100, beta=0.5)

        for i in range(10):
            buffer.add(create_test_experience(), priority=float(i + 1))

        experiences, indices, weights = buffer.sample(5, return_weights=True)

        assert len(weights) == 5
        assert weights.max() == 1.0  # Normalized
        assert all(w > 0 for w in weights)

    def test_priority_update(self) -> None:
        """Test updating priorities after sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        for _ in range(10):
            buffer.add(create_test_experience())

        experiences, indices = buffer.sample(5)

        # Update with new priorities
        new_priorities = [1.0, 2.0, 3.0, 4.0, 5.0]
        buffer.update_priorities(indices, new_priorities)

        # Should not raise error


class TestCreateReplayBuffer:
    """Tests for factory function."""

    def test_create_uniform(self) -> None:
        """Test creating uniform buffer."""
        buffer = create_replay_buffer(capacity=100, prioritized=False)
        assert isinstance(buffer, UniformReplayBuffer)

    def test_create_prioritized(self) -> None:
        """Test creating prioritized buffer."""
        buffer = create_replay_buffer(capacity=100, prioritized=True)
        assert isinstance(buffer, PrioritizedReplayBuffer)
