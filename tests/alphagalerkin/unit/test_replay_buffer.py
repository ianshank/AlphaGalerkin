"""Tests for replay buffer."""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.core.config import ReplayConfig
from src.alphagalerkin.training.replay_buffer import (
    Experience,
    ReplayBuffer,
)


def _make_experience(value: float = 0.5) -> Experience:
    """Helper to create a minimal experience."""
    return Experience(
        state_features=np.zeros(8),
        policy_target={},
        value_target=value,
    )


class TestReplayBuffer:
    """Core replay buffer behaviour."""

    def test_add_and_size(self) -> None:
        config = ReplayConfig(
            capacity=1000, min_size_to_train=2,
        )
        buf = ReplayBuffer(config)
        assert buf.size == 0
        buf.add(_make_experience())
        assert buf.size == 1

    def test_is_ready_respects_min_size(self) -> None:
        config = ReplayConfig(
            capacity=1000, min_size_to_train=3,
        )
        buf = ReplayBuffer(config)
        for i in range(2):
            buf.add(_make_experience())
        assert not buf.is_ready
        buf.add(_make_experience())
        assert buf.is_ready

    def test_sample_returns_correct_size(self) -> None:
        config = ReplayConfig(
            capacity=10000, min_size_to_train=5,
        )
        buf = ReplayBuffer(config)
        for i in range(10):
            buf.add(_make_experience(float(i)))
        batch = buf.sample(4)
        assert len(batch) == 4

    def test_capacity_limit_respected(self) -> None:
        config = ReplayConfig(
            capacity=5000, min_size_to_train=1,
        )
        buf = ReplayBuffer(config)
        for i in range(6000):
            buf.add(_make_experience(float(i)))
        assert buf.size <= 5000

    def test_sample_before_ready_raises(self) -> None:
        config = ReplayConfig(
            capacity=10000, min_size_to_train=5000,
        )
        buf = ReplayBuffer(config)
        buf.add(_make_experience())
        with pytest.raises(RuntimeError, match="not ready"):
            buf.sample(1)

    def test_add_batch(self) -> None:
        config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )
        buf = ReplayBuffer(config)
        batch = [_make_experience(float(i)) for i in range(5)]
        buf.add_batch(batch)
        assert buf.size == 5

    def test_clear_empties_buffer(self) -> None:
        config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )
        buf = ReplayBuffer(config)
        for _ in range(10):
            buf.add(_make_experience())
        buf.clear()
        assert buf.size == 0
        assert not buf.is_ready

    def test_get_state_and_load_state(self) -> None:
        config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )
        buf = ReplayBuffer(config)
        for i in range(5):
            buf.add(_make_experience(float(i)))

        state = buf.get_state()
        assert "experiences" in state
        assert "priorities" in state

        buf2 = ReplayBuffer(config)
        buf2.load_state(state)
        assert buf2.size == 5

    def test_update_priorities(self) -> None:
        config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )
        buf = ReplayBuffer(config)
        for _ in range(5):
            buf.add(_make_experience())
        buf.update_priorities([0, 1], [10.0, 20.0])
        # No crash, priorities updated internally

    def test_sample_clamps_to_buffer_size(self) -> None:
        config = ReplayConfig(
            capacity=1000, min_size_to_train=1,
        )
        buf = ReplayBuffer(config)
        for _ in range(3):
            buf.add(_make_experience())
        batch = buf.sample(100)
        assert len(batch) <= buf.size
