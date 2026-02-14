"""Tests for curriculum learning."""
from __future__ import annotations

from src.alphagalerkin.core.config import CurriculumConfig
from src.alphagalerkin.training.curriculum import CurriculumManager


class TestCurriculumManager:
    """Tests for progressive difficulty management."""

    def test_initial_stage_is_zero(self) -> None:
        config = CurriculumConfig(enabled=True)
        cm = CurriculumManager(config)
        assert cm.current_stage_index == 0

    def test_disabled_curriculum_never_advances(self) -> None:
        config = CurriculumConfig(enabled=False)
        cm = CurriculumManager(config)
        promoted = cm.update(1.0)
        assert not promoted
        assert cm.current_stage_index == 0

    def test_advance_on_high_win_rate(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stages=[
                {"max_dof": 100},
                {"max_dof": 500},
                {"max_dof": 1000},
            ],
            advance_threshold=0.8,
            evaluation_window=10,
        )
        cm = CurriculumManager(config)
        promoted = cm.update(0.9)
        assert promoted
        assert cm.current_stage_index == 1

    def test_no_advance_on_low_win_rate(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stages=[
                {"max_dof": 100},
                {"max_dof": 500},
            ],
            advance_threshold=0.8,
            evaluation_window=10,
        )
        cm = CurriculumManager(config)
        promoted = cm.update(0.5)
        assert not promoted
        assert cm.current_stage_index == 0

    def test_stops_at_final_stage(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stages=[
                {"max_dof": 100},
                {"max_dof": 500},
            ],
            advance_threshold=0.5,
            evaluation_window=10,
        )
        cm = CurriculumManager(config)
        cm.update(0.9)  # advance to stage 1
        promoted = cm.update(0.9)  # try to advance again
        assert not promoted
        assert cm.is_at_final_stage

    def test_is_at_final_stage_with_no_stages(self) -> None:
        config = CurriculumConfig(enabled=True, stages=[])
        cm = CurriculumManager(config)
        assert cm.is_at_final_stage

    def test_num_stages(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stages=[{"a": 1}, {"a": 2}, {"a": 3}],
        )
        cm = CurriculumManager(config)
        assert cm.num_stages == 3

    def test_num_stages_empty(self) -> None:
        config = CurriculumConfig(enabled=True, stages=[])
        cm = CurriculumManager(config)
        assert cm.num_stages == 1  # max(1, 0)

    def test_get_stage_overrides(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stages=[
                {"max_dof": 100, "max_steps": 5},
                {"max_dof": 500, "max_steps": 10},
            ],
            advance_threshold=0.5,
            evaluation_window=10,
        )
        cm = CurriculumManager(config)
        overrides = cm.get_stage_overrides()
        assert overrides["max_dof"] == 100

    def test_get_stage_overrides_disabled(self) -> None:
        config = CurriculumConfig(enabled=False)
        cm = CurriculumManager(config)
        assert cm.get_stage_overrides() == {}

    def test_reset(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stages=[
                {"a": 1},
                {"a": 2},
            ],
            advance_threshold=0.5,
            evaluation_window=10,
        )
        cm = CurriculumManager(config)
        cm.update(0.9)
        assert cm.current_stage_index == 1
        cm.reset()
        assert cm.current_stage_index == 0

    def test_windowed_evaluation(self) -> None:
        """Advancement uses windowed average, not single value."""
        config = CurriculumConfig(
            enabled=True,
            stages=[{"a": 1}, {"a": 2}],
            advance_threshold=0.8,
            evaluation_window=10,
        )
        cm = CurriculumManager(config)
        # Low win rates should not trigger advancement
        for _ in range(5):
            cm.update(0.5)
        assert cm.current_stage_index == 0
        # High win rates to push average above threshold
        for _ in range(10):
            promoted = cm.update(1.0)
        assert cm.current_stage_index == 1
