"""Tests for temperature scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.core.types import ActionType, ElementID, TemperatureScheduleType
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.mcts.temperature import TemperatureSchedule


class TestTemperatureSchedule:
    """Unit tests for TemperatureSchedule."""

    def test_constant_schedule(self) -> None:
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.CONSTANT,
            initial=1.0,
        )
        assert sched.get_temperature(0) == 1.0
        # After decay_steps (default=30), returns final (default=0.1)
        assert sched.get_temperature(100) == 0.1

    def test_step_schedule_before_threshold(self) -> None:
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.STEP,
            initial=1.0,
            final=0.1,
            decay_steps=30,
        )
        assert sched.get_temperature(0) == 1.0
        assert sched.get_temperature(29) == 1.0

    def test_step_schedule_after_threshold(self) -> None:
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.STEP,
            initial=1.0,
            final=0.1,
            decay_steps=30,
        )
        assert sched.get_temperature(30) == 0.1
        assert sched.get_temperature(100) == 0.1

    def test_linear_schedule_decreases(self) -> None:
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.LINEAR,
            initial=1.0,
            final=0.01,
            decay_steps=100,
        )
        t0 = sched.get_temperature(0)
        t50 = sched.get_temperature(50)
        t100 = sched.get_temperature(100)
        assert t0 > t50 > t100

    def test_linear_schedule_endpoints(self) -> None:
        """Linear schedule starts at initial and ends at final."""
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.LINEAR,
            initial=1.0,
            final=0.1,
            decay_steps=100,
        )
        assert sched.get_temperature(0) == pytest.approx(1.0, rel=1e-5)
        assert sched.get_temperature(100) == pytest.approx(0.1, rel=1e-5)

    def test_linear_schedule_monotonic(self) -> None:
        """Linear schedule monotonically decreases."""
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.LINEAR,
            initial=1.0,
            final=0.1,
            decay_steps=50,
        )
        temps = [sched.get_temperature(s) for s in range(51)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]

    def test_exponential_schedule_decreases(self) -> None:
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.EXPONENTIAL,
            initial=1.0,
            final=0.01,
            decay_steps=100,
        )
        t0 = sched.get_temperature(0)
        t50 = sched.get_temperature(50)
        assert t0 > t50

    def test_exponential_schedule_endpoints(self) -> None:
        """Exponential schedule reaches final at decay_steps."""
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.EXPONENTIAL,
            initial=1.0,
            final=0.01,
            decay_steps=100,
        )
        assert sched.get_temperature(0) == pytest.approx(1.0, rel=1e-3)
        assert sched.get_temperature(100) == pytest.approx(0.01, rel=1e-3)

    def test_invalid_initial_raises(self) -> None:
        """Negative or zero initial temperature raises ValueError."""
        with pytest.raises(ValueError, match="initial must be positive"):
            TemperatureSchedule(initial=0.0)

    def test_invalid_final_raises(self) -> None:
        """Negative or zero final temperature raises ValueError."""
        with pytest.raises(ValueError, match="final must be positive"):
            TemperatureSchedule(final=-0.1)

    def test_invalid_decay_steps_raises(self) -> None:
        """Zero or negative decay_steps raises ValueError."""
        with pytest.raises(ValueError, match="decay_steps must be >= 1"):
            TemperatureSchedule(decay_steps=0)

    def test_properties(self) -> None:
        """Properties expose schedule parameters."""
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.LINEAR,
            initial=1.0,
            final=0.1,
            decay_steps=30,
        )
        assert sched.schedule_type == TemperatureScheduleType.LINEAR
        assert sched.initial == 1.0
        assert sched.final == 0.1
        assert sched.decay_steps == 30

    def test_select_action_with_temperature(self) -> None:
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.CONSTANT,
            initial=1.0,
        )
        rng = np.random.default_rng(42)
        a1 = Action(element_id=ElementID("e0"), action_type=ActionType.NO_OP)
        a2 = Action(element_id=ElementID("e1"), action_type=ActionType.P_REFINE)
        visit_counts = {a1: 100, a2: 10}
        selected = sched.select_action_with_temperature(visit_counts, 1.0, rng)
        assert selected in (a1, a2)

    def test_low_temperature_selects_best(self) -> None:
        """Very low temperature should almost always pick highest-visit action."""
        sched = TemperatureSchedule(
            schedule_type=TemperatureScheduleType.CONSTANT,
            initial=0.01,
        )
        rng = np.random.default_rng(42)
        a1 = Action(element_id=ElementID("e0"), action_type=ActionType.NO_OP)
        a2 = Action(element_id=ElementID("e1"), action_type=ActionType.P_REFINE)
        visit_counts = {a1: 1000, a2: 1}
        selections = [
            sched.select_action_with_temperature(visit_counts, 0.01, rng) for _ in range(20)
        ]
        # Most selections should be a1
        assert selections.count(a1) > 15

    def test_deterministic_selection_at_zero_temp(self) -> None:
        """Near-zero temperature selects the most visited action."""
        sched = TemperatureSchedule()
        visit_counts = {"a": 100, "b": 10, "c": 1}
        action = sched.select_action_with_temperature(visit_counts, temperature=1e-10)
        assert action == "a"
