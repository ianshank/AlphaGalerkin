"""Tests for cost tracking and estimation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from src.vertex.config import AcceleratorType, VertexMachineType
from src.vertex.cost import (
    ACCELERATOR_HOURLY_RATES,
    MACHINE_HOURLY_RATES,
    SPOT_DISCOUNT_FACTOR,
    CostBreakdown,
    CostEstimate,
    CostTracker,
    estimate_job_cost,
    format_cost_table,
    get_hourly_rate,
)

if TYPE_CHECKING:
    pass


class TestCostEstimate:
    """Tests for CostEstimate."""

    def test_creation(self) -> None:
        """Test estimate creation."""
        estimate = CostEstimate(
            machine_cost_per_hour=3.67,
            accelerator_cost_per_hour=2.93,
            total_cost_per_hour=6.60,
            estimated_total_cost=158.40,
            duration_hours=24.0,
            is_spot=False,
        )
        assert estimate.machine_cost_per_hour == 3.67
        assert estimate.estimated_total_cost == 158.40

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        estimate = CostEstimate(
            machine_cost_per_hour=1.0,
            accelerator_cost_per_hour=2.0,
            total_cost_per_hour=3.0,
            estimated_total_cost=72.0,
            duration_hours=24.0,
            is_spot=True,
            discount_applied=0.7,
        )
        d = estimate.to_dict()
        assert d["machine_cost_per_hour"] == 1.0
        assert d["is_spot"] is True
        assert d["discount_applied"] == 70.0  # Percentage

    def test_format_summary(self) -> None:
        """Test formatted summary."""
        estimate = CostEstimate(
            machine_cost_per_hour=3.67,
            accelerator_cost_per_hour=0.0,
            total_cost_per_hour=3.67,
            estimated_total_cost=88.08,
            duration_hours=24.0,
            is_spot=False,
        )
        summary = estimate.format_summary()
        assert "Machine: $3.67/hr" in summary
        assert "Duration: 24.0 hours" in summary
        assert "$88.08" in summary


class TestCostBreakdown:
    """Tests for CostBreakdown."""

    def test_total_cost(self) -> None:
        """Test total cost calculation."""
        breakdown = CostBreakdown(
            compute_cost=50.0,
            accelerator_cost=100.0,
            network_cost=5.0,
            storage_cost=2.0,
        )
        assert breakdown.total_cost == 157.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        breakdown = CostBreakdown(
            compute_cost=50.0,
            accelerator_cost=100.0,
        )
        d = breakdown.to_dict()
        assert d["compute_cost"] == 50.0
        assert d["accelerator_cost"] == 100.0
        assert d["total_cost"] == 150.0


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        """Create tracker instance."""
        return CostTracker()

    def test_initialization(self, tracker: CostTracker) -> None:
        """Test tracker initialization."""
        assert tracker._start_time is None
        assert tracker._machine_type is None

    def test_get_current_cost_before_start(self, tracker: CostTracker) -> None:
        """Test getting cost before starting returns None."""
        assert tracker.get_current_cost() is None

    def test_start_tracking(self, tracker: CostTracker) -> None:
        """Test starting cost tracking."""
        tracker.start(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
            is_spot=True,
        )
        assert tracker._start_time is not None
        assert tracker._machine_type == VertexMachineType.A2_HIGHGPU_1G
        assert tracker._is_spot is True

    def test_get_current_cost(self, tracker: CostTracker) -> None:
        """Test getting current cost."""
        tracker.start(
            machine_type=VertexMachineType.N1_STANDARD_8,
            is_spot=False,
        )

        time.sleep(0.1)  # Small delay

        cost = tracker.get_current_cost()
        assert cost is not None
        assert cost.duration_hours > 0
        assert cost.estimated_total_cost >= 0

    def test_get_current_cost_with_gpu(self, tracker: CostTracker) -> None:
        """Test cost calculation with GPU."""
        tracker.start(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
            is_spot=False,
        )

        cost = tracker.get_current_cost()
        assert cost is not None
        # A2 machine + A100 GPU should have accelerator cost
        # Note: A2 machines have integrated GPUs, so we may count both
        assert cost.machine_cost_per_hour > 0

    def test_get_current_cost_spot_discount(self, tracker: CostTracker) -> None:
        """Test spot instance discount."""
        # First get non-spot cost
        tracker.start(
            machine_type=VertexMachineType.N1_STANDARD_8,
            is_spot=False,
        )
        non_spot_cost = tracker.get_current_cost()

        # Reset and get spot cost
        tracker._start_time = None
        tracker.start(
            machine_type=VertexMachineType.N1_STANDARD_8,
            is_spot=True,
        )
        spot_cost = tracker.get_current_cost()

        assert spot_cost is not None
        assert non_spot_cost is not None
        assert spot_cost.total_cost_per_hour < non_spot_cost.total_cost_per_hour
        assert spot_cost.discount_applied > 0

    def test_stop_tracking(self, tracker: CostTracker) -> None:
        """Test stopping cost tracking."""
        tracker.start(
            machine_type=VertexMachineType.N1_STANDARD_8,
        )
        time.sleep(0.1)
        tracker.stop()

        assert tracker._end_time is not None

        # Cost should still be available after stop
        cost = tracker.get_current_cost()
        assert cost is not None

    def test_get_projected_cost(self, tracker: CostTracker) -> None:
        """Test projected cost calculation."""
        tracker.start(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            is_spot=False,
        )

        projected = tracker.get_projected_cost(24.0)
        assert projected is not None
        assert projected.duration_hours == 24.0
        assert projected.estimated_total_cost > 0

    def test_get_breakdown(self, tracker: CostTracker) -> None:
        """Test cost breakdown."""
        tracker.start(
            machine_type=VertexMachineType.N1_STANDARD_8,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=4,
        )

        time.sleep(0.1)
        breakdown = tracker.get_breakdown()

        assert breakdown.compute_cost >= 0
        assert breakdown.accelerator_cost >= 0
        assert breakdown.total_cost >= 0

    def test_replica_scaling(self, tracker: CostTracker) -> None:
        """Test cost scales with replica count."""
        tracker.start(
            machine_type=VertexMachineType.N1_STANDARD_8,
            replica_count=1,
        )
        single_cost = tracker.get_projected_cost(1.0)

        tracker._start_time = None
        tracker.start(
            machine_type=VertexMachineType.N1_STANDARD_8,
            replica_count=4,
        )
        quad_cost = tracker.get_projected_cost(1.0)

        assert single_cost is not None
        assert quad_cost is not None
        # 4 replicas should cost 4x
        assert abs(quad_cost.estimated_total_cost - 4 * single_cost.estimated_total_cost) < 0.01


class TestEstimateJobCost:
    """Tests for estimate_job_cost function."""

    def test_basic_estimate(self) -> None:
        """Test basic cost estimation."""
        estimate = estimate_job_cost(
            machine_type=VertexMachineType.N1_STANDARD_8,
            duration_hours=1.0,
        )
        assert estimate.duration_hours == 1.0
        assert estimate.estimated_total_cost > 0

    def test_gpu_estimate(self) -> None:
        """Test cost estimation with GPU."""
        estimate = estimate_job_cost(
            machine_type=VertexMachineType.N1_STANDARD_8,
            duration_hours=24.0,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=4,
        )
        assert estimate.accelerator_cost_per_hour > 0
        assert estimate.estimated_total_cost > 0

    def test_spot_estimate(self) -> None:
        """Test spot instance estimation."""
        non_spot = estimate_job_cost(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            duration_hours=24.0,
            is_spot=False,
        )
        spot = estimate_job_cost(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            duration_hours=24.0,
            is_spot=True,
        )

        assert spot.estimated_total_cost < non_spot.estimated_total_cost
        expected_ratio = SPOT_DISCOUNT_FACTOR
        actual_ratio = spot.estimated_total_cost / non_spot.estimated_total_cost
        assert abs(actual_ratio - expected_ratio) < 0.01


class TestGetHourlyRate:
    """Tests for get_hourly_rate function."""

    def test_machine_only(self) -> None:
        """Test rate for machine only."""
        rate = get_hourly_rate(VertexMachineType.N1_STANDARD_8)
        expected = MACHINE_HOURLY_RATES[VertexMachineType.N1_STANDARD_8]
        assert rate == expected

    def test_with_accelerator(self) -> None:
        """Test rate with accelerator."""
        rate = get_hourly_rate(
            machine_type=VertexMachineType.N1_STANDARD_8,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=2,
        )
        expected_machine = MACHINE_HOURLY_RATES[VertexMachineType.N1_STANDARD_8]
        expected_gpu = ACCELERATOR_HOURLY_RATES[AcceleratorType.NVIDIA_TESLA_T4] * 2
        assert abs(rate - (expected_machine + expected_gpu)) < 0.01

    def test_spot_rate(self) -> None:
        """Test spot instance rate."""
        regular = get_hourly_rate(VertexMachineType.N1_STANDARD_8, is_spot=False)
        spot = get_hourly_rate(VertexMachineType.N1_STANDARD_8, is_spot=True)
        assert spot < regular
        assert abs(spot / regular - SPOT_DISCOUNT_FACTOR) < 0.01


class TestFormatCostTable:
    """Tests for format_cost_table function."""

    def test_basic_table(self) -> None:
        """Test basic table formatting."""
        table = format_cost_table(
            machine_types=[
                VertexMachineType.N1_STANDARD_8,
                VertexMachineType.A2_HIGHGPU_1G,
            ]
        )
        assert "n1-standard-8" in table
        assert "a2-highgpu-1g" in table
        assert "Hourly Rate" in table

    def test_spot_table(self) -> None:
        """Test table with spot pricing."""
        regular_table = format_cost_table(
            machine_types=[VertexMachineType.N1_STANDARD_8],
            is_spot=False,
        )
        spot_table = format_cost_table(
            machine_types=[VertexMachineType.N1_STANDARD_8],
            is_spot=True,
        )
        # Spot prices should be lower
        # Extract price from table (simplified check)
        assert spot_table != regular_table


class TestPricingData:
    """Tests for pricing data completeness."""

    def test_all_machine_types_have_rates(self) -> None:
        """Verify all machine types have pricing."""
        for machine_type in VertexMachineType:
            assert machine_type in MACHINE_HOURLY_RATES, f"Missing rate for {machine_type}"
            assert MACHINE_HOURLY_RATES[machine_type] >= 0

    def test_all_accelerator_types_have_rates(self) -> None:
        """Verify all accelerator types have pricing."""
        for accel_type in AcceleratorType:
            assert accel_type in ACCELERATOR_HOURLY_RATES, f"Missing rate for {accel_type}"
            assert ACCELERATOR_HOURLY_RATES[accel_type] >= 0
