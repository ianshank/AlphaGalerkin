"""Cost tracking and estimation for Vertex AI training.

This module provides utilities for tracking and estimating training
costs on Google Cloud Vertex AI, helping with budget management and
cost optimization.

Pricing is approximate and based on public GCP pricing. For accurate
costs, use the GCP Billing API or Cost Management console.

Example:
    from src.vertex.cost import CostTracker, estimate_job_cost

    # Estimate job cost
    estimate = estimate_job_cost(
        machine_type=VertexMachineType.A2_HIGHGPU_1G,
        duration_hours=24,
        is_spot=True,
    )
    print(f"Estimated cost: ${estimate.estimated_total_cost:.2f}")

    # Track costs during training
    tracker = CostTracker()
    tracker.start(machine_type=..., accelerator_type=...)

    # Get current cost
    current = tracker.get_current_cost()
    print(f"Current cost: ${current.estimated_total_cost:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from src.vertex.config import AcceleratorType, VertexMachineType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Spot instance discount (approximate)
SPOT_DISCOUNT_FACTOR = 0.3  # 70% discount

# Approximate hourly rates in USD (as of 2025)
# These are estimates and should be updated based on actual GCP pricing
# See: https://cloud.google.com/compute/vm-instance-pricing
MACHINE_HOURLY_RATES: dict[VertexMachineType, float] = {
    # Standard machines
    VertexMachineType.N1_STANDARD_4: 0.19,
    VertexMachineType.N1_STANDARD_8: 0.38,
    VertexMachineType.N1_STANDARD_16: 0.76,
    VertexMachineType.N1_STANDARD_32: 1.52,
    VertexMachineType.N1_STANDARD_64: 3.04,
    VertexMachineType.N1_STANDARD_96: 4.56,

    # High-memory machines
    VertexMachineType.N1_HIGHMEM_2: 0.12,
    VertexMachineType.N1_HIGHMEM_4: 0.24,
    VertexMachineType.N1_HIGHMEM_8: 0.47,
    VertexMachineType.N1_HIGHMEM_16: 0.95,
    VertexMachineType.N1_HIGHMEM_32: 1.90,
    VertexMachineType.N1_HIGHMEM_64: 3.80,
    VertexMachineType.N1_HIGHMEM_96: 5.69,

    # A2 machines (includes A100 GPU)
    VertexMachineType.A2_HIGHGPU_1G: 3.67,
    VertexMachineType.A2_HIGHGPU_2G: 7.35,
    VertexMachineType.A2_HIGHGPU_4G: 14.69,
    VertexMachineType.A2_HIGHGPU_8G: 29.39,
    VertexMachineType.A2_MEGAGPU_16G: 55.74,
    VertexMachineType.A2_ULTRAGPU_1G: 5.00,
    VertexMachineType.A2_ULTRAGPU_2G: 10.00,
    VertexMachineType.A2_ULTRAGPU_4G: 20.00,
    VertexMachineType.A2_ULTRAGPU_8G: 40.00,

    # A3 machines (H100)
    VertexMachineType.A3_HIGHGPU_8G: 101.22,

    # G2 machines (L4)
    VertexMachineType.G2_STANDARD_4: 0.84,
    VertexMachineType.G2_STANDARD_8: 1.17,
    VertexMachineType.G2_STANDARD_12: 1.50,
    VertexMachineType.G2_STANDARD_16: 1.84,
    VertexMachineType.G2_STANDARD_24: 2.67,
    VertexMachineType.G2_STANDARD_32: 3.17,
    VertexMachineType.G2_STANDARD_48: 4.51,
    VertexMachineType.G2_STANDARD_96: 8.01,
}

# Accelerator hourly rates (per GPU)
ACCELERATOR_HOURLY_RATES: dict[AcceleratorType, float] = {
    AcceleratorType.NVIDIA_TESLA_K80: 0.45,
    AcceleratorType.NVIDIA_TESLA_P100: 1.46,
    AcceleratorType.NVIDIA_TESLA_V100: 2.48,
    AcceleratorType.NVIDIA_TESLA_P4: 0.60,
    AcceleratorType.NVIDIA_TESLA_T4: 0.35,
    AcceleratorType.NVIDIA_TESLA_A100: 2.93,
    AcceleratorType.NVIDIA_A100_80GB: 3.67,
    AcceleratorType.NVIDIA_H100_80GB: 8.00,
    AcceleratorType.NVIDIA_L4: 0.81,
    AcceleratorType.TPU_V2: 4.50,
    AcceleratorType.TPU_V3: 8.00,
    AcceleratorType.TPU_V4_POD: 12.88,
}


@dataclass
class CostEstimate:
    """Cost estimate for a training job.

    Attributes:
        machine_cost_per_hour: Hourly cost for VM.
        accelerator_cost_per_hour: Hourly cost for accelerators.
        total_cost_per_hour: Combined hourly cost.
        estimated_total_cost: Total estimated cost.
        duration_hours: Duration in hours.
        is_spot: Whether using spot instances.
        discount_applied: Discount percentage applied.
    """

    machine_cost_per_hour: float
    accelerator_cost_per_hour: float
    total_cost_per_hour: float
    estimated_total_cost: float
    duration_hours: float
    is_spot: bool = False
    discount_applied: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "machine_cost_per_hour": round(self.machine_cost_per_hour, 4),
            "accelerator_cost_per_hour": round(self.accelerator_cost_per_hour, 4),
            "total_cost_per_hour": round(self.total_cost_per_hour, 4),
            "estimated_total_cost": round(self.estimated_total_cost, 2),
            "duration_hours": round(self.duration_hours, 2),
            "is_spot": self.is_spot,
            "discount_applied": round(self.discount_applied * 100, 1),
        }

    def format_summary(self) -> str:
        """Format as human-readable summary."""
        spot_str = " (spot)" if self.is_spot else ""
        return (
            f"Cost Estimate{spot_str}:\n"
            f"  Machine: ${self.machine_cost_per_hour:.2f}/hr\n"
            f"  Accelerators: ${self.accelerator_cost_per_hour:.2f}/hr\n"
            f"  Total: ${self.total_cost_per_hour:.2f}/hr\n"
            f"  Duration: {self.duration_hours:.1f} hours\n"
            f"  Estimated Total: ${self.estimated_total_cost:.2f}"
        )


@dataclass
class CostBreakdown:
    """Detailed cost breakdown by category.

    Attributes:
        compute_cost: VM compute cost.
        accelerator_cost: GPU/TPU cost.
        network_cost: Network egress cost (estimated).
        storage_cost: GCS storage cost (estimated).
        total_cost: Total cost.
    """

    compute_cost: float = 0.0
    accelerator_cost: float = 0.0
    network_cost: float = 0.0
    storage_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Calculate total cost."""
        return (
            self.compute_cost +
            self.accelerator_cost +
            self.network_cost +
            self.storage_cost
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "compute_cost": round(self.compute_cost, 2),
            "accelerator_cost": round(self.accelerator_cost, 2),
            "network_cost": round(self.network_cost, 2),
            "storage_cost": round(self.storage_cost, 2),
            "total_cost": round(self.total_cost, 2),
        }


class CostTracker:
    """Track and estimate Vertex AI training costs.

    This tracker monitors training duration and calculates estimated
    costs based on resource configuration.

    Example:
        tracker = CostTracker()
        tracker.start(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
            is_spot=True,
        )

        # During training...
        cost = tracker.get_current_cost()
        print(f"Current cost: ${cost.estimated_total_cost:.2f}")

        # After training
        tracker.stop()
        final = tracker.get_current_cost()
    """

    def __init__(self) -> None:
        """Initialize cost tracker."""
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._machine_type: VertexMachineType | None = None
        self._accelerator_type: AcceleratorType | None = None
        self._accelerator_count: int = 0
        self._is_spot: bool = False
        self._replica_count: int = 1

    def start(
        self,
        machine_type: VertexMachineType,
        accelerator_type: AcceleratorType | None = None,
        accelerator_count: int = 0,
        replica_count: int = 1,
        is_spot: bool = False,
    ) -> None:
        """Start tracking costs.

        Args:
            machine_type: VM machine type.
            accelerator_type: GPU/TPU type.
            accelerator_count: Number of accelerators per replica.
            replica_count: Number of training replicas.
            is_spot: Whether using spot instances.
        """
        self._start_time = datetime.now()
        self._end_time = None
        self._machine_type = machine_type
        self._accelerator_type = accelerator_type
        self._accelerator_count = accelerator_count
        self._replica_count = replica_count
        self._is_spot = is_spot

        logger.info(
            "cost_tracking_started",
            machine_type=machine_type.value,
            accelerator_type=accelerator_type.value if accelerator_type else None,
            accelerator_count=accelerator_count,
            is_spot=is_spot,
        )

    def stop(self) -> None:
        """Stop tracking and record end time."""
        self._end_time = datetime.now()
        logger.info(
            "cost_tracking_stopped",
            duration_hours=self._get_duration_hours(),
        )

    def get_current_cost(self) -> CostEstimate | None:
        """Get current estimated cost.

        Returns:
            CostEstimate or None if tracking not started.
        """
        if self._start_time is None or self._machine_type is None:
            return None

        duration = self._get_duration_hours()
        return self._calculate_cost(duration)

    def get_projected_cost(
        self,
        target_duration_hours: float,
    ) -> CostEstimate | None:
        """Get projected cost for a target duration.

        Args:
            target_duration_hours: Target duration in hours.

        Returns:
            Projected CostEstimate.
        """
        if self._machine_type is None:
            return None

        return self._calculate_cost(target_duration_hours)

    def get_breakdown(self) -> CostBreakdown:
        """Get detailed cost breakdown.

        Returns:
            CostBreakdown with component costs.
        """
        estimate = self.get_current_cost()
        if estimate is None:
            return CostBreakdown()

        return CostBreakdown(
            compute_cost=estimate.machine_cost_per_hour * estimate.duration_hours,
            accelerator_cost=estimate.accelerator_cost_per_hour * estimate.duration_hours,
        )

    def _get_duration_hours(self) -> float:
        """Calculate duration in hours."""
        if self._start_time is None:
            return 0.0

        end = self._end_time or datetime.now()
        delta = end - self._start_time
        return delta.total_seconds() / 3600.0

    def _calculate_cost(self, duration_hours: float) -> CostEstimate:
        """Calculate cost estimate.

        Args:
            duration_hours: Duration in hours.

        Returns:
            CostEstimate.
        """
        machine_type = self._machine_type or VertexMachineType.N1_STANDARD_8
        machine_rate = MACHINE_HOURLY_RATES.get(machine_type, 0.0)

        # Scale by replica count
        machine_cost = machine_rate * self._replica_count

        # Accelerator cost
        accel_cost = 0.0
        if self._accelerator_type is not None and self._accelerator_count > 0:
            accel_rate = ACCELERATOR_HOURLY_RATES.get(self._accelerator_type, 0.0)
            accel_cost = accel_rate * self._accelerator_count * self._replica_count

        # Apply spot discount
        total_hourly = machine_cost + accel_cost
        discount = 0.0
        if self._is_spot:
            discount = 1.0 - SPOT_DISCOUNT_FACTOR
            total_hourly *= SPOT_DISCOUNT_FACTOR

        total_cost = total_hourly * duration_hours

        return CostEstimate(
            machine_cost_per_hour=machine_cost,
            accelerator_cost_per_hour=accel_cost,
            total_cost_per_hour=total_hourly,
            estimated_total_cost=total_cost,
            duration_hours=duration_hours,
            is_spot=self._is_spot,
            discount_applied=discount,
        )


def estimate_job_cost(
    machine_type: VertexMachineType,
    duration_hours: float,
    accelerator_type: AcceleratorType | None = None,
    accelerator_count: int = 0,
    replica_count: int = 1,
    is_spot: bool = False,
) -> CostEstimate:
    """Estimate cost for a training job.

    This is a convenience function for quick cost estimation without
    starting a tracker.

    Args:
        machine_type: VM machine type.
        duration_hours: Expected duration in hours.
        accelerator_type: GPU/TPU type.
        accelerator_count: Number of accelerators.
        replica_count: Number of replicas.
        is_spot: Whether using spot instances.

    Returns:
        CostEstimate for the job.

    Example:
        estimate = estimate_job_cost(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            duration_hours=24,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
            is_spot=True,
        )
        print(f"Estimated cost: ${estimate.estimated_total_cost:.2f}")
    """
    tracker = CostTracker()
    tracker._machine_type = machine_type
    tracker._accelerator_type = accelerator_type
    tracker._accelerator_count = accelerator_count
    tracker._replica_count = replica_count
    tracker._is_spot = is_spot

    return tracker._calculate_cost(duration_hours)


def get_hourly_rate(
    machine_type: VertexMachineType,
    accelerator_type: AcceleratorType | None = None,
    accelerator_count: int = 0,
    is_spot: bool = False,
) -> float:
    """Get hourly rate for a resource configuration.

    Args:
        machine_type: VM machine type.
        accelerator_type: GPU/TPU type.
        accelerator_count: Number of accelerators.
        is_spot: Whether using spot instances.

    Returns:
        Hourly rate in USD.
    """
    machine_rate = MACHINE_HOURLY_RATES.get(machine_type, 0.0)

    accel_rate = 0.0
    if accelerator_type is not None and accelerator_count > 0:
        accel_rate = ACCELERATOR_HOURLY_RATES.get(accelerator_type, 0.0) * accelerator_count

    total = machine_rate + accel_rate

    if is_spot:
        total *= SPOT_DISCOUNT_FACTOR

    return total


def format_cost_table(
    machine_types: list[VertexMachineType] | None = None,
    is_spot: bool = False,
) -> str:
    """Format a cost comparison table.

    Args:
        machine_types: Machine types to include (None for all).
        is_spot: Show spot pricing.

    Returns:
        Formatted table string.
    """
    if machine_types is None:
        machine_types = list(MACHINE_HOURLY_RATES.keys())

    lines = [
        "Machine Type                    Hourly Rate",
        "-" * 50,
    ]

    for mt in machine_types:
        rate = MACHINE_HOURLY_RATES.get(mt, 0.0)
        if is_spot:
            rate *= SPOT_DISCOUNT_FACTOR
        lines.append(f"{mt.value:<30} ${rate:>8.2f}")

    return "\n".join(lines)
