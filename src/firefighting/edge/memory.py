"""Memory budget manager for edge deployment.

Tracks memory usage and enforces limits to prevent OOM
on memory-constrained edge devices (Jetson: 4GB max).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from src.firefighting.config.edge import EdgeConfig

logger = structlog.get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    model_mb: float
    working_mb: float
    total_mb: float
    budget_mb: float

    @property
    def utilization(self) -> float:
        """Memory utilization as fraction of budget."""
        return self.total_mb / max(self.budget_mb, 1.0)

    @property
    def headroom_mb(self) -> float:
        """Remaining memory headroom in MB."""
        return max(0.0, self.budget_mb - self.total_mb)


class MemoryBudgetManager:
    """Monitors and enforces memory constraints.

    Tracks allocations across model weights, working memory
    (mesh + state + MCTS tree), and overhead.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self._allocations: dict[str, float] = {}

    def register_allocation(self, name: str, size_bytes: int) -> bool:
        """Register a memory allocation.

        Args:
            name: Allocation identifier.
            size_bytes: Size in bytes.

        Returns:
            True if allocation fits within budget, False otherwise.

        """
        size_mb = size_bytes / (1024 * 1024)
        current_total = sum(self._allocations.values())

        if current_total + size_mb > self.config.max_memory_mb:
            logger.warning(
                "memory_budget_exceeded",
                allocation=name,
                requested_mb=size_mb,
                current_mb=current_total,
                budget_mb=self.config.max_memory_mb,
            )
            return False

        self._allocations[name] = size_mb
        return True

    def release_allocation(self, name: str) -> None:
        """Release a previously registered allocation."""
        self._allocations.pop(name, None)

    def estimate_array_mb(self, shape: tuple[int, ...], dtype: type = np.float32) -> float:
        """Estimate memory for a numpy array in MB."""
        n_elements = 1
        for s in shape:
            n_elements *= s
        bytes_per_element = np.dtype(dtype).itemsize
        return n_elements * bytes_per_element / (1024 * 1024)

    def snapshot(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        model_mb = self._allocations.get("model", 0.0)
        working_mb = sum(v for k, v in self._allocations.items() if k != "model")
        return MemorySnapshot(
            model_mb=model_mb,
            working_mb=working_mb,
            total_mb=model_mb + working_mb,
            budget_mb=float(self.config.max_memory_mb),
        )

    def check_budget(self) -> bool:
        """Check if current allocations are within budget."""
        snap = self.snapshot()
        return snap.total_mb <= self.config.max_memory_mb
