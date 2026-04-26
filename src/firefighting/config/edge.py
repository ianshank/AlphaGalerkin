"""Edge deployment configuration for drone compute.

Defines memory, latency, and power constraints for
Jetson Orin Nano and Raspberry Pi 5 targets.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from src.templates.config import BaseModuleConfig


class EdgeDevice(str, Enum):
    """Supported edge compute targets."""

    JETSON_ORIN_NANO = "jetson_orin_nano"
    RASPBERRY_PI_5 = "raspberry_pi_5"
    WORKSTATION = "workstation"


class EdgeConfig(BaseModuleConfig):
    """Edge deployment constraints."""

    device: EdgeDevice = Field(
        default=EdgeDevice.JETSON_ORIN_NANO,
        description="Target edge device.",
    )
    max_memory_mb: int = Field(
        default=4096,
        ge=512,
        description="Maximum peak memory in MB.",
    )
    model_memory_mb: int = Field(
        default=500,
        ge=10,
        description="Maximum model weights memory in MB.",
    )
    max_latency_ms: float = Field(
        default=500.0,
        gt=0.0,
        description="Maximum per-cycle inference latency in ms.",
    )
    sensor_ingest_budget_ms: float = Field(
        default=50.0,
        gt=0.0,
        description="Latency budget for sensor ingest in ms.",
    )
    mcts_budget_ms: float = Field(
        default=300.0,
        gt=0.0,
        description="Latency budget for MCTS search in ms.",
    )
    pde_solve_budget_ms: float = Field(
        default=100.0,
        gt=0.0,
        description="Latency budget for PDE solve in ms.",
    )
    output_budget_ms: float = Field(
        default=50.0,
        gt=0.0,
        description="Latency budget for output encoding in ms.",
    )
    use_onnx: bool = Field(
        default=True,
        description="Use ONNX Runtime for neural inference.",
    )
    quantization: str = Field(
        default="int8",
        description="Quantization mode (none, dynamic, int8, fp16).",
    )
    power_budget_w: float = Field(
        default=15.0,
        gt=0.0,
        description="Power budget in watts.",
    )
