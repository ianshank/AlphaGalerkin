"""Tests for intercept training losses."""

from __future__ import annotations

import pytest
import torch

from src.intercept.losses import (
    ConstraintViolationLoss,
    DynamicsResidualLoss,
    EngagementLoss,
)


class TestDynamicsResidualLoss:
    def test_zero_residual_for_consistent_trajectory(self) -> None:
        """If positions/velocities are consistent with forces, loss should be ~0."""
        dt = 0.01
        T = 10
        mass = 10.0
        force = torch.tensor([100.0, 0.0, 0.0])  # constant force
        accel = force / mass  # 10 m/s^2

        # Build consistent trajectory
        velocities = torch.zeros(T, 3)
        positions = torch.zeros(T, 3)
        for t in range(1, T):
            velocities[t] = velocities[t - 1] + accel * dt
            positions[t] = positions[t - 1] + velocities[t - 1] * dt

        forces = force.unsqueeze(0).expand(T - 1, 3)
        masses = torch.full((T - 1,), mass)

        loss_fn = DynamicsResidualLoss()
        loss = loss_fn(positions, velocities, forces, masses, dt)
        assert loss.item() < 1e-6

    def test_nonzero_for_inconsistent(self) -> None:
        """Random trajectory + forces should give nonzero loss."""
        loss_fn = DynamicsResidualLoss()
        pos = torch.randn(10, 3)
        vel = torch.randn(10, 3)
        forces = torch.randn(9, 3)
        masses = torch.ones(9) * 10.0
        loss = loss_fn(pos, vel, forces, masses, dt=0.01)
        assert loss.item() > 0.0

    def test_quaternion_norm_penalty(self) -> None:
        """Quaternions with norm != 1 should incur penalty."""
        loss_fn = DynamicsResidualLoss(quat_weight=10.0)
        pos = torch.zeros(5, 3)
        vel = torch.zeros(5, 3)
        forces = torch.zeros(4, 3)
        masses = torch.ones(4)

        # Perfect quaternions
        good_q = torch.zeros(5, 4)
        good_q[:, 0] = 1.0
        loss_good = loss_fn(pos, vel, forces, masses, 0.01, good_q)

        # Bad quaternions (norm = 2)
        bad_q = torch.zeros(5, 4)
        bad_q[:, 0] = 2.0
        loss_bad = loss_fn(pos, vel, forces, masses, 0.01, bad_q)

        assert loss_bad.item() > loss_good.item()


class TestConstraintViolationLoss:
    def test_no_violation(self) -> None:
        """Accelerations within limits should give zero loss."""
        loss_fn = ConstraintViolationLoss(max_g=30.0)
        accel = torch.tensor([[10.0, 0.0, 0.0], [0.0, 50.0, 0.0]])  # < 30g
        loss = loss_fn(accel)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_violation_above_max_g(self) -> None:
        """Accelerations exceeding max-g should incur penalty."""
        loss_fn = ConstraintViolationLoss(max_g=5.0)
        # 100 m/s^2 = ~10g, exceeds 5g limit
        accel = torch.tensor([[100.0, 0.0, 0.0]])
        loss = loss_fn(accel)
        assert loss.item() > 0.0


class TestEngagementLoss:
    def test_perfect_predictions(self) -> None:
        """Matching predictions should give low loss."""
        loss_fn = EngagementLoss()
        B, A = 4, 10

        target_policy = torch.softmax(torch.randn(B, A), dim=-1)
        policy_logits = torch.log(target_policy + 1e-8)

        target_value = torch.tensor([1.0, -1.0, 0.0, 1.0])
        value_pred = target_value.clone()

        loss, metrics = loss_fn(policy_logits, value_pred, target_policy, target_value)
        assert metrics["value_loss"] < 1e-6
        assert loss.item() < 2.0  # policy CE won't be exactly 0 due to log numerics

    def test_bad_predictions(self) -> None:
        """Wrong predictions should give high loss."""
        loss_fn = EngagementLoss()
        B, A = 4, 10

        target_policy = torch.softmax(torch.randn(B, A), dim=-1)
        policy_logits = torch.randn(B, A)  # random, won't match

        target_value = torch.ones(B)
        value_pred = -torch.ones(B)  # opposite

        loss, metrics = loss_fn(policy_logits, value_pred, target_policy, target_value)
        assert metrics["value_loss"] > 1.0

    def test_returns_metrics(self) -> None:
        loss_fn = EngagementLoss()
        B, A = 2, 5
        loss, metrics = loss_fn(
            torch.randn(B, A),
            torch.randn(B),
            torch.softmax(torch.randn(B, A), dim=-1),
            torch.randn(B),
        )
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "total_loss" in metrics

    def test_gradient_flows(self) -> None:
        """Loss should be differentiable."""
        loss_fn = EngagementLoss()
        B, A = 2, 5
        logits = torch.randn(B, A, requires_grad=True)
        values = torch.randn(B, requires_grad=True)

        loss, _ = loss_fn(
            logits,
            values,
            torch.softmax(torch.randn(B, A), dim=-1),
            torch.randn(B),
        )
        loss.backward()
        assert logits.grad is not None
        assert values.grad is not None
