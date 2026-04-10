"""Tests for coordinate frames and quaternion operations."""

from __future__ import annotations

import math

import torch

from src.intercept.frames import FrameTransform, QuaternionOps


class TestQuaternionOps:
    def test_identity(self) -> None:
        q = QuaternionOps.identity()
        assert q.shape == (4,)
        assert torch.allclose(q, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))

    def test_identity_batched(self) -> None:
        q = QuaternionOps.identity(batch_shape=(3, 2))
        assert q.shape == (3, 2, 4)
        assert torch.allclose(q[..., 0], torch.ones(3, 2, dtype=torch.float64))

    def test_normalize(self) -> None:
        q = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        qn = QuaternionOps.normalize(q)
        assert torch.allclose(torch.norm(qn), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

    def test_conjugate(self) -> None:
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        qc = QuaternionOps.conjugate(q)
        assert qc[0] == q[0]
        assert torch.allclose(qc[1:], -q[1:])

    def test_multiply_identity(self) -> None:
        q = QuaternionOps.normalize(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        identity = QuaternionOps.identity(dtype=torch.float64)
        result = QuaternionOps.multiply(q, identity)
        assert torch.allclose(result, q, atol=1e-10)

    def test_multiply_inverse_gives_identity(self) -> None:
        q = QuaternionOps.normalize(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        q_inv = QuaternionOps.conjugate(q)
        result = QuaternionOps.multiply(q, q_inv)
        identity = QuaternionOps.identity(dtype=torch.float64)
        assert torch.allclose(result, identity, atol=1e-10)

    def test_rotate_vector_identity(self) -> None:
        q = QuaternionOps.identity(dtype=torch.float64)
        v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = QuaternionOps.rotate_vector(q, v)
        assert torch.allclose(result, v, atol=1e-10)

    def test_rotate_90_degrees_z(self) -> None:
        """Rotate [1, 0, 0] by 90 degrees around z-axis -> [0, 1, 0]."""
        angle = torch.tensor(math.pi / 2, dtype=torch.float64)
        q = QuaternionOps.from_euler(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            angle,
        )
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        result = QuaternionOps.rotate_vector(q, v)
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=1e-10)

    def test_euler_roundtrip(self) -> None:
        roll = torch.tensor(0.3, dtype=torch.float64)
        pitch = torch.tensor(-0.2, dtype=torch.float64)
        yaw = torch.tensor(1.5, dtype=torch.float64)
        q = QuaternionOps.from_euler(roll, pitch, yaw)
        r2, p2, y2 = QuaternionOps.to_euler(q)
        assert torch.allclose(roll, r2, atol=1e-10)
        assert torch.allclose(pitch, p2, atol=1e-10)
        assert torch.allclose(yaw, y2, atol=1e-10)

    def test_rotation_matrix_orthogonal(self) -> None:
        q = QuaternionOps.normalize(torch.tensor([1.0, 0.5, -0.3, 0.7], dtype=torch.float64))
        R = QuaternionOps.to_rotation_matrix(q)
        # R * R^T should be identity
        product = R @ R.transpose(-2, -1)
        assert torch.allclose(product, torch.eye(3, dtype=torch.float64), atol=1e-10)

    def test_rotation_matrix_det_one(self) -> None:
        q = QuaternionOps.normalize(torch.tensor([1.0, 0.5, -0.3, 0.7], dtype=torch.float64))
        R = QuaternionOps.to_rotation_matrix(q)
        assert torch.allclose(torch.det(R), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

    def test_slerp_endpoints(self) -> None:
        q0 = QuaternionOps.identity(dtype=torch.float64)
        q1 = QuaternionOps.from_euler(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(math.pi / 2, dtype=torch.float64),
        )
        assert torch.allclose(QuaternionOps.slerp(q0, q1, 0.0), q0, atol=1e-10)
        assert torch.allclose(QuaternionOps.slerp(q0, q1, 1.0), q1, atol=1e-10)

    def test_batched_operations(self) -> None:
        batch = 5
        q = QuaternionOps.normalize(torch.randn(batch, 4, dtype=torch.float64))
        v = torch.randn(batch, 3, dtype=torch.float64)
        result = QuaternionOps.rotate_vector(q, v)
        assert result.shape == (batch, 3)
        # Rotation preserves vector magnitude
        for i in range(batch):
            assert torch.allclose(torch.norm(result[i]), torch.norm(v[i]), atol=1e-10)


class TestFrameTransform:
    def test_ned_enu_roundtrip(self) -> None:
        v_ned = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        v_enu = FrameTransform.ned_to_enu(v_ned)
        v_ned2 = FrameTransform.enu_to_ned(v_enu)
        assert torch.allclose(v_ned, v_ned2, atol=1e-10)

    def test_ned_to_enu_values(self) -> None:
        # NED(N=1, E=2, D=3) -> ENU(E=2, N=1, U=-3)
        v_ned = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        v_enu = FrameTransform.ned_to_enu(v_ned)
        expected = torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64)
        assert torch.allclose(v_enu, expected, atol=1e-10)

    def test_body_ned_roundtrip(self) -> None:
        q = QuaternionOps.normalize(torch.tensor([1.0, 0.5, -0.3, 0.7], dtype=torch.float64))
        v_body = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        v_ned = FrameTransform.body_to_ned(v_body, q)
        v_body2 = FrameTransform.ned_to_body(v_ned, q)
        assert torch.allclose(v_body, v_body2, atol=1e-10)

    def test_geodetic_ecef_roundtrip(self) -> None:
        lat = torch.tensor(math.radians(45.0), dtype=torch.float64)
        lon = torch.tensor(math.radians(-122.0), dtype=torch.float64)
        alt = torch.tensor(1000.0, dtype=torch.float64)

        ecef = FrameTransform.geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = FrameTransform.ecef_to_geodetic(ecef)

        assert torch.allclose(lat, lat2, atol=1e-8)
        assert torch.allclose(lon, lon2, atol=1e-8)
        assert torch.allclose(alt, alt2, atol=0.01)  # mm precision

    def test_relative_ned(self) -> None:
        own = torch.tensor([100.0, 200.0, -500.0], dtype=torch.float64)
        target = torch.tensor([1100.0, 200.0, -500.0], dtype=torch.float64)
        rel = FrameTransform.relative_ned(own, target)
        assert torch.allclose(rel, torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float64))

    def test_range_and_bearing(self) -> None:
        # Target 1000m due North, same altitude
        rel = torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float64)
        rng, az, el = FrameTransform.range_and_bearing(rel)
        assert torch.allclose(rng, torch.tensor(1000.0, dtype=torch.float64), atol=0.1)
        assert torch.allclose(az, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(el, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)

    def test_range_and_bearing_east(self) -> None:
        # Target 1000m due East
        rel = torch.tensor([0.0, 1000.0, 0.0], dtype=torch.float64)
        rng, az, el = FrameTransform.range_and_bearing(rel)
        assert torch.allclose(rng, torch.tensor(1000.0, dtype=torch.float64), atol=0.1)
        assert torch.allclose(az, torch.tensor(math.pi / 2, dtype=torch.float64), atol=1e-10)

    def test_range_and_bearing_above(self) -> None:
        # Target 1000m directly above (NED down = -1000)
        rel = torch.tensor([0.0, 0.0, -1000.0], dtype=torch.float64)
        rng, az, el = FrameTransform.range_and_bearing(rel)
        assert torch.allclose(rng, torch.tensor(1000.0, dtype=torch.float64), atol=0.1)
        assert torch.allclose(el, torch.tensor(math.pi / 2, dtype=torch.float64), atol=1e-10)
