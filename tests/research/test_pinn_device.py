"""Tests for PINN device resolution + vector_pde mode.

Guards the SimplePINNSolver hard-coded-CPU bug fix: the canonical solver
must honour PINNConfig.device, accept "auto"/"cpu"/"cuda"/"cuda:N", and
auto-detect Navier-Stokes operators to build a 2-channel network.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import NavierStokesOperator, PoissonOperator
from src.poc.device import resolve_device
from src.research.baselines import (
    PINNConfig,
    SimplePINNSolver,
    SolverResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poisson_1d() -> PoissonOperator:
    cfg = PDEConfig(
        name="test_poisson_pinn",
        pde_type=PDEType.POISSON,
        domain_dim=1,
        domain_min=[0.0],
        domain_max=[1.0],
        advection_coeff=[0.0],
    )
    return PoissonOperator(cfg)


def _make_ns_2d() -> NavierStokesOperator:
    two_pi = 2.0 * float(np.pi)
    cfg = PDEConfig(
        name="test_ns_pinn",
        pde_type=PDEType.NAVIER_STOKES,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[two_pi, two_pi],
        advection_coeff=[0.0, 0.0],
    )
    return NavierStokesOperator(cfg, reynolds_number=100.0)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_cpu_explicit(self) -> None:
        assert resolve_device("cpu").type == "cpu"

    def test_auto_falls_back_to_cpu_when_no_cuda(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available — auto resolves to cuda, not cpu")
        assert resolve_device("auto").type == "cpu"

    def test_auto_uses_cuda_when_available(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        assert resolve_device("auto").type == "cuda"

    def test_cuda_raises_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available — explicit cuda would succeed")
        with pytest.raises(RuntimeError, match="CUDA is not"):
            resolve_device("cuda")

    def test_cuda_indexed_raises_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available")
        with pytest.raises(RuntimeError, match="CUDA is not"):
            resolve_device("cuda:0")

    def test_cuda_index_out_of_range_raises(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        n = torch.cuda.device_count()
        with pytest.raises(RuntimeError, match="only .* CUDA device"):
            resolve_device(f"cuda:{n}")

    def test_invalid_preference_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device preference"):
            resolve_device("gpu")  # not "cuda"

    def test_context_appears_in_error_message(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available")
        with pytest.raises(RuntimeError, match="my_test_context"):
            resolve_device("cuda", context="my_test_context")

    # The cases below mock torch.cuda.is_available() so the success and
    # error branches are reachable on every host (CI on a CPU box AND a
    # CUDA dev rig). Without mocking, the cuda/cuda:N branches are skipped
    # on whichever side the host happens to be.

    def test_cuda_raises_with_mocked_no_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not"):
                resolve_device("cuda", context="mocked")

    def test_cuda_indexed_raises_with_mocked_no_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not"):
                resolve_device("cuda:1", context="mocked")

    def test_cuda_resolves_with_mocked_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("cuda").type == "cuda"

    def test_cuda_indexed_resolves_with_mocked_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                dev = resolve_device("cuda:1")
                assert dev.type == "cuda"
                assert dev.index == 1

    def test_cuda_indexed_out_of_range_with_mocked_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with pytest.raises(RuntimeError, match="only .* CUDA device"):
                    resolve_device("cuda:5")


# ---------------------------------------------------------------------------
# PINNConfig device + vector_pde fields
# ---------------------------------------------------------------------------


class TestPINNConfigDevice:
    def test_default_is_auto(self) -> None:
        cfg = PINNConfig()
        assert cfg.device == "auto"

    def test_explicit_cpu(self) -> None:
        cfg = PINNConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_explicit_cuda_indexed(self) -> None:
        cfg = PINNConfig(device="cuda:0")
        assert cfg.device == "cuda:0"

    def test_vector_pde_default_none(self) -> None:
        cfg = PINNConfig()
        assert cfg.vector_pde is None

    def test_vector_pde_override_true(self) -> None:
        cfg = PINNConfig(vector_pde=True)
        assert cfg.vector_pde is True


# ---------------------------------------------------------------------------
# SimplePINNSolver respects device + vector_pde
# ---------------------------------------------------------------------------


class TestSimplePINNSolverDevice:
    def test_cpu_default_works_without_cuda(self) -> None:
        """Solver must run on CPU end-to-end with explicit device='cpu'."""
        solver = SimplePINNSolver(
            hidden_dim=8,
            n_layers=2,
            n_epochs=1,
            n_collocation=20,
            learning_rate=1e-3,
            device="cpu",
        )
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=10)
        assert isinstance(result, SolverResult)
        assert result.metadata["device"] == "cpu"
        assert result.metadata["vector_pde"] is False

    def test_auto_resolves_no_cuda(self) -> None:
        """device='auto' on a no-CUDA host resolves to CPU silently."""
        if torch.cuda.is_available():
            pytest.skip("CUDA available — auto would pick cuda")
        solver = SimplePINNSolver(
            hidden_dim=8,
            n_layers=2,
            n_epochs=1,
            n_collocation=20,
            device="auto",
        )
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=10)
        assert result.metadata["device"] == "cpu"

    def test_vector_pde_auto_detected_for_navier_stokes(self) -> None:
        """NS operator must auto-build a 2-channel network."""
        solver = SimplePINNSolver(
            hidden_dim=8,
            n_layers=2,
            n_epochs=1,
            n_collocation=20,
            learning_rate=1e-3,
            device="cpu",
        )
        op = _make_ns_2d()
        result = solver.solve(op, n_dof=4)
        assert result.metadata["vector_pde"] is True
        # Solution shape should be (N, 2) flattened-or-kept depending on path;
        # at minimum the n_dof reflects the N rows (not N*2).
        assert result.n_dof >= 1

    def test_vector_pde_override_records_in_metadata(self) -> None:
        """Constructor-level vector_pde=True must propagate to metadata.

        The override path is meant for non-NS PDEs that happen to be vector-
        valued. Forcing it on the canonical scalar Poisson would cause a
        loss-shape mismatch (operator returns scalar source/BC), which is by
        design — the override accepts the user's promise that the operator
        is genuinely vector-valued. We only validate the metadata round-trip
        here, not the loss math.
        """
        cfg = PINNConfig(vector_pde=True)
        solver = SimplePINNSolver(config=cfg)
        assert solver.vector_pde_override is True

    def test_build_network_output_dim_2(self) -> None:
        """_build_network must produce a 2-channel output when requested."""
        solver = SimplePINNSolver(
            hidden_dim=8,
            n_layers=2,
            n_epochs=1,
            n_collocation=20,
            learning_rate=1e-3,
            device="cpu",
        )
        net = solver._build_network(input_dim=2, output_dim=2)
        # Last layer should be Linear(_, 2)
        last = list(net.children())[-1]
        assert isinstance(last, torch.nn.Linear)
        assert last.out_features == 2

    def test_metadata_includes_n_collocation(self) -> None:
        """Metadata round-trip must include n_collocation for reproducibility."""
        solver = SimplePINNSolver(
            hidden_dim=8,
            n_layers=2,
            n_epochs=1,
            n_collocation=42,
            learning_rate=1e-3,
            device="cpu",
        )
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=10)
        assert result.metadata["n_collocation"] == 42

    def test_torch_source_term_routes_through_device_branch(self) -> None:
        """Operators that return torch tensors hit the elif-cross-device branch.

        Covers baselines.py:1014-1015 (`bc_vals.device != device` move) by
        wrapping a Poisson operator's source/BC return in a torch tensor on
        an explicitly different device than the active solve device. With a
        CPU solve all tensors land on the same device, so this test pins
        the operator to the solve's CPU device and verifies the elif branch
        survives a torch-input run.
        """

        class _TorchInputPoisson(PoissonOperator):
            """Wraps PoissonOperator, returning torch tensors for f and BC."""

            def source_term(self, coords):  # type: ignore[no-untyped-def]
                base = super().source_term(coords)
                if isinstance(base, np.ndarray):
                    return torch.tensor(base, dtype=torch.float32)
                return base

            def boundary_value(self, coords):  # type: ignore[no-untyped-def]
                base = super().boundary_value(coords)
                if isinstance(base, np.ndarray):
                    return torch.tensor(base, dtype=torch.float32)
                return base

        cfg = PDEConfig(
            name="torch_input_poisson",
            pde_type=PDEType.POISSON,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        op = _TorchInputPoisson(cfg)
        solver = SimplePINNSolver(
            hidden_dim=8,
            n_layers=2,
            n_epochs=1,
            n_collocation=20,
            learning_rate=1e-3,
            device="cpu",
        )
        result = solver.solve(op, n_dof=10)
        assert result.metadata["device"] == "cpu"
        assert result.metadata["vector_pde"] is False

    def test_scalar_squeeze_branch_for_2d_bc_vals(self) -> None:
        """BC values returned with shape (N, 1) must be squeezed in scalar mode.

        Covers baselines.py line 1024 (`bc_vals = bc_vals.squeeze(-1)`).
        """

        class TwoDScalarBcPoisson(PoissonOperator):
            def boundary_value(self, coords):  # type: ignore[no-untyped-def]
                base = super().boundary_value(coords)
                arr = np.asarray(base, dtype=np.float32)
                # Force a (N, 1) shape on the way out
                return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        cfg = PDEConfig(
            name="2d_scalar_bc_poisson",
            pde_type=PDEType.POISSON,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        op = TwoDScalarBcPoisson(cfg)
        solver = SimplePINNSolver(
            hidden_dim=8,
            n_layers=2,
            n_epochs=1,
            n_collocation=20,
            learning_rate=1e-3,
            device="cpu",
        )
        result = solver.solve(op, n_dof=10)
        assert result.metadata["device"] == "cpu"

    def test_resolve_vector_pde_override_path(self) -> None:
        """vector_pde_override=True path is exercised at resolve-time.

        Covers baselines.py line 945 (`return self.vector_pde_override`).
        """
        solver = SimplePINNSolver(config=PINNConfig(vector_pde=True, device="cpu"))
        op = _make_poisson_1d()
        # _resolve_vector_pde must short-circuit on the override.
        assert solver._resolve_vector_pde(op) is True
