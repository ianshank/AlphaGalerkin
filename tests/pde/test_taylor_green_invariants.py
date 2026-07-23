"""Cross-branch invariants for the Taylor-Green vortex exact solution.

Guards against the numpy/torch drift that produced PR follow-up #X — the
``NavierStokesOperator.exact_solution`` had a typo that only affected the
numpy branch (``cos(x)*cos(y)`` instead of ``sin(x)*cos(y)`` for ``uy``).
The two implementation paths must agree elementwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import NavierStokesOperator


def _make_operator(reynolds_number: float = 100.0) -> NavierStokesOperator:
    two_pi = 2.0 * float(np.pi)
    cfg = PDEConfig(
        name="test_tg",
        pde_type=PDEType.NAVIER_STOKES,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[two_pi, two_pi],
        advection_coeff=[0.0, 0.0],
    )
    return NavierStokesOperator(cfg, reynolds_number=reynolds_number)


@pytest.mark.parametrize("time", [0.0, 0.1, 0.5, 1.0])
def test_numpy_and_torch_branches_agree(time: float) -> None:
    """Numpy and torch implementations must produce identical fields."""
    operator = _make_operator()
    rng = np.random.default_rng(42)
    coords_np = rng.uniform(0.0, 2.0 * np.pi, size=(64, 2)).astype(np.float32)
    coords_torch = torch.from_numpy(coords_np)

    out_np = np.asarray(operator.exact_solution(coords_np, time=time))
    out_torch = operator.exact_solution(coords_torch, time=time).detach().cpu().numpy()

    np.testing.assert_allclose(out_np, out_torch, atol=1e-6, rtol=1e-6)


def test_uy_formula_numpy() -> None:
    """Numpy branch must use sin(x)*cos(y), not cos(x)*cos(y)."""
    operator = _make_operator()
    coords = np.array([[1.0, 1.0]], dtype=np.float32)

    out = np.asarray(operator.exact_solution(coords, time=0.0))
    expected_uy = np.sin(1.0) * np.cos(1.0)

    assert out[0, 1] == pytest.approx(expected_uy, abs=1e-5)


def test_initial_condition_matches_exact_at_t0() -> None:
    """initial_condition() must equal exact_solution(t=0) on both branches."""
    operator = _make_operator()
    coords_np = np.array([[0.5, 1.5], [2.0, 3.0]], dtype=np.float32)

    ic = np.asarray(operator.initial_condition(coords_np))
    exact_t0 = np.asarray(operator.exact_solution(coords_np, time=0.0))

    np.testing.assert_allclose(ic, exact_t0, atol=1e-6)
