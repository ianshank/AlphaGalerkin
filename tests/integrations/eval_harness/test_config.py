"""Tests for the eval-harness adapter configuration (CPU, no torch)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.integrations.eval_harness.config import BasisCellParams, OracleDatasetParams


def test_basis_cell_params_defaults() -> None:
    params = BasisCellParams(pde_family="poisson")
    assert params.arm == "random"
    assert params.seed == 0
    assert params.max_basis_functions == 8
    assert params.n_candidate_bases == 16
    assert params.target_residual == pytest.approx(1e-2)
    assert params.topk == 3
    assert params.checkpoint_path is None
    assert params.lm_studio is None


def test_basis_cell_params_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        BasisCellParams(pde_family="poisson", bogus=1)  # type: ignore[call-arg]


def test_basis_cell_params_requires_pde_family() -> None:
    with pytest.raises(ValidationError):
        BasisCellParams()  # type: ignore[call-arg]


def test_basis_cell_params_rejects_unknown_arm() -> None:
    with pytest.raises(ValidationError):
        BasisCellParams(pde_family="poisson", arm="magic")  # type: ignore[arg-type]


def test_topk_cannot_exceed_action_space() -> None:
    with pytest.raises(ValidationError):
        BasisCellParams(pde_family="poisson", n_candidate_bases=4, topk=5)


def test_oracle_dataset_params_requires_pde_families() -> None:
    with pytest.raises(ValidationError):
        OracleDatasetParams(pde_families=[])


def test_oracle_dataset_iter_cells_cartesian_product() -> None:
    params = OracleDatasetParams(
        pde_families=["poisson", "burgers"],
        seeds=[1, 2, 3],
        arm="random",
        n_candidate_bases=12,
        topk=2,
    )
    cells = list(params.iter_cells())
    assert len(cells) == 6
    assert {c.pde_family for c in cells} == {"poisson", "burgers"}
    assert {c.seed for c in cells} == {1, 2, 3}
    # Shared knobs are propagated to every cell.
    assert all(c.arm == "random" and c.n_candidate_bases == 12 and c.topk == 2 for c in cells)
    assert all(isinstance(c, BasisCellParams) for c in cells)


def test_oracle_dataset_default_seed() -> None:
    params = OracleDatasetParams(pde_families=["poisson"])
    cells = list(params.iter_cells())
    assert len(cells) == 1
    assert cells[0].seed == 0
