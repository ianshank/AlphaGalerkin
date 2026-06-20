"""BasisOracleDataset + jsonl cache builder (requires the torch stack)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.integrations.eval_harness.config import OracleDatasetParams  # noqa: E402
from src.integrations.eval_harness.dataset import (  # noqa: E402
    BasisOracleDataset,
    build_dataset_jsonl,
)


def test_basis_oracle_dataset_yields_labeled_items() -> None:
    dataset = BasisOracleDataset(
        pde_families=["poisson"],
        seeds=[0, 1],
        arm="random",
        max_basis_functions=3,
        n_candidate_bases=8,
        target_residual=1e-3,
        topk=3,
    )
    items = list(dataset.load())
    assert len(items) == 2
    for item in items:
        assert item.expected["ranked_actions"]
        assert item.expected["greedy_action"] in item.expected["ranked_actions"]
        assert item.inputs["pde_family"] == "poisson"
        assert item.metadata["pde_family"] == "poisson"


def test_build_dataset_jsonl_round_trip(tmp_path: Path) -> None:
    params = OracleDatasetParams(pde_families=["poisson"], seeds=[0], n_candidate_bases=8)
    path = tmp_path / "labels.jsonl"
    count = build_dataset_jsonl(params, path)
    assert count == 1
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == "poisson/seed0"
    assert record["expected"]["ranked_actions"]
