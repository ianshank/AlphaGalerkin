"""``BasisOracleDataset`` — labelled basis-selection items for the harness.

Each :class:`~eval_harness.core.types.EvalItem` is one MCTS cell whose
``inputs`` is a :class:`BasisCellParams` dump and whose ``expected`` is the greedy
1-step oracle ranking (:func:`~src.integrations.eval_harness.oracle.greedy_basis_oracle`)
at the initial state — the ground truth the :class:`PolicyTopKScorer` checks.

Two access paths:

- ``basis_oracle`` dataset (this class) builds items live (torch) on ``load()``.
- :func:`build_dataset_jsonl` precomputes the labels once and writes a jsonl the
  built-in ``jsonl`` dataset can read, so the oracle cost is paid once rather than
  per harness run.

The class definition and ``load`` signature are torch-free; the heavy work lives
in :func:`_build_items`, imported lazily, so registering this dataset triggers no
torch import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval_harness.core.interfaces import DatasetSource
from eval_harness.core.types import EvalItem

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.integrations.eval_harness.config import OracleDatasetParams


class BasisOracleDataset(DatasetSource):  # eval_harness ships no stubs; mypy 'misc' off for src.*
    """Dataset of labelled basis-selection cells (label = greedy 1-step oracle)."""

    def __init__(self, **params: Any) -> None:
        """Validate the ``basis_oracle`` dataset params (an OracleDatasetParams dump)."""
        from src.integrations.eval_harness.config import OracleDatasetParams  # noqa: PLC0415

        self._params = OracleDatasetParams.model_validate(params)

    def load(self) -> Iterable[EvalItem]:
        """Build and return the labelled items (torch is imported here, lazily)."""
        return _build_items(self._params)


def _build_items(params: OracleDatasetParams) -> list[EvalItem]:
    """Construct one labelled :class:`EvalItem` per ``(pde_family, seed)`` cell."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from src.integrations.eval_harness.oracle import greedy_basis_oracle  # noqa: PLC0415
    from src.poc.scenarios._centaur_common import (  # noqa: PLC0415
        build_basis_game,
        build_pde_operator,
    )

    items: list[EvalItem] = []
    for cell in params.iter_cells():
        np.random.seed(cell.seed)
        torch.manual_seed(cell.seed)
        operator = build_pde_operator(cell.pde_family)
        game = build_basis_game(
            cell.pde_family,
            operator,
            max_basis_functions=cell.max_basis_functions,
            n_candidate_bases=cell.n_candidate_bases,
            target_residual=cell.target_residual,
        )
        label = greedy_basis_oracle(game, game.get_initial_state())
        items.append(
            EvalItem(
                id=f"{cell.pde_family}/seed{cell.seed}",
                inputs=cell.model_dump(),
                expected=label,
                metadata={"pde_family": cell.pde_family, "seed": cell.seed},
            )
        )
    return items


def build_dataset_jsonl(params: OracleDatasetParams, path: str | Path) -> int:
    """Precompute oracle labels and write a jsonl the built-in ``jsonl`` reads.

    Args:
        params: The dataset sweep spec.
        path: Output jsonl path (parent dirs created).

    Returns:
        The number of items written.

    """
    items = _build_items(params)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {
                        "id": item.id,
                        "inputs": item.inputs,
                        "expected": item.expected,
                        "metadata": item.metadata,
                    }
                )
                + "\n"
            )
    return len(items)


__all__ = ["BasisOracleDataset", "build_dataset_jsonl"]
