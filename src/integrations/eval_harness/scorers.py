"""Harness ``Scorer`` adapters for AlphaGalerkin basis-selection runs.

Two scorers operate on the dict returned by
:func:`~src.integrations.eval_harness.target.run_basis_cell`:

- :class:`FinalResidualScorer` (label-free): the achieved final residual; passes
  when ``residual <= target_residual``. The real outcome metric.
- :class:`PolicyTopKScorer` (label-bearing): whether the arm's root ``chosen_action``
  is within the greedy oracle's top-k. **Myopic** — the greedy 1-step oracle is a
  sanity/alignment signal, not a proof of multi-step optimality.

Both subclass the harness ``Scorer`` ABC. ``eval_harness`` ships no type stubs, so
the base resolves to ``Any`` under ``mypy --strict``; the project's ``src.*`` mypy
override disables ``misc`` (subclassing-Any) so no per-line ignore is needed. The
modules imported here are torch-free, so registering these scorers triggers no
heavy imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval_harness.core.interfaces import Scorer
from eval_harness.core.types import ScoreResult

if TYPE_CHECKING:
    from eval_harness.core.types import EvalItem, RunContext, TargetOutput

DEFAULT_TARGET_RESIDUAL: float = 1e-2
"""Default residual threshold when a scorer is constructed without ``target_residual``."""

DEFAULT_TOPK: int = 3
"""Default top-k cutoff when :class:`PolicyTopKScorer` is constructed without ``k``."""

FAILED_RESIDUAL_SENTINEL: float = 1.0e6
"""Finite residual recorded when the target errored or omitted ``final_residual``.

A large finite value (rather than ``inf``) so the aggregate mean stays
JSON-serialisable and a failed cell deterministically fails a ``max`` gate.
"""


class FinalResidualScorer(Scorer):  # eval_harness ships no stubs; mypy 'misc' off for src.*
    """Score the achieved final residual of a basis-selection cell (lower better)."""

    default_name = "final_residual"

    def __init__(
        self,
        name: str | None = None,
        target_residual: float = DEFAULT_TARGET_RESIDUAL,
    ) -> None:
        """Construct the scorer.

        Args:
            name: Optional score label (defaults to ``"final_residual"``).
            target_residual: Pass threshold; a cell passes when its residual is
                at or below this value.

        """
        super().__init__(name)
        if target_residual <= 0.0:
            raise ValueError(f"target_residual must be > 0, got {target_residual!r}")
        self.target_residual = float(target_residual)

    def score(self, item: EvalItem, output: TargetOutput, ctx: RunContext) -> ScoreResult:
        """Return the cell's final residual and whether it met the threshold."""
        out = output.output
        if output.error is not None or not isinstance(out, dict) or "final_residual" not in out:
            return ScoreResult(
                name=self.name,
                value=FAILED_RESIDUAL_SENTINEL,
                passed=False,
                comment=output.error or "missing final_residual in target output",
            )
        residual = float(out["final_residual"])
        return ScoreResult(
            name=self.name,
            value=residual,
            passed=residual <= self.target_residual,
            comment=f"residual {residual:.3g} vs target {self.target_residual:.3g}",
            metadata={"rollouts_used": out.get("rollouts_used")},
        )


class PolicyTopKScorer(Scorer):  # eval_harness ships no stubs; mypy 'misc' off for src.*
    """Score whether the arm's root choice is within the greedy oracle's top-k.

    Requires a labelled item (``item.expected['ranked_actions']``); unlabelled
    items return ``value=0.0, passed=None`` so they neither pass nor fail.
    """

    default_name = "policy_topk"

    def __init__(self, name: str | None = None, k: int = DEFAULT_TOPK) -> None:
        """Construct the scorer.

        Args:
            name: Optional score label (defaults to ``"policy_topk"``).
            k: Top-k cutoff; a hit is the arm's ``chosen_action`` appearing in the
                oracle's first ``k`` ranked actions.

        """
        super().__init__(name)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        self.k = int(k)

    def score(self, item: EvalItem, output: TargetOutput, ctx: RunContext) -> ScoreResult:
        """Return 1.0 if the root choice is in the oracle top-k, else 0.0."""
        expected = item.expected
        if not isinstance(expected, dict) or not expected.get("ranked_actions"):
            return ScoreResult(
                name=self.name,
                value=0.0,
                passed=None,
                comment="no oracle label (ranked_actions missing)",
            )
        out = output.output
        if (
            output.error is not None
            or not isinstance(out, dict)
            or out.get("chosen_action") is None
        ):
            return ScoreResult(
                name=self.name,
                value=0.0,
                passed=False,
                comment=output.error or "missing chosen_action in target output",
            )
        chosen = int(out["chosen_action"])
        ranked = [int(a) for a in expected["ranked_actions"]]
        in_topk = chosen in ranked[: self.k]
        top1 = bool(ranked) and chosen == ranked[0]
        return ScoreResult(
            name=self.name,
            value=1.0 if in_topk else 0.0,
            passed=in_topk,
            comment=f"chosen={chosen} oracle_top1={ranked[0] if ranked else None} k={self.k}",
            metadata={"top1": 1.0 if top1 else 0.0, "k": self.k},
        )


__all__ = [
    "DEFAULT_TARGET_RESIDUAL",
    "DEFAULT_TOPK",
    "FAILED_RESIDUAL_SENTINEL",
    "FinalResidualScorer",
    "PolicyTopKScorer",
]
