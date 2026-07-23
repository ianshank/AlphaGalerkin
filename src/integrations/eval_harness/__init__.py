"""AlphaGalerkin ↔ ``langfuse-eval-harness`` adapter.

Plugs AlphaGalerkin's LLM-prior MCTS basis-selection layer into the external
`langfuse-eval-harness <https://github.com/ianshank/Agents>`_ framework so the
LLM policy can be **traced** (Langfuse) and **scored against a labelled oracle**,
with the resulting metrics fed back into the existing ``src/poc/baselines``
regression gate.

The harness itself is an optional dependency declared under the ``[eval-harness]``
extra in ``pyproject.toml`` and pinned to a commit SHA (it is not published on
PyPI). The base install never imports it: ``eval_harness`` is imported lazily
inside the modules that need it (:mod:`~src.integrations.eval_harness.runner`,
:mod:`~src.integrations.eval_harness.scorers`, :mod:`~src.integrations.eval_harness.sink`,
:mod:`~src.integrations.eval_harness.dataset`), so importing *this* package only
pulls the (light) Pydantic config.

Components:
    - :class:`~src.integrations.eval_harness.config.BasisCellParams`,
      :class:`~src.integrations.eval_harness.config.OracleDatasetParams`: validated,
      no-hardcoded-values configuration for one MCTS cell and a labelled dataset.
    - ``target.run_basis_cell``: the harness built-in ``callable`` target.
    - ``oracle.greedy_basis_oracle``: the (myopic) 1-step greedy label generator.
    - ``scorers.{FinalResidualScorer, PolicyTopKScorer}``: harness ``Scorer`` adapters.
    - ``sink.ScenarioResultSink``: bridges a harness ``RunResult`` into the PoC
      ``ScenarioResult`` JSON layout the ``record-baseline``/``diff`` CLI reads.
    - ``runner.run_eval``: load a harness ``EvalConfig`` YAML and execute it.
    - ``plugins.register_all``: idempotently register the adapters into the harness.
"""

from src.integrations.eval_harness.config import BasisCellParams, OracleDatasetParams

__all__ = ["BasisCellParams", "OracleDatasetParams"]
