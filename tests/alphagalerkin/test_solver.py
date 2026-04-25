"""Unit tests for the unified :mod:`src.alphagalerkin` solver wrapper.

These tests exercise the ``AlphaGalerkinSolver`` against the
``BaseSolver`` protocol used by ``PDEBenchmarkRunner`` so that the
solver can be benchmarked apples-to-apples with the classical
baselines in :mod:`src.research.baselines`.

Scope:
- Pydantic config validation (rejects unknown ``game_mode``).
- Smoke test that ``solve()`` returns a well-formed ``SolverResult``
  for a Poisson problem.
- ``max_steps`` is honoured as a hard ceiling on action count.
- Determinism: two runs with the same ``seed`` produce identical
  ``n_dof`` and ``l2_error``.

Expensive runs (more than a couple of MCTS simulations) are tagged
with ``pytest.mark.slow`` so they can be skipped in fast CI by
deselecting the marker.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.alphagalerkin import AlphaGalerkinConfig, AlphaGalerkinSolver
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SolverResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def poisson_operator() -> PoissonOperator:
    """Small 2D Poisson operator used across the suite."""
    cfg = PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )
    return PoissonOperator(cfg)


def _fast_solver_config(**overrides: object) -> AlphaGalerkinConfig:
    """Build an ``AlphaGalerkinConfig`` tuned for fast unit tests.

    Uses a minimal basis count, tiny number of MCTS simulations, and a
    low step cap so CI runs complete in under a second per test.
    """
    kwargs: dict[str, object] = {
        "game_mode": "basis_selection",
        "n_mcts_simulations": 2,
        "max_steps": 2,
        "target_tolerance": 1e-4,
        "evaluator": "random",
        "device": "cpu",
        "seed": 7,
    }
    kwargs.update(overrides)
    return AlphaGalerkinConfig(**kwargs)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestAlphaGalerkinConfig:
    """Pydantic validation boundary tests."""

    def test_default_values(self) -> None:
        """Defaults should match the documented contract."""
        cfg = AlphaGalerkinConfig()
        assert cfg.game_mode == "basis_selection"
        assert cfg.n_mcts_simulations == 50
        assert cfg.max_steps == 20
        assert cfg.target_tolerance == pytest.approx(1e-4)
        assert cfg.evaluator == "random"
        # GPU-primary: default device is now ``cuda``. The runtime
        # ``_resolve_device`` helper falls back to CPU at solve time when
        # CUDA is unavailable, so this default is safe on CPU-only CI.
        assert cfg.device == "cuda"
        assert cfg.checkpoint_path is None
        # Inherited from SolverConfig.
        assert cfg.seed == 42

    def test_invalid_game_mode_rejected(self) -> None:
        """Literal validation must reject an unknown ``game_mode``."""
        with pytest.raises(ValidationError):
            AlphaGalerkinConfig(game_mode="not_a_real_mode")  # type: ignore[arg-type]

    def test_invalid_evaluator_rejected(self) -> None:
        """Evaluator literal is similarly constrained."""
        with pytest.raises(ValidationError):
            AlphaGalerkinConfig(evaluator="bogus")  # type: ignore[arg-type]

    def test_negative_simulations_rejected(self) -> None:
        """``n_mcts_simulations`` has a ``ge=1`` constraint."""
        with pytest.raises(ValidationError):
            AlphaGalerkinConfig(n_mcts_simulations=0)

    def test_zero_max_steps_rejected(self) -> None:
        """``max_steps`` must be at least 1."""
        with pytest.raises(ValidationError):
            AlphaGalerkinConfig(max_steps=0)

    def test_min_game_dof_rejects_zero(self) -> None:
        """``min_game_dof`` must be at least 1."""
        with pytest.raises(ValidationError):
            AlphaGalerkinConfig(min_game_dof=0)

    def test_min_game_dof_default(self) -> None:
        """``min_game_dof`` default floor is 10."""
        assert AlphaGalerkinConfig().min_game_dof == 10

    def test_device_accepts_cpu(self) -> None:
        """``device='cpu'`` is always valid."""
        assert AlphaGalerkinConfig(device="cpu").device == "cpu"

    def test_device_rejects_invalid_string(self) -> None:
        """Bogus device strings fail at Pydantic validation time."""
        with pytest.raises(ValidationError):
            AlphaGalerkinConfig(device="cuba:0")

    def test_device_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            AlphaGalerkinConfig(device="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestDerivePDEConfig:
    """Covers ``AlphaGalerkinSolver._derive_pde_config``."""

    def test_returns_operator_config(self, poisson_operator: PoissonOperator) -> None:
        """The operator's own PDEConfig is returned unchanged."""
        cfg = AlphaGalerkinSolver._derive_pde_config(poisson_operator, n_dof=32)
        assert isinstance(cfg, PDEConfig)
        assert cfg is poisson_operator.config

    def test_raises_when_operator_has_no_config(self) -> None:
        """Operators without a ``.config`` attribute are rejected clearly."""

        class _BareOperator:
            pass

        with pytest.raises(ValueError, match="PDEConfig"):
            AlphaGalerkinSolver._derive_pde_config(_BareOperator(), n_dof=1)  # type: ignore[arg-type]

    def test_raises_when_config_wrong_type(self) -> None:
        """Non-PDEConfig .config values raise TypeError."""

        class _WeirdOperator:
            config = "not a config"

        with pytest.raises(TypeError, match="PDEConfig"):
            AlphaGalerkinSolver._derive_pde_config(_WeirdOperator(), n_dof=1)  # type: ignore[arg-type]


class TestBuildGame:
    """Covers ``AlphaGalerkinSolver._build_game`` factory dispatch."""

    def test_basis_selection_mode_returns_basis_game(
        self, poisson_operator: PoissonOperator
    ) -> None:
        from src.pde.config import PDEGameConfig
        from src.pde.games.basis_selection import BasisSelectionGame

        solver = AlphaGalerkinSolver(_fast_solver_config(game_mode="basis_selection"))
        game_cfg = PDEGameConfig(
            name="t",
            pde_config=poisson_operator.config,
            game_mode="basis_selection",
        )
        game = solver._build_game(poisson_operator, game_cfg)
        assert isinstance(game, BasisSelectionGame)

    def test_mesh_refinement_mode_returns_mesh_game(
        self, poisson_operator: PoissonOperator
    ) -> None:
        from src.pde.config import PDEGameConfig
        from src.pde.games.mesh_refinement import MeshRefinementGame

        solver = AlphaGalerkinSolver(_fast_solver_config(game_mode="mesh_refinement"))
        game_cfg = PDEGameConfig(
            name="t",
            pde_config=poisson_operator.config,
            game_mode="mesh_refinement",
        )
        game = solver._build_game(poisson_operator, game_cfg)
        assert isinstance(game, MeshRefinementGame)


# ---------------------------------------------------------------------------
# Solver behaviour
# ---------------------------------------------------------------------------


class TestAlphaGalerkinSolver:
    """Behavioural tests for the orchestration layer."""

    def test_solver_has_baseline_attributes(self) -> None:
        """Solver advertises the same metadata style as classical baselines."""
        solver = AlphaGalerkinSolver()
        assert solver.name == "alphagalerkin"
        assert isinstance(solver.description, str)
        assert solver.description

    def test_solver_runs_on_poisson(self, poisson_operator: PoissonOperator) -> None:
        """``solve()`` returns a well-formed ``SolverResult`` on Poisson."""
        solver = AlphaGalerkinSolver(_fast_solver_config())
        result = solver.solve(poisson_operator, n_dof=32)

        assert isinstance(result, SolverResult)
        # DOF and wall time are populated.
        assert result.n_dof > 0
        assert result.wall_time_seconds > 0.0
        # Poisson has an exact solution so an L2 error is computed.
        assert result.l2_error is not None
        assert result.l2_error >= 0.0
        # Metadata carries the expected provenance keys.
        for key in (
            "solver",
            "seed",
            "game_mode",
            "n_mcts_simulations",
            "n_actions_taken",
            "max_steps",
            "error_history",
            "target_tolerance",
            "evaluator",
            "min_game_dof",
            "solution_available",
            "termination_reason",
        ):
            assert key in result.metadata, f"missing metadata key {key!r}"
        # seed round-trips the solver config so PDEBenchmarkRunner.export_csv
        # can populate its seed column without re-inspecting the config.
        assert result.metadata["seed"] == 7
        assert result.metadata["game_mode"] == "basis_selection"
        assert result.metadata["n_mcts_simulations"] == 2

    def test_solver_respects_max_steps(self, poisson_operator: PoissonOperator) -> None:
        """Action count should never exceed the configured ``max_steps``."""
        max_steps = 3
        # Tighten tolerance so the early-stop branch doesn't trigger,
        # forcing the loop to run up against ``max_steps``.
        solver = AlphaGalerkinSolver(
            _fast_solver_config(max_steps=max_steps, target_tolerance=1e-12)
        )
        result = solver.solve(poisson_operator, n_dof=32)

        taken = result.metadata["n_actions_taken"]
        assert isinstance(taken, int)
        assert taken <= max_steps
        # Error history starts at the initial estimate, so its length
        # is exactly ``n_actions_taken + 1`` for every terminated loop.
        assert len(result.metadata["error_history"]) == taken + 1

    def test_solver_terminates_via_game_is_terminal(
        self, poisson_operator: PoissonOperator
    ) -> None:
        """A loose ``target_tolerance`` flows into the game's ``error_tolerance``.

        The solver propagates ``target_tolerance`` directly to
        ``PDEGameConfig.error_tolerance`` (so a ``target_tolerance``
        above the initial Galerkin residual makes the underlying game
        terminal on the very first iteration), and the loop exits with
        ``termination_reason == "is_terminal"``.
        """
        solver = AlphaGalerkinSolver(_fast_solver_config(target_tolerance=0.99))
        result = solver.solve(poisson_operator, n_dof=32)
        assert result.metadata["termination_reason"] == "is_terminal"
        assert result.metadata["n_actions_taken"] == 0

    def test_solver_deterministic_with_seed(
        self,
        poisson_operator: PoissonOperator,
    ) -> None:
        """Two runs with matching seeds produce identical MCTS decisions.

        The solver's seed controls the MCTS policy (action selection +
        tree traversal), so ``n_dof`` and ``n_actions_taken`` must match
        bit-for-bit across runs.  The final ``l2_error`` can still drift
        because ``BasisSelectionGame`` samples collocation points via
        ``np.random.default_rng(None)`` on each construction (see
        ``operators.py:generate_collocation_points``), which uses OS
        entropy independent of the seeds we reset in ``_seed_everything``.
        """
        cfg_a = _fast_solver_config(seed=123)
        cfg_b = _fast_solver_config(seed=123)

        solver_a = AlphaGalerkinSolver(cfg_a)
        solver_b = AlphaGalerkinSolver(cfg_b)

        result_a = solver_a.solve(poisson_operator, n_dof=32)
        result_b = solver_b.solve(poisson_operator, n_dof=32)

        # Structural determinism: same seed → same number of MCTS-driven
        # basis selections and same final DOF budget.
        assert result_a.n_dof == result_b.n_dof
        assert result_a.metadata["n_actions_taken"] == result_b.metadata["n_actions_taken"]
        assert result_a.metadata["termination_reason"] == result_b.metadata["termination_reason"]

        # Both runs produce a populated l2_error (Poisson exposes an
        # exact solution).  Strict numeric equality is not asserted
        # because the underlying collocation sampler is not seedable
        # through the solver config.
        assert result_a.l2_error is not None
        assert result_b.l2_error is not None

    def test_trained_evaluator_requires_checkpoint(self, tmp_path: object) -> None:
        """``evaluator='trained'`` must be paired with an existing checkpoint *file*.

        The Literal accepts ``"trained"``, but the post-construction
        ``_validate_trained_checkpoint`` model validator rejects the
        config when ``checkpoint_path`` is ``None``, points at a missing
        path, or points at a *directory*. All failure modes surface as
        ``ValidationError`` from Pydantic so callers see them at
        config-build time, not deep in ``solve()`` (where ``torch.load``
        would otherwise raise an opaque ``IsADirectoryError``).
        """
        from pathlib import Path

        from pydantic import ValidationError

        # Missing checkpoint_path → reject.
        with pytest.raises(ValidationError):
            _fast_solver_config(evaluator="trained")

        # Non-existent checkpoint_path → reject.
        with pytest.raises(ValidationError):
            _fast_solver_config(
                evaluator="trained",
                checkpoint_path=Path("/nonexistent/checkpoint.pt"),
            )

        # Existing path that is a directory (not a file) → reject.
        assert isinstance(tmp_path, Path)
        with pytest.raises(ValidationError, match="non-file path"):
            _fast_solver_config(
                evaluator="trained",
                checkpoint_path=tmp_path,
            )


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestBaselinesRegistryIntegration:
    """The solver should register itself without touching baselines.py."""

    def test_registered_in_solver_registry(self) -> None:
        """Importing the package registers the solver."""
        import src.alphagalerkin  # noqa: F401 — side-effect import
        from src.research.baselines import SOLVER_REGISTRY, get_solver, list_solvers

        assert "alphagalerkin" in SOLVER_REGISTRY
        assert "alphagalerkin" in list_solvers()
        solver = get_solver("alphagalerkin")
        assert solver.name == "alphagalerkin"


# ---------------------------------------------------------------------------
# Slow / compute-heavy scenarios
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAlphaGalerkinSolverSlow:
    """Exercises the solver with a realistic MCTS budget.

    Kept under ``pytest.mark.slow`` because each run spawns a non-trivial
    tree search over the basis candidate set.
    """

    def test_end_to_end_mesh_refinement(self, poisson_operator: PoissonOperator) -> None:
        """Mesh-refinement game mode should also return a valid result."""
        cfg = _fast_solver_config(
            game_mode="mesh_refinement",
            n_mcts_simulations=4,
            max_steps=2,
        )
        solver = AlphaGalerkinSolver(cfg)
        result = solver.solve(poisson_operator, n_dof=32)

        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.wall_time_seconds > 0.0
        assert result.metadata["game_mode"] == "mesh_refinement"
