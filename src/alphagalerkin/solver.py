"""Unified AlphaGalerkin solver wrapper for benchmark comparison.

Wraps the MCTS + PDE-game stack behind the ``BaseSolver`` protocol
defined in ``src.research.baselines`` so that ``PDEBenchmarkRunner``
can compare AlphaGalerkin against classical solvers on an equal footing.

The solver does not train a network: it uses a ``RandomEvaluator``
(``random`` and ``uniform`` are accepted evaluator modes; the latter is
an alias for the former until a learned evaluator is wired in) so the
MCTS explores the PDE game purely via the tree search policy.  The result is
the canonical ``(l2_error, n_dof, wall_time_seconds)`` triple plus a
solution vector (when the underlying game exposes one) and rich metadata
for downstream analysis.

Design notes
------------
- ``AlphaGalerkinConfig`` extends ``SolverConfig`` so it inherits the
  ``seed``/``max_iterations``/``tolerance`` fields already used by the
  classical baselines.
- The solver registers itself with ``baselines.SOLVER_REGISTRY`` at
  import time via the ``_register_with_baselines`` helper, using
  ``setdefault`` so the side effect is idempotent.  This means
  ``get_solver("alphagalerkin")`` works after any ``import
  src.alphagalerkin`` (including transitive imports through
  ``PDEBenchmarkRunner``) with no explicit wiring.  The helper is
  exposed as a module-level function so callers that want to strip
  the side effect (for example, when loading the package in an
  environment where the MCTS stack is unavailable) can skip it by
  monkey-patching before import.
- For ``mesh_refinement`` game mode the underlying game reconstructs
  the state's solution vector on the adaptive mesh; we pass that
  through in ``SolverResult.solution``.  For ``basis_selection`` the
  solution is the Galerkin approximation sampled at the collocation
  points.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog
import torch
from pydantic import Field, field_validator

from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS
from src.pde.config import PDEConfig, PDEGameConfig
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.games.mesh_refinement import MeshRefinementGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.research.baselines import BaseSolver, SolverConfig, SolverResult
from src.research.metadata_keys import (
    METADATA_KEY_ERROR_HISTORY,
    METADATA_KEY_EVALUATOR,
    METADATA_KEY_GAME_MODE,
    METADATA_KEY_MAX_STEPS,
    METADATA_KEY_MIN_GAME_DOF,
    METADATA_KEY_N_ACTIONS_TAKEN,
    METADATA_KEY_N_MCTS_SIMULATIONS,
    METADATA_KEY_SEED,
    METADATA_KEY_SOLUTION_AVAILABLE,
    METADATA_KEY_SOLVER,
    METADATA_KEY_TARGET_TOLERANCE,
    METADATA_KEY_TERMINATION_REASON,
)

if TYPE_CHECKING:
    from src.pde.game import PDEGame
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


class AlphaGalerkinConfig(SolverConfig):
    """Configuration for the unified :class:`AlphaGalerkinSolver`.

    Inherits ``seed``, ``max_iterations`` and ``tolerance`` from
    :class:`~src.research.baselines.SolverConfig`.

    Attributes:
        game_mode: Which PDE game to drive (basis-function selection
            or adaptive mesh refinement).
        n_mcts_simulations: Simulations per MCTS search call.
        max_steps: Hard cap on game actions per episode.
        target_tolerance: Early-stop threshold on the game's L2 error.
        evaluator: Evaluator type used by MCTS.
        device: Torch device for the evaluator.

    """

    game_mode: Literal["basis_selection", "mesh_refinement"] = Field(
        default="basis_selection",
        description="PDE game formulation used to drive the search.",
    )
    n_mcts_simulations: int = Field(
        default=50,
        ge=1,
        description="Number of MCTS simulations per search call.",
    )
    max_steps: int = Field(
        default=20,
        ge=1,
        description="Maximum game actions applied per episode.",
    )
    target_tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        description="Early-stop threshold on the game error estimate.",
    )
    evaluator: Literal["random", "uniform"] = Field(
        default="random",
        description=(
            "Evaluator strategy. Both 'random' and 'uniform' map to the "
            "RandomEvaluator (uniform prior + zero value) until a network-"
            "backed evaluator is wired in."
        ),
    )
    device: str = Field(
        default="cpu",
        description=(
            "Torch device spec for the evaluator (accepted values match "
            "``torch.device`` - e.g. 'cpu', 'cuda', 'cuda:0'). Validated at "
            "config-construction time via ``torch.device(...)`` so invalid "
            "strings fail fast. Used to log the active device at solve time "
            "and reserved for the future network-backed evaluator."
        ),
    )
    min_game_dof: int = Field(
        default=10,
        ge=1,
        description="Floor on the DOF budget passed to PDEGameConfig.max_dof.",
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        """Reject bogus ``device`` strings at config construction.

        ``torch.device`` raises ``RuntimeError`` on unknown device types or
        malformed indices (e.g. ``'cuba:0'``, ``'cuda:foo'``).  We catch
        that and surface a ``ValueError`` so Pydantic reports it via its
        normal validation machinery (``ValidationError``).
        """
        try:
            torch.device(value)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid torch device string: {value!r} ({exc})") from exc
        return value


class AlphaGalerkinSolver(BaseSolver):
    """MCTS-guided AlphaGalerkin solver exposed via the BaseSolver protocol.

    ``solve()`` builds a ``PDEGame`` for the supplied ``PDEOperator``,
    wraps it in a ``PDEGameAdapter``, and runs an MCTS-driven episode
    for up to :attr:`AlphaGalerkinConfig.max_steps` actions.  The final
    L2 error, degree-of-freedom count and wall time are returned in a
    :class:`~src.research.baselines.SolverResult`.

    The solver does not modify the underlying PDE modules or the MCTS
    core - it is a thin orchestration layer.
    """

    name: str = "alphagalerkin"
    description: str = "MCTS-guided Galerkin / mesh refinement solver"

    def __init__(self, config: AlphaGalerkinConfig | None = None) -> None:
        """Initialize the solver.

        Args:
            config: Solver configuration. A default is used if omitted.

        """
        self.config: AlphaGalerkinConfig = config or AlphaGalerkinConfig()

    # ------------------------------------------------------------------ #
    # BaseSolver API                                                      #
    # ------------------------------------------------------------------ #

    def solve(
        self,
        operator: PDEOperator,
        n_dof: int = 0,
        **kwargs: Any,
    ) -> SolverResult:
        """Run an MCTS-guided PDE game episode.

        Matches ``BaseSolver.solve(operator, n_dof, **kwargs)``.  The
        ``n_dof`` argument is interpreted as a soft hint (equivalent to
        ``max_dof`` for the underlying ``PDEGameConfig``); the actual
        degree-of-freedom count in the returned ``SolverResult`` is
        taken from the terminal game state.

        Args:
            operator: PDE operator defining the problem.
            n_dof: Soft DOF budget hint (``0`` keeps the game default).
            **kwargs: Optional ``domain_params`` dict plus forward-
                compatible keyword arguments.

        Returns:
            ``SolverResult`` with ``l2_error``, ``n_dof`` and
            ``wall_time_seconds`` populated.  Metadata includes the
            per-step error history and the MCTS simulation count.

        """
        # Silence unused-argument warnings while preserving the protocol.
        _ = kwargs.pop("domain_params", None)

        self._seed_everything(self.config.seed)

        log = logger.bind(
            solver=self.name,
            run_id=f"ag_{self.config.seed}_{self.config.game_mode}",
            game_mode=self.config.game_mode,
            n_mcts_simulations=self.config.n_mcts_simulations,
        )
        log.info("alphagalerkin_solve_start", pde=getattr(operator, "name", "unknown"))

        t0 = time.perf_counter()

        # Build PDE + game config from operator metadata so the game
        # sees the same domain / BCs the operator was built with.
        pde_config = self._derive_pde_config(operator, n_dof)
        game_config = PDEGameConfig(
            name=f"ag_{self.config.game_mode}",
            pde_config=pde_config,
            game_mode=self.config.game_mode,
            max_steps=self.config.max_steps,
            max_dof=max(n_dof, self.config.min_game_dof),
            error_tolerance=self.config.target_tolerance,
            seed=self.config.seed,
        )

        pde_game = self._build_game(operator, game_config)
        adapter = PDEGameAdapter(pde_game)
        mcts = self._build_mcts(pde_game)

        # Main episode loop.  We call ``mcts.search`` to obtain an
        # improved policy, pick the mode-argmax action, and advance the
        # adapter until termination / tolerance / max_steps.  The actual
        # break reason is captured in ``termination_reason`` so metadata
        # distinguishes e.g. ``no_legal_actions`` from ``max_steps``.
        n_actions_taken = 0
        termination_reason = "max_steps"
        for step in range(self.config.max_steps):
            if adapter.is_terminal():
                termination_reason = "is_terminal"
                log.debug("terminated_early", step=step, reason=termination_reason)
                break
            # ``adapter.current_error`` is a reduction *fraction*; the raw
            # error estimate lives on the state and is what the tolerance
            # contract is defined against.
            current_error = adapter.state.error_estimate
            if current_error < self.config.target_tolerance:
                termination_reason = "tolerance"
                log.debug(
                    "terminated_early",
                    step=step,
                    reason=termination_reason,
                    error=current_error,
                )
                break

            legal = adapter.get_legal_actions()
            if not legal:
                termination_reason = "no_legal_actions"
                log.debug("terminated_early", step=step, reason=termination_reason)
                break

            policy = mcts.search(adapter, add_noise=False)
            if not policy:
                termination_reason = "empty_policy"
                log.debug("terminated_early", step=step, reason=termination_reason)
                break

            # Mode-argmax (deterministic given the MCTS visit counts).
            action = max(policy, key=lambda a: policy[a])
            adapter.apply_action(action)
            mcts.advance(action)
            n_actions_taken += 1

        wall_time = time.perf_counter() - t0

        # Extract the final solution vector from the game state.  For
        # ``basis_selection`` the solution is the Galerkin approximation
        # at the collocation points; for ``mesh_refinement`` it's the
        # nodal solution on the adaptive mesh.  Older state snapshots
        # may not carry a populated solution - fall back to an empty
        # array and flag it via metadata so downstream consumers can
        # skip solution-dependent plots safely.
        final_state = adapter.state
        solution_arr = np.asarray(final_state.solution, dtype=np.float64)
        grid_arr = np.asarray(final_state.coords, dtype=np.float64)
        solution_available = solution_arr.size > 0

        # Prefer the game's exact-error computation which understands
        # the game's internal representation (ref: compute_exact_error);
        # fall back to the generic helper used by classical baselines.
        errors = pde_game.compute_exact_error(final_state)
        l2_error = errors.get("l2")
        if l2_error is None and solution_available:
            l2_error = self._compute_l2_error(solution_arr, grid_arr, operator)

        # ``final_state.dof`` is the authoritative DOF count (basis functions
        # selected for basis_selection, mesh nodes for mesh_refinement).
        # ``len(solution_arr)`` is the collocation-point count and would be
        # wrong for basis_selection, so do not use it as a fallback.
        n_dof_final = int(final_state.dof)

        log.info(
            "alphagalerkin_solve_done",
            wall_time=wall_time,
            l2_error=l2_error,
            n_dof=n_dof_final,
            n_actions_taken=n_actions_taken,
        )

        return SolverResult(
            solution=solution_arr,
            grid_points=grid_arr,
            n_dof=n_dof_final,
            wall_time_seconds=wall_time,
            l2_error=float(l2_error) if l2_error is not None else None,
            metadata={
                METADATA_KEY_SOLVER: self.name,
                # Determinism-affecting fields - surfaced explicitly so
                # PDEBenchmarkRunner.export_csv() can populate the seed
                # column and downstream analysis can tell two runs apart
                # without inspecting the solver config out-of-band.
                METADATA_KEY_SEED: self.config.seed,
                METADATA_KEY_GAME_MODE: self.config.game_mode,
                METADATA_KEY_N_MCTS_SIMULATIONS: self.config.n_mcts_simulations,
                METADATA_KEY_MAX_STEPS: self.config.max_steps,
                METADATA_KEY_TARGET_TOLERANCE: self.config.target_tolerance,
                METADATA_KEY_EVALUATOR: self.config.evaluator,
                METADATA_KEY_MIN_GAME_DOF: self.config.min_game_dof,
                # Run outcome
                METADATA_KEY_N_ACTIONS_TAKEN: n_actions_taken,
                METADATA_KEY_ERROR_HISTORY: list(adapter.error_history),
                METADATA_KEY_SOLUTION_AVAILABLE: solution_available,
                METADATA_KEY_TERMINATION_REASON: termination_reason,
            },
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed numpy / random / torch deterministically.

        MCTS child-selection includes a small amount of randomness via
        Dirichlet noise (disabled) and policy ties; seeding keeps two
        back-to-back ``solve()`` calls bit-identical.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _derive_pde_config(operator: PDEOperator, n_dof: int) -> PDEConfig:
        """Build a ``PDEConfig`` from an operator's runtime metadata.

        The operator owns its own ``PDEConfig`` (see ``PDEOperator.__init__``).
        We return that config directly so the PDE game sees exactly the
        same domain and coefficients the classical baselines were given.

        ``n_dof`` is a hint only: it is not written back into the PDE
        config here because classical baselines similarly treat it as
        a per-call argument, not as part of the operator definition.
        """
        del n_dof  # accepted for symmetry with BaseSolver.solve()
        cfg = getattr(operator, "config", None)
        if cfg is None:
            raise ValueError(
                "AlphaGalerkinSolver requires operator.config to be a PDEConfig; "
                f"operator {type(operator).__name__} exposes no .config attribute."
            )
        if not isinstance(cfg, PDEConfig):
            raise TypeError(
                "AlphaGalerkinSolver expected operator.config to be a PDEConfig, "
                f"got {type(cfg).__name__!r}."
            )
        return cfg

    def _build_game(
        self,
        operator: PDEOperator,
        game_config: PDEGameConfig,
    ) -> PDEGame:
        """Instantiate the appropriate ``PDEGame`` subclass."""
        if self.config.game_mode == "basis_selection":
            return BasisSelectionGame(operator, game_config)
        if self.config.game_mode == "mesh_refinement":
            return MeshRefinementGame(operator, game_config)
        # Pydantic validation guarantees this is unreachable, but mypy
        # cannot prove that from the Literal type alone.
        raise ValueError(f"Unsupported game_mode: {self.config.game_mode!r}")

    def _build_mcts(self, pde_game: PDEGame) -> MCTS:
        """Construct the MCTS engine with the configured evaluator.

        ``random`` and ``uniform`` both map to the ``RandomEvaluator``
        (uniform prior + zero value). A trained, network-backed
        evaluator is not yet part of the typed Literal - when it is
        added the config schema will gain a ``"trained"`` option.
        """
        # The current ``RandomEvaluator`` samples via numpy and has no
        # device-dependent state, so ``config.device`` is a no-op for it.
        # We resolve and log the device anyway so the solver's run
        # provenance is complete and the value is validated against
        # ``torch.device`` at config construction (see ``_validate_device``).
        torch_device = torch.device(self.config.device)
        logger.debug(
            "alphagalerkin_mcts_built",
            solver=self.name,
            evaluator=self.config.evaluator,
            device=str(torch_device),
            n_actions=pde_game.action_space_size,
            n_simulations=self.config.n_mcts_simulations,
        )
        evaluator = RandomEvaluator(n_actions=pde_game.action_space_size)
        return MCTS(
            evaluator=evaluator,
            n_simulations=self.config.n_mcts_simulations,
        )

    # ------------------------------------------------------------------ #
    # Registry hook                                                       #
    # ------------------------------------------------------------------ #


def _register_with_baselines() -> None:
    """Register :class:`AlphaGalerkinSolver` in the baselines registry.

    ``src.research.baselines`` owns ``SOLVER_REGISTRY`` as a plain dict,
    so we can insert ourselves without importing/modifying that module's
    code.  The function is idempotent and safe to call multiple times.
    """
    from src.research import baselines

    baselines.SOLVER_REGISTRY.setdefault("alphagalerkin", AlphaGalerkinSolver)


# Eagerly register on import so that ``get_solver("alphagalerkin")``
# works once the user has ``import src.alphagalerkin`` anywhere in the
# call graph (e.g. via ``PDEBenchmarkRunner``).
_register_with_baselines()
