"""PDE Training Loop orchestrating MCTS-guided Galerkin basis selection.

This module provides the ``PDETrainer`` class that runs multi-episode
training loops using:

- ``BasisSelectionGame`` for the PDE game environment
- ``PDEGameAdapter`` to bridge PDE games to the MCTS interface
- ``MCTS`` with ``RandomEvaluator`` for action selection
- Structured logging via structlog for observability

Usage::

    from src.pde.trainer import PDETrainer, PDETrainingConfig

    config = PDETrainingConfig(
        name="poisson_training",
        pde_type="poisson",
        n_episodes=10,
        mcts_simulations=50,
        error_tolerance=1e-3,
    )
    trainer = PDETrainer(config)
    result = trainer.run()
    print(f"Converged: {result.converged}, episodes: {result.n_episodes_run}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import structlog
from pydantic import Field, model_validator

from src.pde.config import (
    BasisSelectionConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
)
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.operators import PoissonOperator
from src.templates.config import BaseModuleConfig

if TYPE_CHECKING:
    from src.mcts.evaluator import RandomEvaluator
    from src.mcts.search import MCTS
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)

# PDE types actually supported by _create_operator
SUPPORTED_PDE_TYPES: frozenset[PDEType] = frozenset({
    PDEType.POISSON,
    PDEType.BURGERS,
    PDEType.ADVECTION_DIFFUSION,
})


def _create_operator(pde_config: PDEConfig) -> PDEOperator:
    """Instantiate a PDE operator from a PDEConfig.

    Args:
        pde_config: PDE configuration specifying equation type.

    Returns:
        Concrete ``PDEOperator`` instance.

    Raises:
        ValueError: If the PDE type is not supported.

    """
    pde_type = PDEType(pde_config.pde_type)
    if pde_type == PDEType.POISSON:
        return PoissonOperator(pde_config)

    # Lazy imports for less-common operators to avoid top-level cost
    from src.pde.operators import AdvectionDiffusionOperator, BurgersOperator  # noqa: PLC0415

    if pde_type == PDEType.BURGERS:
        return BurgersOperator(pde_config)
    if pde_type == PDEType.ADVECTION_DIFFUSION:
        return AdvectionDiffusionOperator(pde_config)

    raise ValueError(
        f"Unsupported pde_type: '{pde_config.pde_type}'. "
        f"Supported types: {[t.value for t in SUPPORTED_PDE_TYPES]}"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PDETrainingConfig(BaseModuleConfig):
    """Configuration for the PDE training loop.

    Attributes:
        pde_type: PDE equation type (e.g. "poisson", "burgers").
        n_episodes: Number of training episodes to run.
        mcts_simulations: MCTS simulations per search step.
        error_tolerance: Target error below which an episode is converged.
        max_basis_functions: Maximum basis functions per episode.
        max_steps_per_episode: Maximum game steps per episode.
        basis_type: Basis function family ("fourier", "polynomial", "rbf").
        max_frequency: Maximum Fourier frequency (for fourier basis).
        n_collocation_points: Collocation points for residual evaluation.
        domain_dim: Spatial dimension of the PDE domain.
        computational_budget: Per-episode computational budget (FLOPs).
        seed: Optional RNG seed for reproducibility (None = random).

    """

    pde_type: str = Field(
        default="poisson",
        description="PDE equation type (e.g. 'poisson', 'burgers')",
    )
    n_episodes: int = Field(
        default=10,
        gt=0,
        description="Number of training episodes",
    )
    mcts_simulations: int = Field(
        default=50,
        gt=0,
        description="Number of MCTS simulations per search step",
    )
    error_tolerance: float = Field(
        default=1e-3,
        gt=0.0,
        description="Target error tolerance for convergence",
    )
    max_basis_functions: int = Field(
        default=20,
        ge=1,
        description="Maximum number of basis functions per episode",
    )
    max_steps_per_episode: int = Field(
        default=50,
        ge=1,
        description="Maximum game steps per episode",
    )
    basis_type: str = Field(
        default="fourier",
        description="Basis function type ('fourier', 'polynomial', 'rbf')",
    )
    max_frequency: int = Field(
        default=5,
        ge=1,
        description="Maximum Fourier frequency for Fourier basis",
    )
    n_collocation_points: int = Field(
        default=100,
        ge=10,
        description="Number of collocation points for residual evaluation",
    )
    domain_dim: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Spatial dimension of the PDE domain",
    )
    computational_budget: float = Field(
        default=1e5,
        gt=0.0,
        description="Per-episode computational budget (FLOPs)",
    )
    seed: int | None = Field(  # type: ignore[assignment]
        default=None,
        description="RNG seed for reproducibility (None = random)",
    )

    @model_validator(mode="after")
    def validate_pde_type(self) -> PDETrainingConfig:
        """Validate that the pde_type is one of the supported PDE types.

        Only types actually handled by ``_create_operator`` are accepted so
        that validation failure is caught at config creation time rather than
        at runtime inside the training loop.
        """
        try:
            pde_type = PDEType(self.pde_type)
        except ValueError:
            supported = [t.value for t in SUPPORTED_PDE_TYPES]
            raise ValueError(
                f"Unknown pde_type: '{self.pde_type}'. "
                f"Supported options: {supported}"
            ) from None
        if pde_type not in SUPPORTED_PDE_TYPES:
            supported = [t.value for t in SUPPORTED_PDE_TYPES]
            raise ValueError(
                f"Unsupported pde_type: '{self.pde_type}'. "
                f"Supported options: {supported}"
            )
        return self


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Result from a single training episode.

    Attributes:
        episode_idx: Zero-based episode index.
        error_history: Error after each game step (includes initial).
        actions: Action sequence taken during the episode.
        converged: Whether error fell below tolerance.
        n_steps: Number of game steps taken.
        initial_error: Error at the start of the episode.
        final_error: Error at the end of the episode.

    """

    episode_idx: int
    error_history: list[float]
    actions: list[int]
    converged: bool
    n_steps: int
    initial_error: float
    final_error: float


@dataclass
class PDETrainingResult:
    """Aggregated result from a full training run.

    Attributes:
        episodes: Per-episode results.
        converged: True if any episode converged.
        n_episodes_run: Total episodes completed.
        errors: Final error per episode (convenience accessor).
        actions: Action sequences per episode (convenience accessor).
        best_final_error: Lowest final error observed across episodes.

    """

    episodes: list[EpisodeResult] = field(default_factory=list)
    converged: bool = False
    n_episodes_run: int = 0

    @property
    def errors(self) -> list[float]:
        """Final error for each episode."""
        return [ep.final_error for ep in self.episodes]

    @property
    def actions(self) -> list[list[int]]:
        """Action sequence for each episode."""
        return [ep.actions for ep in self.episodes]

    @property
    def best_final_error(self) -> float:
        """Minimum final error across all episodes."""
        if not self.episodes:
            return float("inf")
        return min(ep.final_error for ep in self.episodes)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PDETrainer:
    """Orchestrates PDE training via MCTS-guided basis selection.

    Each episode:
    1. Resets a fresh ``PDEGameAdapter`` (Poisson or other PDE).
    2. Runs MCTS ``search()`` to get an action-probability map.
    3. Selects the highest-probability legal action.
    4. Steps the game with ``apply_action()``.
    5. Tracks error history and stops if tolerance is met or game ends.

    Logs structured events via ``structlog`` at each episode and step.

    Attributes:
        config: Training configuration.

    """

    def __init__(self, config: PDETrainingConfig) -> None:
        """Initialize the PDE trainer.

        Builds the PDE operator, game, and MCTS engine from ``config``.

        Args:
            config: ``PDETrainingConfig`` specifying all hyperparameters.

        """
        self.config = config

        # Build PDE config
        domain_min = [0.0] * config.domain_dim
        domain_max = [1.0] * config.domain_dim
        advection_coeff = [0.0] * config.domain_dim

        self._pde_config = PDEConfig(
            name=f"{config.name}_pde",
            pde_type=PDEType(config.pde_type),
            domain_dim=config.domain_dim,
            domain_min=domain_min,
            domain_max=domain_max,
            advection_coeff=advection_coeff,
        )

        # Build game config
        _basis_type = cast(
            Literal["fourier", "polynomial", "rbf", "wavelet"],
            config.basis_type,
        )
        self._basis_config = BasisSelectionConfig(
            name=f"{config.name}_basis",
            basis_type=_basis_type,
            max_basis_functions=config.max_basis_functions,
            max_frequency=config.max_frequency,
            n_collocation_points=config.n_collocation_points,
            seed=config.seed,
        )
        self._game_config = PDEGameConfig(
            name=f"{config.name}_game",
            pde_config=self._pde_config,
            game_mode="basis_selection",
            basis_config=self._basis_config,
            error_tolerance=config.error_tolerance,
            max_steps=config.max_steps_per_episode,
            computational_budget=config.computational_budget,
            seed=config.seed,
        )

        # Instantiate PDE operator and game (shared; game is stateless)
        self._operator: PDEOperator = _create_operator(self._pde_config)
        self._game = BasisSelectionGame(self._operator, self._game_config)

        # Build MCTS with random evaluator (no trained network required)
        # These imports are kept here so heavy PyTorch/numpy loads are deferred
        from src.mcts.evaluator import RandomEvaluator  # noqa: PLC0415
        from src.mcts.search import MCTS  # noqa: PLC0415

        self._evaluator: RandomEvaluator = RandomEvaluator(
            n_actions=self._game.action_space_size,
        )
        self._mcts: MCTS = MCTS(
            evaluator=self._evaluator,
            n_simulations=config.mcts_simulations,
        )

        logger.info(
            "pde_trainer_initialized",
            pde_type=config.pde_type,
            n_episodes=config.n_episodes,
            mcts_simulations=config.mcts_simulations,
            error_tolerance=config.error_tolerance,
            action_space_size=self._game.action_space_size,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> PDETrainingResult:
        """Run the full training loop for ``config.n_episodes`` episodes.

        Returns:
            ``PDETrainingResult`` aggregating per-episode outcomes.

        """
        result = PDETrainingResult()

        for ep_idx in range(self.config.n_episodes):
            ep_result = self._run_episode(ep_idx)
            result.episodes.append(ep_result)
            result.n_episodes_run += 1

            if ep_result.converged:
                result.converged = True

            logger.info(
                "pde_episode_complete",
                episode=ep_idx,
                n_steps=ep_result.n_steps,
                initial_error=ep_result.initial_error,
                final_error=ep_result.final_error,
                converged=ep_result.converged,
                error_history=ep_result.error_history,
                actions=ep_result.actions,
            )

        logger.info(
            "pde_training_complete",
            n_episodes_run=result.n_episodes_run,
            converged=result.converged,
            best_final_error=result.best_final_error,
        )
        return result

    # ------------------------------------------------------------------ #
    # Episode internals                                                   #
    # ------------------------------------------------------------------ #

    def _run_episode(self, episode_idx: int) -> EpisodeResult:
        """Run a single episode.

        Args:
            episode_idx: Zero-based index of this episode.

        Returns:
            ``EpisodeResult`` with error history, actions, and convergence.

        """
        # Fresh adapter and MCTS tree for each episode
        adapter = PDEGameAdapter(self._game)
        self._mcts.reset()

        error_history: list[float] = [adapter.current_error]
        actions: list[int] = []

        logger.debug(
            "pde_episode_start",
            episode=episode_idx,
            initial_error=adapter.current_error,
        )

        while not adapter.is_terminal():
            legal_actions = adapter.get_legal_actions()
            if not legal_actions:
                break

            # MCTS search returns action → visit_probability map
            visit_dist = self._mcts.search(adapter, add_noise=True)

            # Select action with highest visit count
            action = max(visit_dist, key=lambda a: visit_dist[a])

            logger.debug(
                "pde_step",
                episode=episode_idx,
                step=adapter.state.step,
                selected_action=action,
                error_before=adapter.current_error,
            )

            adapter.apply_action(action)
            actions.append(action)
            error_history.append(adapter.current_error)

            # Advance tree to reuse subtree
            self._mcts.advance(action)

            logger.debug(
                "pde_step_complete",
                episode=episode_idx,
                step=adapter.state.step,
                error_after=adapter.current_error,
            )

        initial_error = error_history[0] if error_history else float("inf")
        final_error = error_history[-1] if error_history else float("inf")
        converged = final_error < self.config.error_tolerance

        return EpisodeResult(
            episode_idx=episode_idx,
            error_history=error_history,
            actions=actions,
            converged=converged,
            n_steps=len(actions),
            initial_error=initial_error,
            final_error=final_error,
        )
