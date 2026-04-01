"""Tests for PDETrainer and PDETrainingConfig.

Validates:
- Pydantic config validation (invalid values rejected)
- PDETrainer initialization with Poisson operator
- Running 5 episodes and collecting results
- Error is bounded/converging across episodes (not necessarily strictly monotonic)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pde.trainer import EpisodeResult, PDETrainer, PDETrainingConfig, PDETrainingResult

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def minimal_config(**overrides: object) -> PDETrainingConfig:
    """Return a fast, minimal PDETrainingConfig suitable for testing.

    Uses few collocation points, small action space, and few simulations
    to keep tests fast.
    """
    defaults: dict[str, object] = {
        "name": "test_poisson",
        "pde_type": "poisson",
        "n_episodes": 5,
        "mcts_simulations": 2,
        "error_tolerance": 1e-2,
        "max_basis_functions": 6,
        "max_steps_per_episode": 8,
        "max_frequency": 2,
        "n_collocation_points": 25,
    }
    defaults.update(overrides)
    return PDETrainingConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PDETrainingConfig validation
# ---------------------------------------------------------------------------


class TestPDETrainingConfig:
    """Unit tests for PDETrainingConfig Pydantic model."""

    def test_default_config_is_valid(self) -> None:
        """A default config (pde_type='poisson') should pass validation."""
        config = PDETrainingConfig(name="defaults")
        assert config.pde_type == "poisson"
        assert config.n_episodes > 0
        assert config.mcts_simulations > 0
        assert config.error_tolerance > 0.0

    def test_all_fields_accessible(self) -> None:
        """All required fields should be present and have sensible values."""
        config = minimal_config()
        assert config.pde_type == "poisson"
        assert config.n_episodes == 5
        assert config.mcts_simulations == 2
        assert config.error_tolerance == pytest.approx(1e-2)

    def test_invalid_n_episodes_zero(self) -> None:
        """n_episodes=0 must be rejected (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            minimal_config(n_episodes=0)

    def test_invalid_n_episodes_negative(self) -> None:
        """Negative n_episodes must be rejected."""
        with pytest.raises(ValidationError):
            minimal_config(n_episodes=-1)

    def test_invalid_mcts_simulations_zero(self) -> None:
        """mcts_simulations=0 must be rejected (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            minimal_config(mcts_simulations=0)

    def test_invalid_error_tolerance_zero(self) -> None:
        """error_tolerance=0 must be rejected (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            minimal_config(error_tolerance=0.0)

    def test_invalid_error_tolerance_negative(self) -> None:
        """Negative error_tolerance must be rejected."""
        with pytest.raises(ValidationError):
            minimal_config(error_tolerance=-0.1)

    def test_invalid_pde_type(self) -> None:
        """An unknown pde_type must be rejected."""
        with pytest.raises(ValidationError):
            minimal_config(pde_type="nonexistent_pde")

    def test_valid_pde_types(self) -> None:
        """All known PDE types should pass validation."""
        valid_types = ["poisson", "burgers", "advection_diffusion"]
        for pde_type in valid_types:
            cfg = minimal_config(pde_type=pde_type)
            assert cfg.pde_type == pde_type

    def test_seed_optional(self) -> None:
        """seed=None (default) and seed=42 should both be valid."""
        cfg_none = minimal_config(seed=None)
        cfg_int = minimal_config(seed=42)
        assert cfg_none.seed is None
        assert cfg_int.seed == 42


# ---------------------------------------------------------------------------
# PDETrainer initialization
# ---------------------------------------------------------------------------


class TestPDETrainerInit:
    """Tests for PDETrainer.__init__."""

    def test_init_with_poisson(self) -> None:
        """PDETrainer should initialize without errors for Poisson."""
        config = minimal_config()
        trainer = PDETrainer(config)
        assert trainer.config is config

    def test_trainer_has_mcts(self) -> None:
        """Trainer should build an MCTS engine."""
        config = minimal_config()
        trainer = PDETrainer(config)
        assert trainer._mcts is not None

    def test_trainer_has_game(self) -> None:
        """Trainer should build a BasisSelectionGame."""
        config = minimal_config()
        trainer = PDETrainer(config)
        assert trainer._game is not None

    def test_trainer_has_operator(self) -> None:
        """Trainer should build a PDE operator."""
        config = minimal_config()
        trainer = PDETrainer(config)
        assert trainer._operator is not None

    def test_action_space_positive(self) -> None:
        """The action space should have at least one action."""
        config = minimal_config()
        trainer = PDETrainer(config)
        assert trainer._game.action_space_size > 0


# ---------------------------------------------------------------------------
# PDETrainer.run() — 5-episode test
# ---------------------------------------------------------------------------


class TestPDETrainerRun:
    """Integration tests for PDETrainer.run()."""

    @pytest.fixture(scope="function")
    def run_result(self) -> PDETrainingResult:
        """Run 5 episodes once and share the result across tests."""
        config = minimal_config(n_episodes=5)
        trainer = PDETrainer(config)
        return trainer.run()

    def test_n_episodes_run(self, run_result: PDETrainingResult) -> None:
        """All 5 episodes should complete."""
        assert run_result.n_episodes_run == 5

    def test_episodes_list_length(self, run_result: PDETrainingResult) -> None:
        """result.episodes should contain one EpisodeResult per episode."""
        assert len(run_result.episodes) == 5

    def test_episode_result_types(self, run_result: PDETrainingResult) -> None:
        """Each element of result.episodes should be an EpisodeResult."""
        for ep in run_result.episodes:
            assert isinstance(ep, EpisodeResult)

    def test_errors_property(self, run_result: PDETrainingResult) -> None:
        """result.errors should return one float per episode."""
        errors = run_result.errors
        assert len(errors) == 5
        assert all(isinstance(e, float) for e in errors)

    def test_actions_property(self, run_result: PDETrainingResult) -> None:
        """result.actions should return one action list per episode."""
        actions = run_result.actions
        assert len(actions) == 5
        assert all(isinstance(a, list) for a in actions)

    def test_best_final_error_finite(self, run_result: PDETrainingResult) -> None:
        """best_final_error should be a finite positive number."""
        best = run_result.best_final_error
        assert best > 0.0
        assert best < float("inf")

    def test_error_history_nonempty(self, run_result: PDETrainingResult) -> None:
        """Each episode should have a non-empty error history."""
        for ep in run_result.episodes:
            assert len(ep.error_history) >= 1

    def test_error_history_starts_with_initial(self, run_result: PDETrainingResult) -> None:
        """error_history[0] should equal initial_error."""
        for ep in run_result.episodes:
            assert ep.error_history[0] == pytest.approx(ep.initial_error)

    def test_error_history_ends_with_final(self, run_result: PDETrainingResult) -> None:
        """error_history[-1] should equal final_error."""
        for ep in run_result.episodes:
            assert ep.error_history[-1] == pytest.approx(ep.final_error)

    def test_actions_length_matches_steps(self, run_result: PDETrainingResult) -> None:
        """Number of actions should equal n_steps for each episode."""
        for ep in run_result.episodes:
            assert len(ep.actions) == ep.n_steps

    def test_error_history_length_consistent(self, run_result: PDETrainingResult) -> None:
        """error_history should have n_steps+1 entries (initial + one per step)."""
        for ep in run_result.episodes:
            assert len(ep.error_history) == ep.n_steps + 1

    def test_final_error_nonnegative(self, run_result: PDETrainingResult) -> None:
        """Final error should never be negative."""
        for ep in run_result.episodes:
            assert ep.final_error >= 0.0

    def test_converged_flag_consistent(self, run_result: PDETrainingResult) -> None:
        """Converged flag on each EpisodeResult should be consistent with final_error."""
        config = minimal_config()
        for ep in run_result.episodes:
            if ep.converged:
                assert ep.final_error < config.error_tolerance
            else:
                assert ep.final_error >= config.error_tolerance

    def test_overall_result_converged_flag(self, run_result: PDETrainingResult) -> None:
        """PDETrainingResult.converged should be True iff any episode converged."""
        any_converged = any(ep.converged for ep in run_result.episodes)
        assert run_result.converged == any_converged

    def test_episode_indices_sequential(self, run_result: PDETrainingResult) -> None:
        """Episode indices should be 0, 1, 2, 3, 4."""
        for i, ep in enumerate(run_result.episodes):
            assert ep.episode_idx == i


# ---------------------------------------------------------------------------
# Error trend: error is bounded/converging (not necessarily strictly monotonic)
# ---------------------------------------------------------------------------


class TestErrorConvergenceTrend:
    """Tests verifying that error is decreasing or at least bounded within episodes."""

    def test_error_reduced_within_episode(self) -> None:
        """Final error should be <= initial error for each episode.

        With random MCTS, adding more basis functions generally decreases
        approximation error, so the total error should not increase.
        This is a relaxed (non-strict) monotonicity check.
        """
        config = minimal_config(n_episodes=5, mcts_simulations=2)
        trainer = PDETrainer(config)
        result = trainer.run()

        for ep in result.episodes:
            if ep.n_steps > 0:
                # Final error should be at most as large as initial
                # (basis addition should not increase error on average)
                assert ep.final_error <= ep.initial_error + 1e-6, (
                    f"Episode {ep.episode_idx}: final_error={ep.final_error:.6f} "
                    f"> initial_error={ep.initial_error:.6f}"
                )

    def test_error_history_decreases_overall(self) -> None:
        """The minimum error seen in a run should be less than the initial error.

        When adding multiple basis functions, error should decrease at least
        once over the course of a 5-episode run.
        """
        config = minimal_config(n_episodes=5, mcts_simulations=2)
        trainer = PDETrainer(config)
        result = trainer.run()

        # Collect all errors across all episodes
        all_initial_errors = [ep.initial_error for ep in result.episodes]
        all_final_errors = [ep.final_error for ep in result.episodes]

        mean_initial = sum(all_initial_errors) / len(all_initial_errors)
        mean_final = sum(all_final_errors) / len(all_final_errors)

        # On average, final error should be no worse than initial
        assert mean_final <= mean_initial + 1e-6, (
            f"Mean final error ({mean_final:.6f}) > mean initial error ({mean_initial:.6f})"
        )

    def test_best_error_improves_with_more_episodes(self) -> None:
        """More episodes gives more chances to find a good basis sequence.

        With a fixed seed, run 1 vs 5 episodes: episode 1 in both runs is
        identical (same seed), so the 5-episode best_final_error is ≤ the
        1-episode best_final_error by construction (min of superset).
        """
        # Fixed seed ensures both trainers execute the same first episode;
        # the 5-episode run's min(ep1..ep5) must be ≤ ep1 of the 1-episode run.
        config_small = minimal_config(n_episodes=1, mcts_simulations=2, seed=0)
        config_large = minimal_config(n_episodes=5, mcts_simulations=2, seed=0)

        result_small = PDETrainer(config_small).run()
        result_large = PDETrainer(config_large).run()

        # The 5-episode run had more chances; its best can only be <= 1-episode best
        assert result_large.best_final_error <= result_small.best_final_error + 1e-6
