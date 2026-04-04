"""Tests for shared constants module."""

from __future__ import annotations

import src.constants as C


class TestBoardDefaults:
    """Verify board/game default constants."""

    def test_default_board_sizes_are_standard(self) -> None:
        assert C.DEFAULT_BOARD_SIZES == [9, 13, 19]

    def test_default_board_size_is_full(self) -> None:
        assert C.DEFAULT_BOARD_SIZE == 19

    def test_default_max_moves_positive(self) -> None:
        assert C.DEFAULT_MAX_MOVES > 0


class TestMCTSDefaults:
    """Verify MCTS default constants."""

    def test_simulations_positive(self) -> None:
        assert C.DEFAULT_MCTS_SIMULATIONS > 0

    def test_puct_positive(self) -> None:
        assert C.DEFAULT_PUCT_CONSTANT > 0

    def test_dirichlet_alpha_in_range(self) -> None:
        assert 0 < C.DEFAULT_DIRICHLET_ALPHA < 1

    def test_dirichlet_epsilon_in_range(self) -> None:
        assert 0 < C.DEFAULT_DIRICHLET_EPSILON < 1

    def test_virtual_loss_positive(self) -> None:
        assert C.DEFAULT_VIRTUAL_LOSS > 0


class TestTrainingDefaults:
    """Verify training default constants."""

    def test_temperature_schedule_starts_at_zero(self) -> None:
        assert 0 in C.DEFAULT_TEMPERATURE_SCHEDULE
        assert C.DEFAULT_TEMPERATURE_SCHEDULE[0] == 1.0

    def test_curriculum_schedule_starts_at_zero(self) -> None:
        assert 0 in C.DEFAULT_CURRICULUM_SCHEDULE
        assert C.DEFAULT_CURRICULUM_SCHEDULE[0] == [9]

    def test_dropout_in_unit_interval(self) -> None:
        assert 0 < C.DEFAULT_DROPOUT < 1


class TestPERDefaults:
    """Verify PER constants."""

    def test_per_alpha_in_range(self) -> None:
        assert 0 < C.DEFAULT_PER_ALPHA <= 1

    def test_per_beta_in_range(self) -> None:
        assert 0 < C.DEFAULT_PER_BETA <= 1

    def test_per_beta_increment_positive(self) -> None:
        assert C.DEFAULT_PER_BETA_INCREMENT > 0


class TestLBBDefaults:
    """Verify LBB stability constants."""

    def test_lbb_weight_positive(self) -> None:
        assert C.DEFAULT_LBB_WEIGHT > 0

    def test_lbb_threshold_small_positive(self) -> None:
        assert 0 < C.DEFAULT_LBB_THRESHOLD < 1e-3

    def test_lbb_target_positive(self) -> None:
        assert C.DEFAULT_LBB_TARGET > 0

    def test_lbb_eps_small_positive(self) -> None:
        assert 0 < C.DEFAULT_LBB_EPS < 1e-5


class TestWinRateThresholds:
    """Verify win rate thresholds are sensible."""

    def test_accept_greater_than_reject(self) -> None:
        assert C.WIN_RATE_ACCEPT_THRESHOLD > C.WIN_RATE_REJECT_THRESHOLD

    def test_thresholds_in_unit_interval(self) -> None:
        assert 0 < C.WIN_RATE_REJECT_THRESHOLD < 1
        assert 0 < C.WIN_RATE_ACCEPT_THRESHOLD < 1


class TestNumericConstants:
    """Verify numeric stability constants."""

    def test_layer_norm_epsilon_positive(self) -> None:
        assert C.LAYER_NORM_EPSILON > 0

    def test_attention_epsilon_positive(self) -> None:
        assert C.ATTENTION_EPSILON > 0

    def test_numeric_epsilon_positive(self) -> None:
        assert C.NUMERIC_EPSILON > 0

    def test_boundary_tolerance_positive(self) -> None:
        assert C.DEFAULT_BOUNDARY_TOLERANCE > 0


class TestCheckpointNames:
    """Verify checkpoint naming constants."""

    def test_best_checkpoint_has_extension(self) -> None:
        assert C.CHECKPOINT_BEST.endswith(".pt")

    def test_final_checkpoint_has_extension(self) -> None:
        assert C.CHECKPOINT_FINAL.endswith(".pt")
