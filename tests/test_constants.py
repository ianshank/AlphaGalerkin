"""Tests for shared constants module."""

from __future__ import annotations

import src.constants as C


class TestBoardDefaults:
    """Verify board/game default constants."""

    def test_default_board_sizes_are_standard(self) -> None:
        assert C.DEFAULT_BOARD_SIZES == [9, 13, 19]

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


class TestPicoGKBoundaryTolerance:
    """Verify the SDF boundary-classification band constant."""

    def test_picogk_tolerance_positive(self) -> None:
        assert C.DEFAULT_PICOGK_BOUNDARY_TOLERANCE > 0

    def test_picogk_tolerance_is_looser_than_the_analytic_band(self) -> None:
        """Documented relationship: the SDF band is deliberately looser."""
        assert C.DEFAULT_PICOGK_BOUNDARY_TOLERANCE > C.DEFAULT_BOUNDARY_TOLERANCE

    def test_two_tolerances_are_distinguishable(self) -> None:
        """They differ, so mis-binding a call site is observable."""
        assert C.DEFAULT_PICOGK_BOUNDARY_TOLERANCE != C.DEFAULT_BOUNDARY_TOLERANCE

    def test_helical_operators_default_to_the_picogk_band(self) -> None:
        """All three SDF operators bind is_boundary_point's default to it."""
        import inspect

        from src.pde.operators_picogk import (
            HelicalHeatOperator,
            HelicalMagnetostaticsOperator,
            HelicalStokesOperator,
        )

        for operator_cls in (
            HelicalHeatOperator,
            HelicalStokesOperator,
            HelicalMagnetostaticsOperator,
        ):
            default = (
                inspect.signature(operator_cls.is_boundary_point).parameters["tolerance"].default
            )
            assert default == C.DEFAULT_PICOGK_BOUNDARY_TOLERANCE, operator_cls.__name__

    def test_analytic_operators_default_to_the_analytic_band(self) -> None:
        """The analytic operators keep the tighter coordinate-compare band."""
        import inspect

        from src.pde.operators import LShapedPoissonOperator, PDEOperator

        for operator_cls in (PDEOperator, LShapedPoissonOperator):
            default = (
                inspect.signature(operator_cls.is_boundary_point).parameters["tolerance"].default
            )
            assert default == C.DEFAULT_BOUNDARY_TOLERANCE, operator_cls.__name__

    def test_projection_tolerance_is_a_separate_symbol(self) -> None:
        """Numerically equal but semantically distinct -- must stay unmerged."""
        from src.pde.geometry_picogk import DEFAULT_BOUNDARY_PROJECTION_TOL

        assert "DEFAULT_BOUNDARY_PROJECTION_TOL" not in vars(C)
        assert DEFAULT_BOUNDARY_PROJECTION_TOL == C.DEFAULT_PICOGK_BOUNDARY_TOLERANCE


class TestBoardSizesCopySemantics:
    """DEFAULT_BOARD_SIZES is a mutable module-level list.

    Every consumer that adopted the constant must hand out a *copy*: a shared
    reference would let one config's ``.append()`` silently retune every other
    config in the process, and permanently corrupt the constant itself. These
    tests fail on any site that leaks the shared list.
    """

    EXPECTED = [9, 13, 19]

    def test_constant_value(self) -> None:
        assert C.DEFAULT_BOARD_SIZES == self.EXPECTED

    @staticmethod
    def _config_board_sizes() -> dict[str, list[int]]:
        """Field values from every config that defaults to the constant."""
        from src.demos.config import PhysicsDemoConfig
        from src.poc.config import TransferScenarioConfig
        from src.research.config import BenchmarkConfig, ComparisonConfig, TransferConfig
        from src.templates.config import BoardSizeConfig

        return {
            "PhysicsDemoConfig.eval_grid_sizes": PhysicsDemoConfig().eval_grid_sizes,
            "TransferScenarioConfig.eval_resolutions": TransferScenarioConfig(
                name="t", description="d"
            ).eval_resolutions,
            "BenchmarkConfig.sizes": BenchmarkConfig(name="b").sizes,
            "TransferConfig.target_sizes": TransferConfig().target_sizes,
            "ComparisonConfig.eval_sizes": ComparisonConfig().eval_sizes,
            "BoardSizeConfig.sizes": BoardSizeConfig().sizes,
        }

    def test_all_config_defaults_equal_the_constant(self) -> None:
        for label, sizes in self._config_board_sizes().items():
            assert sizes == self.EXPECTED, label

    def test_no_config_default_aliases_the_constant(self) -> None:
        """Each default_factory must produce a fresh list object."""
        for label, sizes in self._config_board_sizes().items():
            assert sizes is not C.DEFAULT_BOARD_SIZES, label

    def test_mutating_one_instance_does_not_affect_another(self) -> None:
        """Two independently-constructed configs hold independent lists."""
        from src.research.config import BenchmarkConfig

        first = BenchmarkConfig(name="a")
        second = BenchmarkConfig(name="b")
        first.sizes.append(25)

        assert second.sizes == self.EXPECTED
        assert C.DEFAULT_BOARD_SIZES == self.EXPECTED

    def test_mutating_a_config_does_not_corrupt_the_constant(self) -> None:
        """The module constant survives mutation of every adopting config."""
        for sizes in self._config_board_sizes().values():
            sizes.append(99)
        assert C.DEFAULT_BOARD_SIZES == self.EXPECTED

    def test_factory_functions_return_fresh_lists(self) -> None:
        """`sizes or list(DEFAULT_BOARD_SIZES)` fallbacks must not alias either."""
        from src.research.benchmark import create_benchmark
        from src.research.comparison import create_comparison
        from src.research.config import create_transfer_config
        from src.research.validator import create_transfer_validator

        produced = {
            "create_benchmark": create_benchmark().config.sizes,
            "create_comparison": create_comparison().config.eval_sizes,
            "create_transfer_config": create_transfer_config().target_sizes,
            "create_transfer_validator": create_transfer_validator().config.target_sizes,
        }
        for label, sizes in produced.items():
            assert sizes == self.EXPECTED, label
            assert sizes is not C.DEFAULT_BOARD_SIZES, label

        for sizes in produced.values():
            sizes.append(99)
        assert C.DEFAULT_BOARD_SIZES == self.EXPECTED

    def test_default_curriculum_uses_the_constant_without_aliasing(self) -> None:
        """create_default_curriculum(board_sizes=None) builds stages from a copy."""
        from src.curriculum.config import create_default_curriculum

        curriculum = create_default_curriculum()
        assert [stage.board_size for stage in curriculum.stages] == self.EXPECTED
        assert C.DEFAULT_BOARD_SIZES == self.EXPECTED

    def test_verify_transfer_module_default_is_a_copy(self) -> None:
        """The module-level DEFAULT_EVAL_SIZES alias must not be the constant."""
        from src.experiments.verify_transfer import DEFAULT_EVAL_SIZES

        assert DEFAULT_EVAL_SIZES == self.EXPECTED
        assert DEFAULT_EVAL_SIZES is not C.DEFAULT_BOARD_SIZES

    def test_alphagalerkin_config_board_sizes_copy_the_constant(self) -> None:
        """``AlphaGalerkinConfig.board_sizes`` derives from the constant, by copy.

        This site sits in ``config/`` rather than ``src/`` and was missed by the
        original 13-site migration, so it could drift from the constant
        silently. It now copies like every other site: equal in value, never the
        same object, and independently mutable per instance.
        """
        from config.schemas import AlphaGalerkinConfig

        first = AlphaGalerkinConfig()
        assert first.board_sizes == C.DEFAULT_BOARD_SIZES
        assert first.board_sizes is not C.DEFAULT_BOARD_SIZES

        first.board_sizes.append(21)
        assert AlphaGalerkinConfig().board_sizes == C.DEFAULT_BOARD_SIZES
        assert C.DEFAULT_BOARD_SIZES == self.EXPECTED
