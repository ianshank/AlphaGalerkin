"""Tests for the exception hierarchy (core/exceptions.py)."""
from __future__ import annotations

import pytest

from src.alphagalerkin.core.exceptions import (
    AlphaGalerkinError,
    CheckpointError,
    CheckpointVersionMismatchError,
    ConfigError,
    ConfigValidationError,
    DiscretizationError,
    DOFBudgetExceededError,
    InvariantViolationError,
    MCTSError,
    MeshIntegrityError,
    NoValidActionsError,
    PhysicsError,
    SolverDivergenceError,
    StabilityViolationError,
    TrainingError,
    TreeDepthExceededError,
)

# ---------------------------------------------------------------
# AlphaGalerkinError (base)
# ---------------------------------------------------------------


class TestAlphaGalerkinError:
    """Root exception with context."""

    def test_message_stored(self) -> None:
        err = AlphaGalerkinError("something failed")

        assert str(err) == "something failed"

    def test_empty_context_by_default(self) -> None:
        err = AlphaGalerkinError("msg")

        assert err.context == {}

    def test_context_preserved(self) -> None:
        ctx = {"key": "value", "count": 3}
        err = AlphaGalerkinError("msg", context=ctx)

        assert err.context == ctx

    def test_repr_without_context(self) -> None:
        err = AlphaGalerkinError("msg")
        r = repr(err)

        assert "AlphaGalerkinError" in r
        assert "msg" in r

    def test_repr_with_context(self) -> None:
        err = AlphaGalerkinError("msg", context={"a": 1})
        r = repr(err)

        assert "context" in r
        assert "a=1" in r

    def test_is_exception(self) -> None:
        err = AlphaGalerkinError()

        assert isinstance(err, Exception)

    def test_catchable_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise AlphaGalerkinError("test")


# ---------------------------------------------------------------
# ConfigError
# ---------------------------------------------------------------


class TestConfigError:
    """ConfigError with path."""

    def test_inherits_from_base(self) -> None:
        assert issubclass(ConfigError, AlphaGalerkinError)

    def test_path_stored(self) -> None:
        err = ConfigError("bad config", path="/etc/config.yaml")

        assert err.path == "/etc/config.yaml"
        assert err.context["path"] == "/etc/config.yaml"

    def test_path_none_by_default(self) -> None:
        err = ConfigError("no path")

        assert err.path is None
        assert "path" not in err.context

    def test_context_merged(self) -> None:
        err = ConfigError(
            "x", path="/a.yaml", context={"extra": True},
        )

        assert err.context["path"] == "/a.yaml"
        assert err.context["extra"] is True


# ---------------------------------------------------------------
# ConfigValidationError
# ---------------------------------------------------------------


class TestConfigValidationError:
    """ConfigValidationError with field and value."""

    def test_inherits_from_config_error(self) -> None:
        assert issubclass(ConfigValidationError, ConfigError)

    def test_field_and_value(self) -> None:
        err = ConfigValidationError(
            "invalid", field="mcts.c_puct", value=-1.0,
        )

        assert err.field == "mcts.c_puct"
        assert err.value == -1.0
        assert err.context["field"] == "mcts.c_puct"
        assert err.context["value"] == -1.0

    def test_defaults_none(self) -> None:
        err = ConfigValidationError("msg")

        assert err.field is None
        assert err.value is None

    def test_all_params(self) -> None:
        err = ConfigValidationError(
            "bad",
            field="lr",
            value=999,
            path="/cfg.yaml",
            context={"more": "info"},
        )

        assert err.field == "lr"
        assert err.value == 999
        assert err.path == "/cfg.yaml"
        assert err.context["more"] == "info"


# ---------------------------------------------------------------
# MCTS errors
# ---------------------------------------------------------------


class TestTreeDepthExceededError:
    """TreeDepthExceededError with depth info."""

    def test_inherits(self) -> None:
        assert issubclass(TreeDepthExceededError, MCTSError)
        assert issubclass(TreeDepthExceededError, AlphaGalerkinError)

    def test_depth_and_max(self) -> None:
        err = TreeDepthExceededError(
            "too deep", depth=150, max_depth=100,
        )

        assert err.depth == 150
        assert err.max_depth == 100
        assert err.context["depth"] == 150
        assert err.context["max_depth"] == 100

    def test_defaults_none(self) -> None:
        err = TreeDepthExceededError("msg")

        assert err.depth is None
        assert err.max_depth is None


class TestNoValidActionsError:
    """NoValidActionsError with state_id."""

    def test_inherits(self) -> None:
        assert issubclass(NoValidActionsError, MCTSError)

    def test_state_id(self) -> None:
        err = NoValidActionsError("stuck", state_id="s42")

        assert err.state_id == "s42"
        assert err.context["state_id"] == "s42"

    def test_defaults_none(self) -> None:
        err = NoValidActionsError()

        assert err.state_id is None


# ---------------------------------------------------------------
# Discretization errors
# ---------------------------------------------------------------


class TestInvariantViolationError:
    """InvariantViolationError with invariant name."""

    def test_inherits(self) -> None:
        assert issubclass(
            InvariantViolationError, DiscretizationError,
        )

    def test_invariant_field(self) -> None:
        err = InvariantViolationError(
            "broken", invariant="positive_jacobian",
        )

        assert err.invariant == "positive_jacobian"
        assert err.context["invariant"] == "positive_jacobian"

    def test_defaults_none(self) -> None:
        err = InvariantViolationError("msg")

        assert err.invariant is None


class TestDOFBudgetExceededError:
    """DOFBudgetExceededError with current and max DOF."""

    def test_inherits(self) -> None:
        assert issubclass(
            DOFBudgetExceededError, DiscretizationError,
        )

    def test_dof_fields(self) -> None:
        err = DOFBudgetExceededError(
            "over budget",
            current_dof=60000,
            max_dof=50000,
        )

        assert err.current_dof == 60000
        assert err.max_dof == 50000
        assert err.context["current_dof"] == 60000
        assert err.context["max_dof"] == 50000

    def test_defaults_none(self) -> None:
        err = DOFBudgetExceededError()

        assert err.current_dof is None
        assert err.max_dof is None


class TestMeshIntegrityError:
    """MeshIntegrityError with element_id."""

    def test_inherits(self) -> None:
        assert issubclass(MeshIntegrityError, DiscretizationError)

    def test_element_id(self) -> None:
        err = MeshIntegrityError(
            "dangling", element_id="e42",
        )

        assert err.element_id == "e42"
        assert err.context["element_id"] == "e42"

    def test_defaults_none(self) -> None:
        err = MeshIntegrityError()

        assert err.element_id is None


# ---------------------------------------------------------------
# Physics errors
# ---------------------------------------------------------------


class TestSolverDivergenceError:
    """SolverDivergenceError with iteration and residual."""

    def test_inherits(self) -> None:
        assert issubclass(SolverDivergenceError, PhysicsError)
        assert issubclass(SolverDivergenceError, AlphaGalerkinError)

    def test_iteration_and_residual(self) -> None:
        err = SolverDivergenceError(
            "diverged", iteration=50, residual=1e10,
        )

        assert err.iteration == 50
        assert err.residual == pytest.approx(1e10)
        assert err.context["iteration"] == 50
        assert err.context["residual"] == pytest.approx(1e10)

    def test_defaults_none(self) -> None:
        err = SolverDivergenceError()

        assert err.iteration is None
        assert err.residual is None


class TestStabilityViolationError:
    """StabilityViolationError with condition, value, threshold."""

    def test_inherits(self) -> None:
        assert issubclass(StabilityViolationError, PhysicsError)

    def test_all_fields(self) -> None:
        err = StabilityViolationError(
            "LBB violated",
            condition="inf_sup",
            value=1e-8,
            threshold=1e-6,
        )

        assert err.condition == "inf_sup"
        assert err.value == pytest.approx(1e-8)
        assert err.threshold == pytest.approx(1e-6)
        assert err.context["condition"] == "inf_sup"

    def test_defaults_none(self) -> None:
        err = StabilityViolationError()

        assert err.condition is None
        assert err.value is None
        assert err.threshold is None


# ---------------------------------------------------------------
# Training errors
# ---------------------------------------------------------------


class TestCheckpointError:
    """CheckpointError with checkpoint_path."""

    def test_inherits(self) -> None:
        assert issubclass(CheckpointError, TrainingError)
        assert issubclass(CheckpointError, AlphaGalerkinError)

    def test_checkpoint_path(self) -> None:
        err = CheckpointError(
            "corrupt", checkpoint_path="/ckpt/step_100.pt",
        )

        assert err.checkpoint_path == "/ckpt/step_100.pt"
        assert err.context["checkpoint_path"] == "/ckpt/step_100.pt"

    def test_defaults_none(self) -> None:
        err = CheckpointError()

        assert err.checkpoint_path is None


class TestCheckpointVersionMismatchError:
    """CheckpointVersionMismatchError with version fields."""

    def test_inherits(self) -> None:
        assert issubclass(
            CheckpointVersionMismatchError, CheckpointError,
        )

    def test_all_fields(self) -> None:
        err = CheckpointVersionMismatchError(
            "version mismatch",
            expected_version="2.0",
            found_version="1.0",
            checkpoint_path="/ckpt/old.pt",
        )

        assert err.expected_version == "2.0"
        assert err.found_version == "1.0"
        assert err.checkpoint_path == "/ckpt/old.pt"
        assert err.context["expected_version"] == "2.0"
        assert err.context["found_version"] == "1.0"

    def test_defaults_none(self) -> None:
        err = CheckpointVersionMismatchError()

        assert err.expected_version is None
        assert err.found_version is None
        assert err.checkpoint_path is None

    def test_context_passthrough(self) -> None:
        err = CheckpointVersionMismatchError(
            "err",
            expected_version="3.0",
            found_version="2.0",
            context={"extra": "data"},
        )

        assert err.context["extra"] == "data"
        assert err.context["expected_version"] == "3.0"


# ---------------------------------------------------------------
# Catch-all hierarchy checks
# ---------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify the full inheritance tree."""

    def test_mcts_error_base(self) -> None:
        assert issubclass(MCTSError, AlphaGalerkinError)

    def test_discretization_error_base(self) -> None:
        assert issubclass(DiscretizationError, AlphaGalerkinError)

    def test_physics_error_base(self) -> None:
        assert issubclass(PhysicsError, AlphaGalerkinError)

    def test_training_error_base(self) -> None:
        assert issubclass(TrainingError, AlphaGalerkinError)

    def test_all_catchable_via_root(self) -> None:
        """All custom exceptions are catchable via the root class."""
        exception_classes = [
            ConfigError,
            ConfigValidationError,
            MCTSError,
            TreeDepthExceededError,
            NoValidActionsError,
            DiscretizationError,
            InvariantViolationError,
            DOFBudgetExceededError,
            MeshIntegrityError,
            PhysicsError,
            SolverDivergenceError,
            StabilityViolationError,
            TrainingError,
            CheckpointError,
            CheckpointVersionMismatchError,
        ]

        for exc_cls in exception_classes:
            assert issubclass(exc_cls, AlphaGalerkinError), (
                f"{exc_cls.__name__} is not a subclass of "
                "AlphaGalerkinError"
            )
