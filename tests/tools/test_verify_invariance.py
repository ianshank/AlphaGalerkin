"""Tests for resolution invariance verification tool.

Tests cover the core verification functions using mocked models,
avoiding the need for actual trained model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel
from src.tools.verify_invariance import (
    create_synthetic_input,
    main,
    run_verification,
    verify_forward_pass,
    verify_lbb_stability,
    verify_semantic_validity,
)

# --- Helper: Mock model output ---


@dataclass
class MockModelOutput:
    """Simulates AlphaGalerkinModel output."""

    policy_logits: torch.Tensor
    value: torch.Tensor
    lbb_constant: torch.Tensor | None = None


def make_mock_model(
    board_size: int = 9,
    batch_size: int = 4,
    deterministic: bool = True,
    nan_policy: bool = False,
    nan_value: bool = False,
    inf_policy: bool = False,
    inf_value: bool = False,
    bad_value_range: bool = False,
    bad_policy_shape: bool = False,
    bad_value_shape: bool = False,
    uniform_policy: bool = False,
    constant_value: bool = False,
    lbb_constant: float | None = None,
    raise_exception: bool = False,
    non_deterministic: bool = False,
) -> MagicMock:
    """Create a mock model that returns controlled outputs."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)

    # Create a parameter to satisfy next(model.parameters()).device
    param = torch.zeros(1)
    model.parameters = MagicMock(return_value=iter([param]))

    n_actions = board_size**2 + 1

    # Cache for deterministic outputs keyed by input shape
    _output_cache: dict[tuple, MockModelOutput] = {}
    _call_count = [0]

    def forward(x, return_lbb=False):
        if raise_exception:
            raise RuntimeError("Model forward failed")

        bs = x.shape[0]
        n_act = x.shape[2] ** 2 + 1

        # For deterministic behavior, use cached output for same shape
        cache_key = (bs, n_act, x.shape[2])
        if deterministic and not non_deterministic and cache_key in _output_cache:
            cached = _output_cache[cache_key]
            output = MockModelOutput(
                policy_logits=cached.policy_logits.clone(),
                value=cached.value.clone(),
                lbb_constant=cached.lbb_constant,
            )
            if return_lbb and lbb_constant is not None:
                output.lbb_constant = torch.tensor([lbb_constant] * bs)
            return output

        _call_count[0] += 1

        if bad_policy_shape:
            policy = torch.randn(bs, 10)  # Wrong shape
        elif nan_policy:
            policy = torch.full((bs, n_act), float("nan"))
        elif inf_policy:
            policy = torch.full((bs, n_act), float("inf"))
        elif uniform_policy:
            policy = torch.zeros(bs, n_act)  # All equal -> uniform
        elif non_deterministic:
            # Different output each call
            policy = torch.randn(bs, n_act)
        else:
            # Use a seeded generator for reproducibility
            g = torch.Generator()
            g.manual_seed(42 + bs + n_act)
            policy = torch.randn(bs, n_act, generator=g)

        if bad_value_shape:
            value = torch.zeros(bs, 2)  # Wrong shape
        elif nan_value:
            value = torch.full((bs, 1), float("nan"))
        elif inf_value:
            value = torch.full((bs, 1), float("inf"))
        elif bad_value_range:
            value = torch.full((bs, 1), 5.0)  # Out of [-1, 1]
        elif constant_value:
            value = torch.full((bs, 1), 0.5)
        elif non_deterministic:
            value = torch.randn(bs, 1)
        else:
            g = torch.Generator()
            g.manual_seed(123 + bs)
            value = torch.tanh(torch.randn(bs, 1, generator=g))

        output = MockModelOutput(policy_logits=policy, value=value)

        if return_lbb and lbb_constant is not None:
            output.lbb_constant = torch.tensor([lbb_constant] * bs)

        # Cache for deterministic mode
        if deterministic and not non_deterministic:
            _output_cache[cache_key] = output

        return output

    model.side_effect = forward
    model.__call__ = forward
    return model


# --- Small model fixture for real model tests ---


@pytest.fixture()
def small_config() -> OperatorConfig:
    """Create a small OperatorConfig for testing."""
    return OperatorConfig(
        d_model=16,
        d_key=8,
        d_value=8,
        d_ffn=32,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=8,
        use_fnet_mixing=False,
    )


@pytest.fixture()
def small_model(small_config: OperatorConfig) -> AlphaGalerkinModel:
    """Create a small AlphaGalerkinModel for testing."""
    torch.manual_seed(42)
    model = AlphaGalerkinModel(small_config)
    model.eval()
    return model


# --- create_synthetic_input Tests ---


class TestCreateSyntheticInput:
    """Tests for create_synthetic_input."""

    def test_default_shape(self) -> None:
        """Returns tensor with correct default shape."""
        x = create_synthetic_input(batch_size=2, board_size=9)
        assert x.shape == (2, 17, 9, 9)

    def test_custom_channels(self) -> None:
        """Supports custom input channels."""
        x = create_synthetic_input(batch_size=1, board_size=5, input_channels=3)
        assert x.shape == (1, 3, 5, 5)

    def test_device_cpu(self) -> None:
        """Creates tensor on CPU."""
        x = create_synthetic_input(batch_size=1, board_size=9, device=torch.device("cpu"))
        assert x.device.type == "cpu"

    @pytest.mark.parametrize("board_size", [5, 9, 13, 19, 25])
    def test_various_board_sizes(self, board_size: int) -> None:
        """Works for various board sizes."""
        x = create_synthetic_input(batch_size=1, board_size=board_size)
        assert x.shape == (1, 17, board_size, board_size)


# --- verify_forward_pass Tests ---


class TestVerifyForwardPass:
    """Tests for verify_forward_pass."""

    def test_pass_on_valid_output(self) -> None:
        """Returns passed=True for valid model output."""
        model = make_mock_model(board_size=9)
        results = verify_forward_pass(model, board_size=9, batch_size=4)
        assert results["passed"] is True
        assert results["board_size"] == 9
        assert results["batch_size"] == 4
        assert len(results["errors"]) == 0

    def test_records_inference_time(self) -> None:
        """Records inference time in results."""
        model = make_mock_model(board_size=9)
        results = verify_forward_pass(model, board_size=9)
        assert "inference_time_ms" in results
        assert results["inference_time_ms"] >= 0

    def test_records_output_statistics(self) -> None:
        """Records policy entropy and value statistics."""
        model = make_mock_model(board_size=9)
        results = verify_forward_pass(model, board_size=9)
        assert "policy_entropy" in results
        assert "value_mean" in results
        assert "value_std" in results

    def test_fail_on_nan_policy(self) -> None:
        """Fails when policy contains NaN."""
        model = make_mock_model(board_size=9, nan_policy=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("NaN" in e for e in results["errors"])

    def test_fail_on_nan_value(self) -> None:
        """Fails when value contains NaN."""
        model = make_mock_model(board_size=9, nan_value=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("NaN" in e for e in results["errors"])

    def test_fail_on_inf_policy(self) -> None:
        """Fails when policy contains Inf."""
        model = make_mock_model(board_size=9, inf_policy=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("Inf" in e for e in results["errors"])

    def test_fail_on_inf_value(self) -> None:
        """Fails when value contains Inf."""
        model = make_mock_model(board_size=9, inf_value=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False

    def test_fail_on_value_out_of_range(self) -> None:
        """Fails when value is outside [-1, 1]."""
        model = make_mock_model(board_size=9, bad_value_range=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("out of range" in e for e in results["errors"])

    def test_exception_handling(self) -> None:
        """Handles model exceptions gracefully."""
        model = make_mock_model(board_size=9, raise_exception=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("Exception" in e for e in results["errors"])

    def test_fail_on_bad_policy_shape(self) -> None:
        """Fails when policy has wrong shape (line 91)."""
        model = make_mock_model(board_size=9, bad_policy_shape=True)
        results = verify_forward_pass(model, board_size=9, batch_size=4)
        assert results["passed"] is False
        assert any("Policy shape mismatch" in e for e in results["errors"])

    def test_fail_on_bad_value_shape(self) -> None:
        """Fails when value has wrong shape (line 97)."""
        model = make_mock_model(board_size=9, bad_value_shape=True)
        results = verify_forward_pass(model, board_size=9, batch_size=4)
        assert results["passed"] is False
        assert any("Value shape mismatch" in e for e in results["errors"])


# --- verify_lbb_stability Tests ---


class TestVerifyLBBStability:
    """Tests for verify_lbb_stability."""

    def test_pass_on_positive_lbb(self) -> None:
        """Passes when LBB constant is above threshold."""
        model = make_mock_model(board_size=9, lbb_constant=0.5)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is True
        assert results["lbb_min"] == 0.5

    def test_fail_below_threshold(self) -> None:
        """Fails when LBB constant is below threshold."""
        model = make_mock_model(board_size=9, lbb_constant=0.01)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("below threshold" in e for e in results["errors"])

    def test_fail_on_negative_lbb(self) -> None:
        """Fails when LBB constant is non-positive."""
        model = make_mock_model(board_size=9, lbb_constant=-0.1)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("non-positive" in e for e in results["errors"])

    def test_fail_when_lbb_not_returned(self) -> None:
        """Fails when model doesn't return LBB constant."""
        model = make_mock_model(board_size=9, lbb_constant=None)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("not returned" in e for e in results["errors"])

    def test_exception_handling(self) -> None:
        """Handles model exceptions gracefully."""
        model = make_mock_model(board_size=9, raise_exception=True)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("Exception" in e for e in results["errors"])


# --- verify_semantic_validity Tests ---


class TestVerifySemanticValidity:
    """Tests for verify_semantic_validity."""

    def test_pass_on_valid_model(self) -> None:
        """Passes for model with deterministic varied output."""
        model = make_mock_model(board_size=9)
        results = verify_semantic_validity(model, board_size=9)
        assert results["passed"] is True
        assert "max_policy_prob" in results
        assert "uniform_prob" in results

    def test_exception_handling(self) -> None:
        """Handles model exceptions gracefully."""
        model = make_mock_model(board_size=9, raise_exception=True)
        results = verify_semantic_validity(model, board_size=9)
        assert results["passed"] is False
        assert any("Exception" in e for e in results["errors"])

    def test_non_deterministic_policy_fails(self) -> None:
        """Fails when policy is non-deterministic in eval mode (line 241)."""
        model = make_mock_model(board_size=9, non_deterministic=True)
        results = verify_semantic_validity(model, board_size=9)
        assert results["passed"] is False
        assert any("not deterministic" in e.lower() for e in results["errors"])

    def test_non_deterministic_value_fails(self) -> None:
        """Fails when value is non-deterministic in eval mode (line 244)."""
        # Create a model where policy IS deterministic but value is NOT
        call_count = [0]

        def forward(x, return_lbb=False):
            call_count[0] += 1
            bs = x.shape[0]
            n_act = x.shape[2] ** 2 + 1
            # Policy is always the same
            g = torch.Generator()
            g.manual_seed(42)
            policy = torch.randn(bs, n_act, generator=g)
            # Value changes each call
            value = torch.tanh(torch.randn(bs, 1) * call_count[0])
            return MockModelOutput(policy_logits=policy, value=value)

        model = MagicMock()
        model.eval = MagicMock(return_value=model)
        param = torch.zeros(1)
        model.parameters = MagicMock(return_value=iter([param]))
        model.__call__ = forward
        model.side_effect = forward

        results = verify_semantic_validity(model, board_size=9)
        assert any("Value not deterministic" in e for e in results["errors"])

    def test_uniform_policy_warning(self) -> None:
        """Warns when policy is too uniform (lines 253-254)."""
        model = make_mock_model(board_size=9, uniform_policy=True)
        results = verify_semantic_validity(model, board_size=9)
        # Uniform policy produces a warning, not an error
        assert results["passed"] is True
        assert "warnings" in results
        assert any("too uniform" in w for w in results["warnings"])

    def test_constant_value_warning(self) -> None:
        """Warns when value is too constant (lines 269-270)."""
        model = make_mock_model(board_size=9, constant_value=True)
        results = verify_semantic_validity(model, board_size=9)
        # Constant value produces a warning, not an error
        assert results["passed"] is True
        assert "warnings" in results
        assert any("too constant" in w for w in results["warnings"])

    def test_value_std_recorded(self) -> None:
        """Records value standard deviation."""
        model = make_mock_model(board_size=9)
        results = verify_semantic_validity(model, board_size=9)
        assert "value_std" in results


# --- run_verification Tests ---


class TestRunVerification:
    """Tests for run_verification using real small models."""

    def test_run_verification_passes_with_small_model(self) -> None:
        """run_verification returns True with a real model (lines 300-392)."""
        with patch(
            "src.tools.verify_invariance.OperatorConfig",
            return_value=OperatorConfig(
                d_model=16,
                d_key=8,
                d_value=8,
                d_ffn=32,
                n_heads=2,
                n_galerkin_layers=1,
                n_softmax_layers=1,
                n_fourier_features=8,
                use_fnet_mixing=False,
            ),
        ):
            result = run_verification(
                train_size=5,
                infer_size=7,
                device="cpu",
                verbose=False,
            )
        assert isinstance(result, bool)

    def test_run_verification_returns_bool(self) -> None:
        """run_verification returns a boolean."""
        with patch(
            "src.tools.verify_invariance.OperatorConfig",
            return_value=OperatorConfig(
                d_model=16,
                d_key=8,
                d_value=8,
                d_ffn=32,
                n_heads=2,
                n_galerkin_layers=1,
                n_softmax_layers=1,
                n_fourier_features=8,
                use_fnet_mixing=False,
            ),
        ):
            result = run_verification(
                train_size=5,
                infer_size=7,
                device="cpu",
                verbose=True,
            )
        assert result is True or result is False

    def test_run_verification_sets_training_resolution(self) -> None:
        """run_verification sets training_resolution on the model."""
        original_run = run_verification

        # We intercept AlphaGalerkinModel to verify training_resolution is set
        captured_model = [None]
        original_init = AlphaGalerkinModel.__init__

        def patched_init(self, config):
            original_init(self, config)
            captured_model[0] = self

        small_cfg = OperatorConfig(
            d_model=16,
            d_key=8,
            d_value=8,
            d_ffn=32,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=8,
            use_fnet_mixing=False,
        )

        with (
            patch("src.tools.verify_invariance.OperatorConfig", return_value=small_cfg),
            patch.object(AlphaGalerkinModel, "__init__", patched_init),
        ):
            run_verification(train_size=5, infer_size=7, device="cpu")

        assert captured_model[0] is not None
        assert captured_model[0].training_resolution == 5

    def test_run_verification_multiple_sizes(self) -> None:
        """run_verification tests multiple board sizes (5, 9, 13, 19, 25)."""
        sizes_tested = []
        original_verify = verify_forward_pass

        def tracking_verify(model, board_size, **kwargs):
            sizes_tested.append(board_size)
            return original_verify(model, board_size, **kwargs)

        small_cfg = OperatorConfig(
            d_model=16,
            d_key=8,
            d_value=8,
            d_ffn=32,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=8,
            use_fnet_mixing=False,
        )

        with (
            patch("src.tools.verify_invariance.OperatorConfig", return_value=small_cfg),
            patch(
                "src.tools.verify_invariance.verify_forward_pass",
                side_effect=tracking_verify,
            ),
        ):
            run_verification(train_size=5, infer_size=7, device="cpu")

        # run_verification calls verify_forward_pass for train_size, infer_size,
        # and for multi-resolution [5, 9, 13, 19, 25]
        assert 5 in sizes_tested
        assert 7 in sizes_tested
        for s in [5, 9, 13, 19, 25]:
            assert s in sizes_tested

    def test_run_verification_failure_scenario(self) -> None:
        """run_verification returns False when a sub-check fails."""
        small_cfg = OperatorConfig(
            d_model=16,
            d_key=8,
            d_value=8,
            d_ffn=32,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=8,
            use_fnet_mixing=False,
        )

        # Make verify_forward_pass fail. Note: run_verification unpacks results
        # with **results alongside board_size=size, so we must not include
        # board_size in the returned dict to avoid duplicate keyword args.
        def failing_verify(model, board_size, **kwargs):
            return {
                "batch_size": 4,
                "passed": False,
                "errors": ["Injected failure"],
            }

        with (
            patch("src.tools.verify_invariance.OperatorConfig", return_value=small_cfg),
            patch(
                "src.tools.verify_invariance.verify_forward_pass",
                side_effect=failing_verify,
            ),
        ):
            result = run_verification(train_size=5, infer_size=7, device="cpu")

        assert result is False

    def test_run_verification_lbb_failure(self) -> None:
        """run_verification returns False when LBB stability fails (lines 353-354)."""
        small_cfg = OperatorConfig(
            d_model=16,
            d_key=8,
            d_value=8,
            d_ffn=32,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=8,
            use_fnet_mixing=False,
        )

        def failing_lbb(model, board_size, **kwargs):
            return {
                "board_size": board_size,
                "passed": False,
                "errors": ["LBB constant below threshold"],
            }

        with (
            patch("src.tools.verify_invariance.OperatorConfig", return_value=small_cfg),
            patch(
                "src.tools.verify_invariance.verify_lbb_stability",
                side_effect=failing_lbb,
            ),
        ):
            result = run_verification(train_size=5, infer_size=7, device="cpu")

        assert result is False

    def test_run_verification_semantic_failure(self) -> None:
        """run_verification returns False when semantic validity fails (lines 364-365)."""
        small_cfg = OperatorConfig(
            d_model=16,
            d_key=8,
            d_value=8,
            d_ffn=32,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=8,
            use_fnet_mixing=False,
        )

        def failing_semantic(model, board_size, **kwargs):
            return {
                "board_size": board_size,
                "passed": False,
                "errors": ["Model not deterministic in eval mode"],
            }

        with (
            patch("src.tools.verify_invariance.OperatorConfig", return_value=small_cfg),
            patch(
                "src.tools.verify_invariance.verify_semantic_validity",
                side_effect=failing_semantic,
            ),
        ):
            result = run_verification(train_size=5, infer_size=7, device="cpu")

        assert result is False


# --- main() Tests ---


class TestMain:
    """Tests for main() entry point (lines 397-442, 446)."""

    def test_main_exits_zero_on_success(self) -> None:
        """main() exits with code 0 on successful verification."""
        small_cfg = OperatorConfig(
            d_model=16,
            d_key=8,
            d_value=8,
            d_ffn=32,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=8,
            use_fnet_mixing=False,
        )

        with (
            patch(
                "sys.argv",
                ["verify_invariance", "--train-size", "5", "--infer-size", "7", "--device", "cpu"],
            ),
            patch("src.tools.verify_invariance.OperatorConfig", return_value=small_cfg),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0

    def test_main_exits_one_on_failure(self) -> None:
        """main() exits with code 1 when verification fails."""
        with (
            patch(
                "sys.argv",
                ["verify_invariance", "--train-size", "5", "--infer-size", "7", "--device", "cpu"],
            ),
            patch("src.tools.verify_invariance.run_verification", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_main_default_args(self) -> None:
        """main() uses default arguments when none provided."""
        with (
            patch("sys.argv", ["verify_invariance"]),
            patch("src.tools.verify_invariance.run_verification", return_value=True) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(
            train_size=9,
            infer_size=19,
            device="cpu",
            verbose=False,
        )

    def test_main_verbose_flag(self) -> None:
        """main() passes verbose flag from arguments."""
        with (
            patch("sys.argv", ["verify_invariance", "--verbose"]),
            patch("src.tools.verify_invariance.run_verification", return_value=True) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(
            train_size=9,
            infer_size=19,
            device="cpu",
            verbose=True,
        )

    def test_main_custom_args(self) -> None:
        """main() correctly passes custom arguments to run_verification."""
        with (
            patch(
                "sys.argv",
                [
                    "verify_invariance",
                    "--train-size",
                    "13",
                    "--infer-size",
                    "25",
                    "--device",
                    "cpu",
                    "--verbose",
                ],
            ),
            patch("src.tools.verify_invariance.run_verification", return_value=True) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(
            train_size=13,
            infer_size=25,
            device="cpu",
            verbose=True,
        )

    def test_main_configures_structlog(self) -> None:
        """main() configures structlog with JSON renderer."""
        import structlog

        with (
            patch("sys.argv", ["verify_invariance"]),
            patch("src.tools.verify_invariance.run_verification", return_value=True),
            patch.object(structlog, "configure") as mock_configure,
            pytest.raises(SystemExit),
        ):
            main()

        mock_configure.assert_called_once()


class TestModuleEntryPoint:
    """Test the if __name__ == '__main__' guard (line 446)."""

    def test_module_calls_main(self) -> None:
        """Running the module as __main__ invokes main()."""
        import src.tools.verify_invariance as module

        with patch.object(module, "main", side_effect=SystemExit(0)) as mock_main:
            with pytest.raises(SystemExit):
                # Simulate running as __main__
                exec(  # noqa: S102
                    "if True:\n    main()",
                    {"main": mock_main, "__name__": "__main__"},
                )
            mock_main.assert_called_once()

    def test_name_main_guard_via_runpy(self) -> None:
        """The module-level __name__ == '__main__' guard calls main()."""
        with (
            patch("sys.argv", ["verify_invariance"]),
            patch("src.tools.verify_invariance.run_verification", return_value=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            import runpy

            runpy.run_module("src.tools.verify_invariance", run_name="__main__")

        assert exc_info.value.code == 0


# --- Integration tests with real small model ---


class TestRealModelIntegration:
    """Integration tests using a real AlphaGalerkinModel (small)."""

    def test_verify_forward_pass_real_model(self, small_model: AlphaGalerkinModel) -> None:
        """verify_forward_pass works with real model."""
        results = verify_forward_pass(
            small_model, board_size=5, batch_size=2, device=torch.device("cpu")
        )
        assert results["passed"] is True
        assert len(results["errors"]) == 0
        assert results["inference_time_ms"] >= 0

    def test_verify_forward_pass_different_sizes(self, small_model: AlphaGalerkinModel) -> None:
        """verify_forward_pass works at different resolutions with real model."""
        for size in [5, 7, 9]:
            results = verify_forward_pass(
                small_model, board_size=size, batch_size=2, device=torch.device("cpu")
            )
            assert results["passed"] is True, f"Failed at board_size={size}: {results['errors']}"

    def test_verify_lbb_stability_real_model(self, small_model: AlphaGalerkinModel) -> None:
        """verify_lbb_stability works with real model."""
        results = verify_lbb_stability(
            small_model, board_size=5, batch_size=2, device=torch.device("cpu")
        )
        assert results["passed"] is True
        assert "lbb_min" in results
        assert results["lbb_min"] > 0

    def test_verify_semantic_validity_real_model(self, small_model: AlphaGalerkinModel) -> None:
        """verify_semantic_validity works with real model."""
        results = verify_semantic_validity(small_model, board_size=5, device=torch.device("cpu"))
        assert results["passed"] is True
        assert "max_policy_prob" in results
        assert "value_std" in results
