"""Tests for transfer validation."""

from __future__ import annotations

from src.research.validator import (
    TransferMetrics,
    TransferResult,
    TransferValidator,
    create_transfer_validator,
)


class TestTransferMetrics:
    """Tests for TransferMetrics dataclass."""

    def test_initialization(self, transfer_metrics: TransferMetrics) -> None:
        """Test metrics initialization."""
        assert transfer_metrics.target_size == 19
        assert transfer_metrics.source_size == 9
        assert transfer_metrics.mse == 0.01
        assert transfer_metrics.passed is True

    def test_passed_threshold(self) -> None:
        """Test passed threshold logic."""
        passing = TransferMetrics(
            target_size=19,
            source_size=9,
            n_samples=100,
            mse=0.01,
            mae=0.05,
            rmse=0.1,
            max_error=0.2,
            threshold=0.05,
            passed=True,
        )
        assert passing.passed

        failing = TransferMetrics(
            target_size=19,
            source_size=9,
            n_samples=100,
            mse=0.1,
            mae=0.15,
            rmse=0.3,
            max_error=0.5,
            threshold=0.05,
            passed=False,
        )
        assert not failing.passed

    def test_to_dict(self, transfer_metrics: TransferMetrics) -> None:
        """Test serialization to dict."""
        data = transfer_metrics.to_dict()

        assert data["target_size"] == 19
        assert data["mse"] == 0.01
        assert data["passed"] is True


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_initialization(self, transfer_result: TransferResult) -> None:
        """Test result initialization."""
        assert transfer_result.result_id == "test123"
        assert transfer_result.source_size == 9
        assert transfer_result.primary_target == 19
        assert transfer_result.passed

    def test_primary_metrics(self, transfer_result: TransferResult) -> None:
        """Test primary_metrics property."""
        metrics = transfer_result.primary_metrics
        assert metrics is not None
        assert metrics.target_size == 19

    def test_primary_mse(self, transfer_result: TransferResult) -> None:
        """Test primary_mse property."""
        mse = transfer_result.primary_mse
        assert mse is not None
        assert mse == 0.02

    def test_to_dict(self, transfer_result: TransferResult) -> None:
        """Test serialization to dict."""
        data = transfer_result.to_dict()

        assert data["result_id"] == "test123"
        assert data["passed"] is True
        assert 19 in data["target_metrics"]

    def test_summary(self, transfer_result: TransferResult) -> None:
        """Test summary generation."""
        summary = transfer_result.summary()

        assert "test123" in summary
        assert "PASS" in summary
        assert "9x9" in summary
        assert "19x19" in summary


class TestTransferValidator:
    """Tests for TransferValidator."""

    def test_initialization(self, transfer_validator: TransferValidator) -> None:
        """Test validator initialization."""
        assert transfer_validator.config.source_size == 9
        assert len(transfer_validator.results) == 0

    def test_validate(self, transfer_validator: TransferValidator) -> None:
        """Test validate method."""

        class MockModel:
            def __call__(self, x: list[float]) -> list[float]:
                return [v * 0.99 for v in x]

        model = MockModel()

        def data_generator(
            size: int, n_samples: int
        ) -> tuple[list[list[float]], list[list[float]]]:
            inputs = [[float(i) / size for i in range(size * size)] for _ in range(n_samples)]
            targets = [[float(i) / size for i in range(size * size)] for _ in range(n_samples)]
            return inputs, targets

        def evaluate_fn(
            model: MockModel,
            inputs: list[list[float]],
            targets: list[list[float]],
        ) -> dict[str, float]:
            # Mock evaluation
            return {"mse": 0.01, "mae": 0.05, "rmse": 0.1, "max_error": 0.2}

        result = transfer_validator.validate(
            model=model,
            data_generator=data_generator,
            evaluate_fn=evaluate_fn,
        )

        assert result.passed
        assert len(result.target_metrics) == 2
        assert len(transfer_validator.results) == 1

    def test_validate_failing(self, transfer_validator: TransferValidator) -> None:
        """Test validate with failing metrics."""

        class MockModel:
            pass

        def data_generator(
            size: int, n_samples: int
        ) -> tuple[list[list[float]], list[list[float]]]:
            return [[0.0]], [[0.0]]

        def evaluate_fn(
            model: MockModel,
            inputs: list[list[float]],
            targets: list[list[float]],
        ) -> dict[str, float]:
            # Return failing MSE
            return {"mse": 0.1, "mae": 0.2, "rmse": 0.3, "max_error": 0.5}

        result = transfer_validator.validate(
            model=MockModel(),
            data_generator=data_generator,
            evaluate_fn=evaluate_fn,
        )

        assert not result.passed
        assert not result.all_passed

    def test_get_best_result(self, transfer_validator: TransferValidator) -> None:
        """Test getting best result."""
        # Add some results
        result1 = TransferResult(
            result_id="r1",
            source_size=9,
            primary_target=13,
            passed=True,
        )
        result1.target_metrics = {
            13: TransferMetrics(
                target_size=13,
                source_size=9,
                n_samples=100,
                mse=0.02,
                mae=0.05,
                rmse=0.1,
                max_error=0.2,
                threshold=0.05,
                passed=True,
            )
        }

        result2 = TransferResult(
            result_id="r2",
            source_size=9,
            primary_target=13,
            passed=True,
        )
        result2.target_metrics = {
            13: TransferMetrics(
                target_size=13,
                source_size=9,
                n_samples=100,
                mse=0.01,
                mae=0.03,
                rmse=0.07,
                max_error=0.15,
                threshold=0.05,
                passed=True,
            )
        }

        transfer_validator._results = [result1, result2]

        best = transfer_validator.get_best_result()
        assert best is not None
        assert best.result_id == "r2"

    def test_compare_results(self, transfer_validator: TransferValidator) -> None:
        """Test comparing results."""
        result1 = TransferResult(
            result_id="r1",
            source_size=9,
            primary_target=13,
        )
        result1.target_metrics = {
            13: TransferMetrics(
                target_size=13,
                source_size=9,
                n_samples=100,
                mse=0.02,
                mae=0.05,
                rmse=0.1,
                max_error=0.2,
                threshold=0.05,
                passed=True,
            )
        }

        result2 = TransferResult(
            result_id="r2",
            source_size=9,
            primary_target=13,
        )
        result2.target_metrics = {
            13: TransferMetrics(
                target_size=13,
                source_size=9,
                n_samples=100,
                mse=0.01,
                mae=0.03,
                rmse=0.07,
                max_error=0.15,
                threshold=0.05,
                passed=True,
            )
        }

        comparison = transfer_validator.compare_results(result1, result2)

        assert comparison["result1_id"] == "r1"
        assert comparison["result2_id"] == "r2"
        assert 13 in comparison["targets"]
        assert comparison["targets"][13]["improvement"] > 0


class TestCreateTransferValidator:
    """Tests for create_transfer_validator factory."""

    def test_create_default(self) -> None:
        """Test creating default validator."""
        validator = create_transfer_validator()
        assert validator.config.source_size == 9
        assert 19 in validator.config.target_sizes

    def test_create_with_custom_config(self) -> None:
        """Test creating with custom config."""
        validator = create_transfer_validator(
            source_size=5,
            target_sizes=[5, 9],
            mse_threshold=0.01,
        )
        assert validator.config.source_size == 5
        assert validator.config.mse_threshold == 0.01
