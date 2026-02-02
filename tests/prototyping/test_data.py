"""Tests for data generation."""

from __future__ import annotations

import pytest

from src.prototyping.data import (
    DataGenerator,
    SyntheticData,
    create_data_generator,
)


class TestSyntheticData:
    """Tests for SyntheticData dataclass."""

    def test_initialization(self, synthetic_data: SyntheticData) -> None:
        """Test data initialization."""
        assert synthetic_data.n_samples == 100
        assert len(synthetic_data.inputs) == 100
        assert len(synthetic_data.targets) == 100

    def test_len(self, synthetic_data: SyntheticData) -> None:
        """Test __len__."""
        assert len(synthetic_data) == 100

    def test_iter(self, synthetic_data: SyntheticData) -> None:
        """Test iteration."""
        count = 0
        for inp, target in synthetic_data:
            count += 1
            assert inp is not None
            assert target is not None
        assert count == 100

    def test_split(self, synthetic_data: SyntheticData) -> None:
        """Test train/test split."""
        train, test = synthetic_data.split(train_ratio=0.8)

        assert train.n_samples == 80
        assert test.n_samples == 20
        assert train.metadata["split"] == "train"
        assert test.metadata["split"] == "test"

    def test_split_no_shuffle(self, synthetic_data: SyntheticData) -> None:
        """Test split without shuffling."""
        train1, test1 = synthetic_data.split(shuffle=False, seed=42)
        train2, test2 = synthetic_data.split(shuffle=False, seed=42)

        # Without shuffle, same seed should give same split
        assert train1.inputs[0] == train2.inputs[0]

    def test_batch(self, synthetic_data: SyntheticData) -> None:
        """Test batch iteration."""
        batches = list(synthetic_data.batch(batch_size=32, shuffle=False))

        assert len(batches) == 4  # 100 / 32 = 3.125, so 4 batches
        assert len(batches[0][0]) == 32
        assert len(batches[-1][0]) <= 32

    def test_batch_drop_last(self, synthetic_data: SyntheticData) -> None:
        """Test batch with drop_last."""
        batches = list(
            synthetic_data.batch(
                batch_size=32,
                shuffle=False,
                drop_last=True,
            )
        )

        assert len(batches) == 3  # Drop incomplete batch
        for batch_inputs, batch_targets in batches:
            assert len(batch_inputs) == 32

    def test_to_dict(self, synthetic_data: SyntheticData) -> None:
        """Test serialization (without data)."""
        data = synthetic_data.to_dict()

        assert "data_id" in data
        assert data["n_samples"] == 100
        assert "inputs" not in data  # Data not serialized


class TestDataGenerator:
    """Tests for DataGenerator."""

    def test_initialization(self, data_generator: DataGenerator) -> None:
        """Test generator initialization."""
        assert data_generator.seed == 42

    def test_generate_linear(self, data_generator: DataGenerator) -> None:
        """Test linear data generation."""
        data = data_generator.generate(
            "linear",
            n_samples=50,
            n_features=5,
        )

        assert data.n_samples == 50
        assert data.input_shape == (5,)
        assert data.target_shape == (1,)
        assert data.metadata["type"] == "linear"

    def test_generate_polynomial(self, data_generator: DataGenerator) -> None:
        """Test polynomial data generation."""
        data = data_generator.generate(
            "polynomial",
            n_samples=50,
            degree=3,
        )

        assert data.n_samples == 50
        assert data.metadata["type"] == "polynomial"
        assert data.metadata["degree"] == 3

    def test_generate_sinusoidal(self, data_generator: DataGenerator) -> None:
        """Test sinusoidal data generation."""
        data = data_generator.generate(
            "sinusoidal",
            n_samples=50,
            frequency=2.0,
        )

        assert data.n_samples == 50
        assert data.metadata["type"] == "sinusoidal"
        assert data.metadata["frequency"] == 2.0

    def test_generate_classification(self, data_generator: DataGenerator) -> None:
        """Test classification data generation."""
        data = data_generator.generate(
            "classification",
            n_samples=50,
            n_classes=3,
        )

        assert data.n_samples == 50
        assert data.metadata["type"] == "classification"
        assert data.metadata["n_classes"] == 3

        # Check classes are valid
        for target in data.targets:
            assert 0 <= target[0] < 3

    def test_generate_board(self, data_generator: DataGenerator) -> None:
        """Test board data generation."""
        data = data_generator.generate(
            "board",
            n_samples=10,
            board_size=9,
        )

        assert data.n_samples == 10
        assert data.input_shape == (81,)  # 9x9
        assert data.target_shape == (81,)
        assert data.metadata["type"] == "board"

    def test_generate_poisson(self, data_generator: DataGenerator) -> None:
        """Test Poisson field data generation."""
        data = data_generator.generate(
            "poisson",
            n_samples=10,
            grid_size=7,
        )

        assert data.n_samples == 10
        assert data.input_shape == (49,)  # 7x7
        assert data.metadata["type"] == "poisson"

    def test_register_generator(self, data_generator: DataGenerator) -> None:
        """Test registering custom generator."""

        def custom_generator(n_samples: int, **kwargs: int) -> SyntheticData:
            return SyntheticData(
                data_id="custom",
                inputs=[[1.0]] * n_samples,
                targets=[[2.0]] * n_samples,
                n_samples=n_samples,
                metadata={"type": "custom"},
            )

        data_generator.register_generator("custom", custom_generator)
        data = data_generator.generate("custom", n_samples=5)

        assert data.metadata["type"] == "custom"
        assert data.n_samples == 5

    def test_unknown_generator(self, data_generator: DataGenerator) -> None:
        """Test unknown generator raises error."""
        with pytest.raises(ValueError, match="Unknown generator"):
            data_generator.generate("nonexistent", n_samples=10)

    def test_reproducibility(self) -> None:
        """Test seed reproducibility."""
        gen1 = DataGenerator(seed=123)
        gen2 = DataGenerator(seed=123)

        data1 = gen1.generate("linear", n_samples=10)
        data2 = gen2.generate("linear", n_samples=10)

        assert data1.inputs == data2.inputs


class TestSyntheticDataValidation:
    """Tests for SyntheticData validation."""

    def test_split_invalid_train_ratio_zero(self, synthetic_data: SyntheticData) -> None:
        """Test split with train_ratio=0 raises error."""
        with pytest.raises(ValueError, match="train_ratio must be between"):
            synthetic_data.split(train_ratio=0)

    def test_split_invalid_train_ratio_one(self, synthetic_data: SyntheticData) -> None:
        """Test split with train_ratio=1 raises error."""
        with pytest.raises(ValueError, match="train_ratio must be between"):
            synthetic_data.split(train_ratio=1)

    def test_split_invalid_train_ratio_negative(self, synthetic_data: SyntheticData) -> None:
        """Test split with negative train_ratio raises error."""
        with pytest.raises(ValueError, match="train_ratio must be between"):
            synthetic_data.split(train_ratio=-0.5)

    def test_split_invalid_train_ratio_greater_than_one(
        self, synthetic_data: SyntheticData
    ) -> None:
        """Test split with train_ratio>1 raises error."""
        with pytest.raises(ValueError, match="train_ratio must be between"):
            synthetic_data.split(train_ratio=1.5)

    def test_batch_invalid_batch_size_zero(self, synthetic_data: SyntheticData) -> None:
        """Test batch with batch_size=0 raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(synthetic_data.batch(batch_size=0))

    def test_batch_invalid_batch_size_negative(self, synthetic_data: SyntheticData) -> None:
        """Test batch with negative batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(synthetic_data.batch(batch_size=-1))


class TestCreateDataGenerator:
    """Tests for create_data_generator factory."""

    def test_create_default(self) -> None:
        """Test creating default generator."""
        generator = create_data_generator()
        assert generator is not None

    def test_create_with_seed(self) -> None:
        """Test creating with seed."""
        generator = create_data_generator(seed=42)
        assert generator.seed == 42
