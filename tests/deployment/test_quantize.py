"""Tests for ``CalibrationDataReader`` in ``src.deployment.quantize``.

``CalibrationDataReader`` is pure Python/numpy (no onnx/onnxruntime import at
class-body scope -- those are imported lazily inside ``ModelQuantizer``'s own
methods), so unlike the rest of this package's tests it needs no
``pytest.importorskip`` gating; see ``test_export_onnx_integration.py`` for
the package's convention on tests that *do* need it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.deployment.quantize import CalibrationDataReader


def _sample_generator(n: int, shape: tuple[int, ...] = (1, 4, 3, 3)) -> Iterator[np.ndarray]:
    """Yield ``n`` deterministic numpy arrays of the given shape.

    Args:
        n: Number of samples to yield.
        shape: Shape of each yielded array.

    Yields:
        Numpy arrays filled with their generation index (sample 0 is all
        0.0, sample 1 is all 1.0, and so on), so identity is easy to assert.

    """
    for i in range(n):
        yield np.full(shape, fill_value=float(i), dtype=np.float32)


class TestCalibrationDataReaderIteration:
    """Normal iteration through a calibration dataset."""

    def test_get_next_returns_samples_in_order(self) -> None:
        """get_next() yields wrapped samples in generator order."""
        reader = CalibrationDataReader(_sample_generator(3), input_name="board_state")

        first = reader.get_next()
        second = reader.get_next()
        third = reader.get_next()

        assert first is not None
        assert second is not None
        assert third is not None
        assert np.all(first["board_state"] == 0.0)
        assert np.all(second["board_state"] == 1.0)
        assert np.all(third["board_state"] == 2.0)

    def test_get_next_wraps_each_sample_under_input_name(self) -> None:
        """Each sample is returned as ``{input_name: array}``."""
        reader = CalibrationDataReader(_sample_generator(1), input_name="custom_input")

        sample = reader.get_next()

        assert sample is not None
        assert set(sample.keys()) == {"custom_input"}
        assert isinstance(sample["custom_input"], np.ndarray)

    def test_default_input_name_is_board_state(self) -> None:
        """Default ``input_name`` matches the ONNX graph's expected tensor name."""
        reader = CalibrationDataReader(_sample_generator(1))

        assert reader.input_name == "board_state"
        sample = reader.get_next()
        assert sample is not None
        assert "board_state" in sample

    def test_get_next_returns_none_after_exhaustion(self) -> None:
        """get_next() returns None once every sample has been consumed."""
        reader = CalibrationDataReader(_sample_generator(2))

        reader.get_next()
        reader.get_next()

        assert reader.get_next() is None
        # Exhaustion is sticky: repeated calls keep returning None rather
        # than raising or wrapping around.
        assert reader.get_next() is None

    def test_generator_is_materialized_eagerly_at_construction(self) -> None:
        """``__init__`` fully drains ``data_generator`` up front.

        ``_enum_data`` is built via a list comprehension over the generator
        inside ``__init__``, so its length is fixed at construction time and
        does not depend on how many ``get_next()`` calls follow.
        """
        reader = CalibrationDataReader(_sample_generator(5))

        assert len(reader._enum_data) == 5
        assert reader._current_index == 0


class TestCalibrationDataReaderBoundary:
    """Boundary case: an exhausted/empty ``data_generator``."""

    def test_get_next_on_empty_generator_returns_none_immediately(self) -> None:
        """get_next() on an empty ``data_generator`` returns None on the first call."""
        reader = CalibrationDataReader(_sample_generator(0))

        assert reader.get_next() is None

    def test_current_index_matches_or_exceeds_length_when_exhausted(self) -> None:
        """The exhaustion guard is ``current_index >= len(enum_data)``."""
        reader = CalibrationDataReader(_sample_generator(1))

        reader.get_next()  # consumes the only sample

        assert reader._current_index >= len(reader._enum_data)
        assert reader.get_next() is None


class TestCalibrationDataReaderSetRangeAndRewind:
    """Tests for the ONNX Runtime interface methods ``set_range()`` and ``rewind()``."""

    def test_set_range_seeks_to_start_index(self) -> None:
        """``set_range(start, end)`` repositions ``get_next()`` to ``start_index``."""
        reader = CalibrationDataReader(_sample_generator(4))

        reader.set_range(2, 4)
        sample = reader.get_next()

        assert sample is not None
        assert np.all(sample["board_state"] == 2.0)

    def test_set_range_end_index_is_unused(self) -> None:
        """``end_index`` is accepted for interface compatibility but has no effect."""
        reader = CalibrationDataReader(_sample_generator(3))

        reader.set_range(0, 999)  # end_index far beyond len(_enum_data)

        assert reader._current_index == 0
        assert reader.get_next() is not None

    def test_set_range_past_end_makes_get_next_return_none(self) -> None:
        """Seeking past the end of the data makes ``get_next()`` report exhaustion."""
        reader = CalibrationDataReader(_sample_generator(2))

        reader.set_range(5, 10)

        assert reader.get_next() is None

    def test_rewind_resets_to_beginning(self) -> None:
        """``rewind()`` allows re-iterating the full calibration set."""
        reader = CalibrationDataReader(_sample_generator(2))

        reader.get_next()
        reader.get_next()
        assert reader.get_next() is None  # exhausted

        reader.rewind()

        first_again = reader.get_next()
        assert first_again is not None
        assert np.all(first_again["board_state"] == 0.0)
