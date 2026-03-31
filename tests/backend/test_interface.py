"""Tests for BackendInterface operations across all available backends.

Uses the parametrized ``backend`` fixture from root conftest.py which runs
each test on all available backends (torch, jax). Backends that are not
installed are automatically skipped.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from src.backend.types import BackendType, Precision

# ------------------------------------------------------------------
# Tensor creation
# ------------------------------------------------------------------


class TestTensorCreation:
    """Test tensor creation operations across backends."""

    def test_zeros_shape(self, backend) -> None:
        x = backend.zeros((3, 4))
        assert backend.shape(x) == (3, 4)

    def test_zeros_values(self, backend) -> None:
        x = backend.zeros((2, 3))
        npt.assert_allclose(backend.to_numpy(x), 0.0)

    def test_zeros_1d(self, backend) -> None:
        x = backend.zeros((5,))
        assert backend.shape(x) == (5,)
        npt.assert_allclose(backend.to_numpy(x), 0.0)

    def test_ones_shape(self, backend) -> None:
        x = backend.ones((2, 5))
        assert backend.shape(x) == (2, 5)

    def test_ones_values(self, backend) -> None:
        x = backend.ones((3, 3))
        npt.assert_allclose(backend.to_numpy(x), 1.0)

    def test_full_shape(self, backend) -> None:
        x = backend.full((3, 3), 7.0)
        assert backend.shape(x) == (3, 3)

    def test_full_values(self, backend) -> None:
        x = backend.full((2, 2), 3.14)
        npt.assert_allclose(backend.to_numpy(x), 3.14, rtol=1e-5)

    def test_full_zero_fill(self, backend) -> None:
        x = backend.full((2, 2), 0.0)
        npt.assert_allclose(backend.to_numpy(x), 0.0)

    def test_randn_shape(self, backend) -> None:
        x = backend.randn((5, 6))
        assert backend.shape(x) == (5, 6)

    def test_randn_not_all_zero(self, backend) -> None:
        x = backend.randn((100,))
        assert np.any(backend.to_numpy(x) != 0.0)

    def test_rand_shape(self, backend) -> None:
        x = backend.rand((8, 4))
        assert backend.shape(x) == (8, 4)

    def test_rand_range(self, backend) -> None:
        x = backend.rand((1000,))
        arr = backend.to_numpy(x)
        assert arr.min() >= 0.0
        assert arr.max() < 1.0

    def test_arange_start_stop_step(self, backend) -> None:
        x = backend.arange(0.0, 5.0, 1.0)
        npt.assert_allclose(backend.to_numpy(x), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_arange_single_arg(self, backend) -> None:
        x = backend.arange(3.0)
        npt.assert_allclose(backend.to_numpy(x), [0.0, 1.0, 2.0])

    def test_arange_with_step(self, backend) -> None:
        x = backend.arange(0.0, 10.0, 2.0)
        npt.assert_allclose(backend.to_numpy(x), [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_linspace(self, backend) -> None:
        x = backend.linspace(0.0, 1.0, 5)
        npt.assert_allclose(backend.to_numpy(x), [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_linspace_shape(self, backend) -> None:
        x = backend.linspace(-1.0, 1.0, 50)
        assert backend.shape(x) == (50,)

    def test_tensor_from_list(self, backend) -> None:
        x = backend.tensor([1.0, 2.0, 3.0])
        npt.assert_allclose(backend.to_numpy(x), [1.0, 2.0, 3.0])

    def test_tensor_from_nested_list(self, backend) -> None:
        data = [[1.0, 2.0], [3.0, 4.0]]
        x = backend.tensor(data)
        npt.assert_allclose(backend.to_numpy(x), np.array(data))

    def test_from_numpy_roundtrip(self, backend) -> None:
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = backend.from_numpy(original)
        result = backend.to_numpy(x)
        npt.assert_allclose(result, original)

    def test_from_numpy_2d(self, backend) -> None:
        original = np.random.default_rng(42).standard_normal((4, 5)).astype(np.float32)
        x = backend.from_numpy(original)
        result = backend.to_numpy(x)
        npt.assert_allclose(result, original, rtol=1e-6)

    def test_to_numpy_returns_ndarray(self, backend) -> None:
        x = backend.ones((3,))
        result = backend.to_numpy(x)
        assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# Tensor properties
# ------------------------------------------------------------------


class TestTensorProperties:
    """Test tensor property operations."""

    def test_shape(self, backend) -> None:
        x = backend.zeros((2, 3, 4))
        assert backend.shape(x) == (2, 3, 4)

    def test_shape_1d(self, backend) -> None:
        x = backend.zeros((7,))
        assert backend.shape(x) == (7,)

    def test_numel(self, backend) -> None:
        x = backend.zeros((2, 3, 4))
        assert backend.numel(x) == 24

    def test_numel_1d(self, backend) -> None:
        x = backend.zeros((10,))
        assert backend.numel(x) == 10

    def test_dtype_returns_something(self, backend) -> None:
        x = backend.zeros((2,))
        dt = backend.dtype(x)
        assert dt is not None


# ------------------------------------------------------------------
# Tensor manipulation
# ------------------------------------------------------------------


class TestShapeOperations:
    """Test tensor shape manipulation."""

    def test_reshape(self, backend) -> None:
        x = backend.arange(0.0, 12.0)
        y = backend.reshape(x, (3, 4))
        assert backend.shape(y) == (3, 4)

    def test_reshape_preserves_values(self, backend) -> None:
        x = backend.arange(6.0)
        y = backend.reshape(x, (2, 3))
        npt.assert_allclose(backend.to_numpy(y).ravel(), backend.to_numpy(x))

    def test_transpose_2d(self, backend) -> None:
        x = backend.zeros((2, 3))
        y = backend.transpose(x)
        assert backend.shape(y) == (3, 2)

    def test_transpose_3d_with_axes(self, backend) -> None:
        x = backend.zeros((2, 3, 4))
        y = backend.transpose(x, axes=(0, 2, 1))
        assert backend.shape(y) == (2, 4, 3)

    def test_expand_dims_front(self, backend) -> None:
        x = backend.zeros((3, 4))
        y = backend.expand_dims(x, axis=0)
        assert backend.shape(y) == (1, 3, 4)

    def test_expand_dims_back(self, backend) -> None:
        x = backend.zeros((3, 4))
        y = backend.expand_dims(x, axis=-1)
        assert backend.shape(y) == (3, 4, 1)

    def test_squeeze_specific_axis(self, backend) -> None:
        x = backend.zeros((1, 3, 1, 4))
        y = backend.squeeze(x, axis=0)
        assert backend.shape(y) == (3, 1, 4)

    def test_squeeze_all(self, backend) -> None:
        x = backend.zeros((1, 3, 1))
        y = backend.squeeze(x)
        assert backend.shape(y) == (3,)

    def test_cat_axis0(self, backend) -> None:
        a = backend.ones((2, 3))
        b = backend.zeros((2, 3))
        c = backend.cat([a, b], axis=0)
        assert backend.shape(c) == (4, 3)

    def test_cat_axis1(self, backend) -> None:
        a = backend.ones((2, 3))
        b = backend.zeros((2, 4))
        c = backend.cat([a, b], axis=1)
        assert backend.shape(c) == (2, 7)

    def test_cat_values(self, backend) -> None:
        a = backend.ones((2, 3))
        b = backend.zeros((2, 3))
        c = backend.cat([a, b], axis=0)
        arr = backend.to_numpy(c)
        npt.assert_allclose(arr[:2, :], 1.0)
        npt.assert_allclose(arr[2:, :], 0.0)

    def test_stack_axis0(self, backend) -> None:
        a = backend.ones((3,))
        b = backend.zeros((3,))
        c = backend.stack([a, b], axis=0)
        assert backend.shape(c) == (2, 3)

    def test_stack_axis1(self, backend) -> None:
        a = backend.ones((3, 4))
        b = backend.zeros((3, 4))
        c = backend.stack([a, b], axis=1)
        assert backend.shape(c) == (3, 2, 4)

    def test_split(self, backend) -> None:
        x = backend.arange(0.0, 12.0)
        x = backend.reshape(x, (4, 3))
        parts = backend.split(x, 2, axis=0)
        assert len(parts) == 2
        assert backend.shape(parts[0]) == (2, 3)
        assert backend.shape(parts[1]) == (2, 3)

    def test_pad_shape(self, backend) -> None:
        x = backend.ones((2, 3))
        y = backend.pad(x, [(1, 1), (2, 2)], value=0.0)
        assert backend.shape(y) == (4, 7)

    def test_pad_values(self, backend) -> None:
        x = backend.ones((2, 3))
        y = backend.pad(x, [(1, 1), (2, 2)], value=0.0)
        arr = backend.to_numpy(y)
        # Center should be ones
        npt.assert_allclose(arr[1:3, 2:5], 1.0)
        # Corners should be zeros
        assert arr[0, 0] == 0.0


# ------------------------------------------------------------------
# Math operations
# ------------------------------------------------------------------


class TestMathOperations:
    """Test element-wise and reduction math operations."""

    def test_add(self, backend) -> None:
        a = backend.tensor([1.0, 2.0, 3.0])
        b = backend.tensor([4.0, 5.0, 6.0])
        c = backend.add(a, b)
        npt.assert_allclose(backend.to_numpy(c), [5.0, 7.0, 9.0])

    def test_mul(self, backend) -> None:
        a = backend.tensor([2.0, 3.0])
        b = backend.tensor([4.0, 5.0])
        c = backend.mul(a, b)
        npt.assert_allclose(backend.to_numpy(c), [8.0, 15.0])

    def test_matmul(self, backend) -> None:
        a = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = backend.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = backend.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        npt.assert_allclose(backend.to_numpy(c), expected)

    def test_einsum_sum(self, backend) -> None:
        a = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = backend.einsum("ij->", a)
        npt.assert_allclose(backend.float_scalar(result), 10.0, rtol=1e-5)

    def test_einsum_trace(self, backend) -> None:
        a = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = backend.einsum("ii->", a)
        npt.assert_allclose(backend.float_scalar(result), 5.0, rtol=1e-5)

    def test_einsum_outer(self, backend) -> None:
        a = backend.tensor([1.0, 2.0])
        b = backend.tensor([3.0, 4.0, 5.0])
        result = backend.einsum("i,j->ij", a, b)
        expected = np.array([[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])
        npt.assert_allclose(backend.to_numpy(result), expected)

    def test_sum_all(self, backend) -> None:
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        total = backend.sum(x)
        npt.assert_allclose(backend.float_scalar(total), 10.0, rtol=1e-5)

    def test_sum_axis(self, backend) -> None:
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        row_sums = backend.sum(x, axis=1)
        npt.assert_allclose(backend.to_numpy(row_sums), [3.0, 7.0])

    def test_sum_keepdims(self, backend) -> None:
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        s = backend.sum(x, axis=1, keepdims=True)
        assert backend.shape(s) == (2, 1)

    def test_mean_all(self, backend) -> None:
        x = backend.tensor([1.0, 2.0, 3.0, 4.0])
        m = backend.mean(x)
        npt.assert_allclose(backend.float_scalar(m), 2.5, rtol=1e-5)

    def test_mean_axis(self, backend) -> None:
        x = backend.tensor([[1.0, 3.0], [5.0, 7.0]])
        m = backend.mean(x, axis=1)
        npt.assert_allclose(backend.to_numpy(m), [2.0, 6.0])

    def test_mean_keepdims(self, backend) -> None:
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        m = backend.mean(x, axis=0, keepdims=True)
        assert backend.shape(m) == (1, 2)

    def test_max_all(self, backend) -> None:
        x = backend.tensor([1.0, 5.0, 3.0])
        m = backend.max(x)
        npt.assert_allclose(backend.float_scalar(m), 5.0, rtol=1e-5)

    def test_max_axis(self, backend) -> None:
        x = backend.tensor([[1.0, 4.0], [3.0, 2.0]])
        result = backend.max(x, axis=1)
        npt.assert_allclose(backend.to_numpy(result), [4.0, 3.0])

    def test_min_all(self, backend) -> None:
        x = backend.tensor([1.0, 5.0, 3.0])
        m = backend.min(x)
        npt.assert_allclose(backend.float_scalar(m), 1.0, rtol=1e-5)

    def test_min_axis(self, backend) -> None:
        x = backend.tensor([[1.0, 4.0], [3.0, 2.0]])
        result = backend.min(x, axis=0)
        npt.assert_allclose(backend.to_numpy(result), [1.0, 2.0])

    def test_abs(self, backend) -> None:
        x = backend.tensor([-1.0, 2.0, -3.0])
        y = backend.abs(x)
        npt.assert_allclose(backend.to_numpy(y), [1.0, 2.0, 3.0])

    def test_sqrt(self, backend) -> None:
        x = backend.tensor([4.0, 9.0, 16.0])
        y = backend.sqrt(x)
        npt.assert_allclose(backend.to_numpy(y), [2.0, 3.0, 4.0], rtol=1e-5)

    def test_exp(self, backend) -> None:
        x = backend.tensor([0.0, 1.0])
        y = backend.exp(x)
        npt.assert_allclose(backend.to_numpy(y), [1.0, np.e], rtol=1e-5)

    def test_log(self, backend) -> None:
        x = backend.tensor([1.0, np.e])
        y = backend.log(x)
        npt.assert_allclose(backend.to_numpy(y), [0.0, 1.0], atol=1e-5)

    def test_cos(self, backend) -> None:
        x = backend.tensor([0.0, np.pi])
        c = backend.cos(x)
        npt.assert_allclose(backend.to_numpy(c), [1.0, -1.0], atol=1e-6)

    def test_sin(self, backend) -> None:
        x = backend.tensor([0.0, np.pi / 2])
        s = backend.sin(x)
        npt.assert_allclose(backend.to_numpy(s), [0.0, 1.0], atol=1e-6)

    def test_clamp(self, backend) -> None:
        x = backend.tensor([-1.0, 0.5, 2.0])
        y = backend.clamp(x, min_val=0.0, max_val=1.0)
        npt.assert_allclose(backend.to_numpy(y), [0.0, 0.5, 1.0])

    def test_clamp_min_only(self, backend) -> None:
        x = backend.tensor([-2.0, 0.5, 3.0])
        result = backend.clamp(x, min_val=0.0)
        arr = backend.to_numpy(result)
        npt.assert_allclose(arr[0], 0.0)
        npt.assert_allclose(arr[2], 3.0)

    def test_clamp_max_only(self, backend) -> None:
        x = backend.tensor([-2.0, 0.5, 3.0])
        result = backend.clamp(x, max_val=1.0)
        arr = backend.to_numpy(result)
        npt.assert_allclose(arr[0], -2.0)
        npt.assert_allclose(arr[2], 1.0)

    def test_where(self, backend) -> None:
        cond = backend.tensor([1.0, 0.0, 1.0])
        a = backend.tensor([10.0, 20.0, 30.0])
        b = backend.tensor([100.0, 200.0, 300.0])
        threshold = backend.full((3,), 0.5)
        # Use element-wise comparison for cross-backend compat
        # cond > 0.5 produces a boolean tensor in both torch and jax
        result = backend.where(cond > 0.5, a, b)
        npt.assert_allclose(backend.to_numpy(result), [10.0, 200.0, 30.0])

    def test_pow_scalar_exponent(self, backend) -> None:
        x = backend.tensor([2.0, 3.0])
        y = backend.pow(x, 2.0)
        npt.assert_allclose(backend.to_numpy(y), [4.0, 9.0])

    def test_pow_tensor_exponent(self, backend) -> None:
        x = backend.tensor([2.0, 3.0])
        exp = backend.tensor([3.0, 2.0])
        result = backend.pow(x, exp)
        npt.assert_allclose(backend.to_numpy(result), [8.0, 9.0], rtol=1e-5)


# ------------------------------------------------------------------
# Activation functions
# ------------------------------------------------------------------


class TestActivations:
    """Test activation functions."""

    def test_softmax_sums_to_one(self, backend) -> None:
        x = backend.randn((4, 10))
        y = backend.softmax(x, axis=-1)
        sums = backend.sum(y, axis=-1)
        npt.assert_allclose(backend.to_numpy(sums), 1.0, atol=1e-5)

    def test_softmax_positive(self, backend) -> None:
        x = backend.randn((3, 5))
        y = backend.softmax(x, axis=-1)
        assert backend.to_numpy(y).min() >= 0.0

    def test_log_softmax_equals_log_of_softmax(self, backend) -> None:
        x = backend.tensor([[1.0, 2.0, 3.0]])
        ls = backend.log_softmax(x, axis=-1)
        s = backend.softmax(x, axis=-1)
        expected = backend.log(s)
        npt.assert_allclose(backend.to_numpy(ls), backend.to_numpy(expected), atol=1e-5)

    def test_relu_positive(self, backend) -> None:
        x = backend.tensor([1.0, 2.0, 3.0])
        y = backend.relu(x)
        npt.assert_allclose(backend.to_numpy(y), [1.0, 2.0, 3.0])

    def test_relu_negative(self, backend) -> None:
        x = backend.tensor([-1.0, -2.0, -3.0])
        y = backend.relu(x)
        npt.assert_allclose(backend.to_numpy(y), [0.0, 0.0, 0.0])

    def test_relu_mixed(self, backend) -> None:
        x = backend.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = backend.relu(x)
        npt.assert_allclose(backend.to_numpy(y), [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_gelu_at_zero(self, backend) -> None:
        x = backend.tensor([0.0])
        y = backend.gelu(x)
        npt.assert_allclose(backend.to_numpy(y), [0.0], atol=1e-5)

    def test_gelu_positive(self, backend) -> None:
        x = backend.tensor([1.0, 2.0])
        y = backend.gelu(x)
        arr = backend.to_numpy(y)
        assert np.all(arr > 0)

    def test_sigmoid_range(self, backend) -> None:
        x = backend.randn((100,))
        y = backend.sigmoid(x)
        arr = backend.to_numpy(y)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_sigmoid_at_zero(self, backend) -> None:
        x = backend.tensor([0.0])
        y = backend.sigmoid(x)
        npt.assert_allclose(backend.to_numpy(y), [0.5], rtol=1e-5)

    def test_tanh_range(self, backend) -> None:
        x = backend.randn((100,))
        y = backend.tanh(x)
        arr = backend.to_numpy(y)
        assert arr.min() >= -1.0
        assert arr.max() <= 1.0

    def test_tanh_at_zero(self, backend) -> None:
        x = backend.tensor([0.0])
        y = backend.tanh(x)
        npt.assert_allclose(backend.to_numpy(y), [0.0], atol=1e-6)


# ------------------------------------------------------------------
# FFT operations
# ------------------------------------------------------------------


class TestFFTOperations:
    """Test FFT operations."""

    def test_fft2_ifft2_roundtrip(self, backend) -> None:
        x = backend.randn((4, 8))
        freq = backend.fft2(x)
        recovered = backend.ifft2(freq)
        npt.assert_allclose(
            backend.to_numpy(x),
            np.real(backend.to_numpy(recovered)),
            atol=1e-5,
        )

    def test_rfft2_irfft2_roundtrip(self, backend) -> None:
        x = backend.randn((2, 3, 8, 8))
        freq = backend.rfft2(x)
        reconstructed = backend.irfft2(freq, s=(8, 8))
        npt.assert_allclose(
            backend.to_numpy(x),
            backend.to_numpy(reconstructed),
            atol=1e-5,
        )

    def test_rfft2_output_shape(self, backend) -> None:
        x = backend.randn((4, 8))
        freq = backend.rfft2(x)
        shape = backend.shape(freq)
        assert shape[0] == 4
        assert shape[1] == 5  # 8 // 2 + 1

    def test_fftfreq(self, backend) -> None:
        freqs = backend.fftfreq(4)
        expected = np.array([0.0, 0.25, -0.5, -0.25])
        npt.assert_allclose(backend.to_numpy(freqs), expected, atol=1e-6)

    def test_rfftfreq(self, backend) -> None:
        freqs = backend.rfftfreq(4)
        expected = np.array([0.0, 0.25, 0.5])
        npt.assert_allclose(backend.to_numpy(freqs), expected, atol=1e-6)

    def test_fftfreq_with_spacing(self, backend) -> None:
        freqs = backend.fftfreq(8, d=0.5)
        expected = np.fft.fftfreq(8, d=0.5)
        npt.assert_allclose(backend.to_numpy(freqs), expected, atol=1e-6)

    def test_rfftfreq_with_spacing(self, backend) -> None:
        freqs = backend.rfftfreq(8, d=0.5)
        expected = np.fft.rfftfreq(8, d=0.5)
        npt.assert_allclose(backend.to_numpy(freqs), expected, atol=1e-6)


# ------------------------------------------------------------------
# Linear algebra
# ------------------------------------------------------------------


class TestLinearAlgebra:
    """Test linear algebra operations."""

    def test_svdvals_identity(self, backend) -> None:
        x = backend.tensor([[1.0, 0.0], [0.0, 1.0]])
        sv = backend.svdvals(x)
        npt.assert_allclose(backend.to_numpy(sv), [1.0, 1.0], atol=1e-5)

    def test_svdvals_shape(self, backend) -> None:
        x = backend.randn((3, 4))
        sv = backend.svdvals(x)
        assert backend.shape(sv) == (3,)

    def test_svdvals_positive(self, backend) -> None:
        x = backend.randn((4, 3))
        sv = backend.svdvals(x)
        arr = backend.to_numpy(sv)
        assert np.all(arr >= -1e-6)  # Allow small negative due to fp

    def test_svdvals_sorted_descending(self, backend) -> None:
        x = backend.randn((5, 3))
        sv = backend.svdvals(x)
        arr = backend.to_numpy(sv)
        for i in range(len(arr) - 1):
            assert arr[i] >= arr[i + 1] - 1e-6

    def test_norm_l2_vector(self, backend) -> None:
        x = backend.tensor([3.0, 4.0])
        n = backend.norm(x)
        npt.assert_allclose(backend.float_scalar(n), 5.0, rtol=1e-5)

    def test_norm_frobenius(self, backend) -> None:
        x = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        n = backend.norm(x)
        expected = np.sqrt(1 + 4 + 9 + 16)
        npt.assert_allclose(backend.float_scalar(n), expected, rtol=1e-5)


# ------------------------------------------------------------------
# Dtype management
# ------------------------------------------------------------------


class TestDtypeManagement:
    """Test dtype conversion and querying."""

    def test_get_dtype_float32(self, backend) -> None:
        dt = backend.get_dtype(Precision.FLOAT32)
        x = backend.zeros((2,), dtype=dt)
        assert "float32" in str(backend.dtype(x))

    def test_get_dtype_float64(self, backend) -> None:
        dt = backend.get_dtype(Precision.FLOAT64)
        assert dt is not None

    def test_cast(self, backend) -> None:
        x = backend.zeros((2,))
        dt64 = backend.get_dtype(Precision.FLOAT64)
        y = backend.cast(x, dt64)
        assert "float64" in str(backend.dtype(y))

    def test_cast_preserves_values(self, backend) -> None:
        x = backend.tensor([1.5, 2.5])
        dt64 = backend.get_dtype(Precision.FLOAT64)
        y = backend.cast(x, dt64)
        npt.assert_allclose(backend.to_numpy(y), [1.5, 2.5])


# ------------------------------------------------------------------
# Device management
# ------------------------------------------------------------------


class TestDeviceManagement:
    """Test device management operations."""

    def test_get_default_device_is_string(self, backend) -> None:
        device = backend.get_default_device()
        assert isinstance(device, str)
        assert device in ("cpu", "cuda", "gpu", "tpu")

    def test_to_device_cpu(self, backend) -> None:
        x = backend.ones((3,))
        y = backend.to_device(x, "cpu")
        npt.assert_allclose(backend.to_numpy(y), [1.0, 1.0, 1.0])


# ------------------------------------------------------------------
# Random state
# ------------------------------------------------------------------


class TestRandomSeed:
    """Test reproducibility via set_seed."""

    def test_set_seed_reproducibility(self, backend) -> None:
        backend.set_seed(12345)
        a = backend.to_numpy(backend.randn((10,)))
        backend.set_seed(12345)
        b = backend.to_numpy(backend.randn((10,)))
        npt.assert_allclose(a, b)

    def test_different_seeds_different_values(self, backend) -> None:
        backend.set_seed(111)
        a = backend.to_numpy(backend.randn((100,)))
        backend.set_seed(222)
        b = backend.to_numpy(backend.randn((100,)))
        assert not np.allclose(a, b)


# ------------------------------------------------------------------
# Meshgrid
# ------------------------------------------------------------------


class TestMeshgrid:
    """Test meshgrid operation."""

    def test_meshgrid_ij_shapes(self, backend) -> None:
        x = backend.arange(0.0, 3.0)
        y = backend.arange(0.0, 2.0)
        xx, yy = backend.meshgrid(x, y, indexing="ij")
        assert backend.shape(xx) == (3, 2)
        assert backend.shape(yy) == (3, 2)

    def test_meshgrid_values(self, backend) -> None:
        x = backend.tensor([1.0, 2.0])
        y = backend.tensor([3.0, 4.0, 5.0])
        xx, yy = backend.meshgrid(x, y, indexing="ij")
        xx_np = backend.to_numpy(xx)
        yy_np = backend.to_numpy(yy)
        npt.assert_allclose(xx_np[0, :], [1.0, 1.0, 1.0])
        npt.assert_allclose(xx_np[1, :], [2.0, 2.0, 2.0])
        npt.assert_allclose(yy_np[:, 0], [3.0, 3.0])
        npt.assert_allclose(yy_np[:, 2], [5.0, 5.0])


# ------------------------------------------------------------------
# ones_like / zeros_like / float_scalar
# ------------------------------------------------------------------


class TestTensorHelpers:
    """Test ones_like, zeros_like, and float_scalar."""

    def test_ones_like_values(self, backend) -> None:
        x = backend.zeros((2, 3))
        y = backend.ones_like(x)
        npt.assert_allclose(backend.to_numpy(y), 1.0)

    def test_ones_like_shape(self, backend) -> None:
        x = backend.zeros((4, 5))
        y = backend.ones_like(x)
        assert backend.shape(y) == (4, 5)

    def test_zeros_like_values(self, backend) -> None:
        x = backend.ones((2, 3))
        y = backend.zeros_like(x)
        npt.assert_allclose(backend.to_numpy(y), 0.0)

    def test_zeros_like_shape(self, backend) -> None:
        x = backend.ones((3, 7))
        y = backend.zeros_like(x)
        assert backend.shape(y) == (3, 7)

    def test_float_scalar(self, backend) -> None:
        x = backend.tensor(3.14)
        val = backend.float_scalar(x)
        assert isinstance(val, float)
        npt.assert_allclose(val, 3.14, rtol=1e-5)

    def test_float_scalar_from_sum(self, backend) -> None:
        x = backend.tensor([1.0, 2.0, 3.0])
        s = backend.sum(x)
        val = backend.float_scalar(s)
        npt.assert_allclose(val, 6.0, rtol=1e-5)


# ------------------------------------------------------------------
# Backend properties
# ------------------------------------------------------------------


class TestBackendProperties:
    """Test backend name and dtype properties."""

    def test_name_is_backend_type(self, backend) -> None:
        assert isinstance(backend.name, BackendType)

    def test_default_dtype_not_none(self, backend) -> None:
        assert backend.default_dtype is not None


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Test behavior with edge-case inputs."""

    def test_empty_tensor(self, backend) -> None:
        x = backend.zeros((0, 3))
        assert backend.shape(x) == (0, 3)
        assert backend.numel(x) == 0

    def test_scalar_tensor(self, backend) -> None:
        x = backend.tensor(5.0)
        val = backend.float_scalar(x)
        npt.assert_allclose(val, 5.0)

    def test_single_element(self, backend) -> None:
        x = backend.tensor([42.0])
        assert backend.shape(x) == (1,)
        npt.assert_allclose(backend.float_scalar(backend.sum(x)), 42.0)

    def test_large_tensor_sum(self, backend) -> None:
        x = backend.ones((100, 100))
        s = backend.sum(x)
        npt.assert_allclose(backend.float_scalar(s), 10000.0)
