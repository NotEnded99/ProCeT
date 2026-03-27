import numpy as np


class NumpyTranslator:
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = np.float64
        self.dtype = dtype

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: np.ndarray of floats [n, m]
        :param b: np.ndarray of floats [m]

        :return: np.ndarray of floats [n]
        """
        return self.squeeze(np.matmul(a, self.unsqueeze(b, dim=-1)), dim=-1)

    def sin(self, a):
        """
        Element-wise sine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.sin(a)

    def cos(self, a):
        """
        Element-wise cosine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.cos(a)

    def tan(self, a):
        """
        Element-wise tangent

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.tan(a)

    def exp(self, a):
        """
        Element-wise exponential

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.exp(a)

    def log(self, a):
        """
        Element-wise logarithm

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.log(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.sqrt(a)

    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.cbrt(a)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: np.ndarray of floats
        :param b: np.ndarray or float

        Warning: Negative values raised to a non-integral value will return nan.

        :return: np.ndarray of floats
        """
        return np.power(a, b)

    def min(self, a, dim=None):
        """
        Return the minimum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return np.min(a, axis=dim)

    def max(self, a, dim=None):
        """
        Return the maximum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return np.max(a, axis=dim)

    def minimum(self, a, b):
        """
        Element-wise minimum of two numpy arrays

        :param a: np.ndarray of floats
        :param b: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.minimum(a, b)

    def maximum(self, a, b):
        """
        Element-wise maximum of two numpy arrays

        :param a: np.ndarray of floats
        :param b: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.maximum(a, b)

    def clamp(self, a, min=None, max=None):
        """
        Clamp array values

        :param a: np.ndarray
        :param min: minimum value
        :param max: maximum value

        :return: np.ndarray
        """
        return np.clip(a, a_min=min, a_max=max)

    def all(self, a, dim=None):
        """
        Check if all elements are True along a dimension

        :param a: np.ndarray of booleans

        :return: bool
        """
        return np.all(a, axis=dim)

    def any(self, a, dim=None):
        """
        Check if any elements are True along a dimension

        :param a: np.ndarray of booleans

        :return: bool
        """
        return np.any(a, axis=dim)

    def stack(self, a, dim=0):
        """
        Stack a list of numpy arrays vertically

        :param a: list of np.ndarray

        :return: np.ndarray
        """
        return np.stack(a, axis=dim)

    def cat(self, a, dim=0):
        """
        Concatenate a list of numpy arrays along the first axis
        :param a: list of np.ndarray
        :return: np.ndarray
        """
        return np.concatenate(a, axis=dim)

    def hstack(self, a):
        """
        Stack a list of numpy arrays horizontally

        :param a: list of np.ndarray

        :return: np.ndarray
        """
        return np.column_stack(a)

    def to_numpy(self, a):
        """
        Convert to a numpy array (no-op since already numpy)

        :param a: np.ndarray

        :return: np.ndarray
        """
        return a

    def to_format(self, a):
        """
        Convert to numpy format (no-op if already numpy, otherwise convert)

        :param a: array-like

        :return: np.ndarray
        """
        return np.asarray(a, dtype=self.dtype)

    def einsum(self, equation, *operands):
        """
        Einstein summation convention

        :param equation: str
        :param operands: np.ndarray arguments

        :return: np.ndarray
        """
        return np.einsum(equation, *operands)

    def zeros(self, shape):
        """
        Create an array filled with zeros

        :param shape: array shape

        :return: np.ndarray
        """
        return np.zeros(shape, dtype=self.dtype)

    def ones(self, shape):
        """
        Create an array filled with ones

        :param shape: array shape

        :return: np.ndarray
        """
        return np.ones(shape, dtype=self.dtype)

    def eye(self, n):
        """
        Create an identity matrix

        :param n: size of the identity matrix

        :return: np.ndarray
        """
        return np.eye(n, dtype=self.dtype)

    def zeros_like(self, a):
        """
        Create an array of zeros with the same shape as a

        :param a: np.ndarray

        :return: np.ndarray
        """
        return np.zeros_like(a, dtype=self.dtype)

    def ones_like(self, a):
        """
        Create an array of ones with the same shape as a

        :param a: np.ndarray

        :return: np.ndarray
        """
        return np.ones_like(a, dtype=self.dtype)

    def full_like(self, a, fill_value):
        """
        Create an array filled with fill_value with the same shape as a

        :param a: np.ndarray
        :param fill_value: value to fill

        :return: np.ndarray
        """
        return np.full_like(a, fill_value, dtype=self.dtype)

    def eye_like(self, x):
        """
        Create an identity matrix

        :param x: np.ndarray of floats [..., n]

        :return: np.ndarray of floats [n, n]
        """
        return np.eye(x.shape[-1])

    def atleast_1d(self, a):
        """
        Convert input to at least 1D array

        :param a: np.ndarray

        :return: np.ndarray
        """
        return np.atleast_1d(a)

    def abs(self, a):
        """
        Element-wise absolute value

        :param a: np.ndarray

        :return: np.ndarray
        """
        return np.abs(a)

    def clamp(self, a, min=None, max=None):
        """
        Clamp array values

        :param a: np.ndarray
        :param min: minimum value
        :param max: maximum value

        :return: np.ndarray
        """
        return np.clip(a, a_min=min, a_max=max)

    def where(self, condition, x, y):
        """
        Select elements from x or y based on condition

        :param condition: boolean array
        :param x: array for True values
        :param y: array for False values

        :return: np.ndarray
        """
        return np.where(condition, x, y)

    def unsqueeze(self, a, dim):
        """
        Add a dimension to array (simulate torch.unsqueeze)

        :param a: np.ndarray
        :param dim: dimension to add

        :return: np.ndarray
        """
        return np.expand_dims(a, axis=dim)

    def squeeze(self, a, dim=None):
        """
        Remove dimension from array

        :param a: np.ndarray
        :param dim: dimension to remove (None for all size-1 dims)

        :return: np.ndarray
        """
        if dim is None:
            return np.squeeze(a)
        return np.squeeze(a, axis=dim)

    def expand(self, a, sizes, dim=None):
        """
        Expand array to new sizes along specified dimension

        :param a: np.ndarray
        :param sizes: new sizes
        :param dim: dimension to expand

        :return: np.ndarray
        """
        shape = list(a.shape)
        shape[dim] = sizes
        return np.broadcast_to(a, shape)

    def flatten(self, a):
        """
        Flatten array

        :param a: np.ndarray

        :return: np.ndarray
        """
        return a.flatten()

    def diag(self, a):
        """
        Create diagonal matrix from vector

        :param a: np.ndarray (1D)

        :return: np.ndarray (2D diagonal matrix)
        """
        return np.diag(a)

    def transpose(self, a):
        """
        Transpose array

        :param a: np.ndarray

        :return: np.ndarray
        """
        indices = [i for i in range(a.ndim)]  # just to emphasize ndim usage
        indices[-1], indices[-2] = indices[-2], indices[-1]
        return np.transpose(a, axes=indices)

    def sum(self, a, dim=None):
        """
        Sum array along specified axis

        :param a: np.ndarray
        :param dim: axis to sum over (None for all)

        :return: np.ndarray
        """
        return np.sum(a, axis=dim)

    def prod(self, a, dim=None):
        """
        Product of array elements along specified axis

        :param a: np.ndarray
        :param dim: axis to compute product over (None for all)

        :return: np.ndarray
        """
        return np.prod(a, axis=dim)

    def mean(self, a, dim=None):
        """
        Mean of array elements along specified axis

        :param a: np.ndarray
        :param axis: axis to compute mean over (None for all)

        :return: np.ndarray
        """
        return np.mean(a, axis=dim)

    def ceil(self, a):
        """
        Element-wise ceiling

        :param a: np.ndarray

        :return: np.ndarray
        """
        return np.ceil(a)

    def floor(self, a):
        """
        Element-wise floor

        :param a: np.ndarray

        :return: np.ndarray
        """
        return np.floor(a)

    def allclose(self, a, b, tol=1e-5):
        """
        Check if two arrays are element-wise equal within a tolerance

        :param a: np.ndarray
        :param b: np.ndarray
        :param tol: tolerance

        :return: bool
        """
        return np.allclose(a, b, atol=tol)

    def equal(self, a, b):
        """
        Check element-wise equality

        :param a: np.ndarray
        :param b: np.ndarray

        :return: np.ndarray of booleans
        """
        return np.array_equal(a, b)
