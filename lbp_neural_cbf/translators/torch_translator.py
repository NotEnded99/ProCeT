import torch


class TorchTranslator:
    def __init__(self, device=None, dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: torch.tensor of floats [n, m]
        :param b: torch.tensor of floats [m]

        :return: torch.tensor of floats [n]
        """
        return torch.matmul(a, b.unsqueeze(-1)).squeeze(-1)

    def sin(self, a):
        """
        Element-wise sine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.sin(a)

    def cos(self, a):
        """
        Element-wise cosine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.cos(a)

    def tan(self, a):
        """
        Element-wise tangent

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.tan(a)

    def exp(self, a):
        """
        Element-wise exponential

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.exp(a)

    def log(self, a):
        """
        Element-wise logarithm

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.log(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.sqrt(a)

    def cbrt(self, a):
        """
        Element-wise _real_ cube root

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.copysign(torch.pow(a.abs(), 1 / 3), a)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: torch.tensor of floats
        :param b: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.pow(a, b)

    def min(self, a, dim=None):
        """
        Return the minimum value of a torch tensor

        :param a: torch.tensor of floats

        :return: float or torch.tensor if dim is specified
        """
        return torch.min(a, dim=dim).values if dim is not None else torch.min(a)

    def max(self, a, dim=None):
        """
        Return the maximum value of a torch tensor

        :param a: torch.tensor of floats

        :return: float
        """
        return torch.max(a, dim=dim).values if dim is not None else torch.max(a)

    def minimum(self, a, b):
        """
        Element-wise minimum of two tensors

        :param a: torch.tensor of floats
        :param b: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.minimum(a, b)

    def maximum(self, a, b):
        """
        Element-wise maximum of two tensors

        :param a: torch.tensor of floats
        :param b: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.maximum(a, b)

    def clamp(self, a, min=None, max=None):
        """
        Clamp tensor values

        :param a: torch.tensor
        :param min: minimum value
        :param max: maximum value

        :return: torch.tensor
        """
        return torch.clamp(a, min=min, max=max)

    def all(self, condition, dim=None):
        """
        Check if all elements in the tensor are True

        :param condition: torch.tensor of booleans

        :return: bool
        """
        return torch.all(condition, dim=dim)

    def any(self, condition, dim=None):
        """
        Check if any element in the tensor is True

        :param condition: torch.tensor of booleans

        :return: bool
        """
        return torch.any(condition, dim=dim)

    def stack(self, a, dim=0):
        """
        Stack a list of torch tensors vertically

        :param a: list of torch tensors

        :return: torch.tensor
        """
        return torch.stack(a, dim=dim)

    def cat(self, a, dim=0):
        """
        Stack a list of torch tensors vertically

        :param a: list of torch tensors

        :return: self.torch.tensor
        """
        return torch.cat(a, dim=dim)

    def hstack(self, a):
        """
        Stack a list of torch tensors horizontally

        :param a: list of torch tensors

        :return: torch.tensor
        """
        return torch.stack(a, dim=1)

    def to_numpy(self, a):
        """
        Convert a tensor to a numpy array

        :param a: torch.tensor

        :return: np.array
        """
        return a.detach().cpu().numpy()

    def to_format(self, a):
        """
        Convert a numpy array to a tensor

        :param a: np.array

        :return: torch.tensor
        """
        return torch.as_tensor(a, dtype=self.dtype, device=self.device)

    def einsum(self, equation, *operands):
        """
        Einstein summation convention

        :param equation: str
        :param operands: torch.tensor arguments

        :return: torch.tensor
        """
        return torch.einsum(equation, *operands)

    def zeros(self, shape):
        """
        Create a tensor filled with zeros

        :param shape: tensor shape
        :param dtype: data type (default: float64)

        :return: torch.tensor
        """
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def ones(self, shape):
        """
        Create a tensor filled with ones

        :param shape: tensor shape
        :param dtype: data type (default: float64)

        :return: torch.tensor
        """
        return torch.ones(shape, dtype=self.dtype, device=self.device)

    def eye(self, n):
        """
        Create an identity matrix

        :param n: size of the identity matrix
        :param dtype: data type (default: float64)

        :return: torch.tensor
        """
        return torch.eye(n, dtype=self.dtype, device=self.device)

    def zeros_like(self, a):
        """
        Create a tensor of zeros with the same shape as a

        :param a: torch.tensor

        :return: torch.tensor
        """
        return torch.zeros_like(a, dtype=self.dtype, device=self.device)

    def ones_like(self, a):
        """
        Create a tensor of ones with the same shape as a

        :param a: torch.tensor

        :return: torch.tensor
        """
        return torch.ones_like(a, dtype=self.dtype, device=self.device)

    def full_like(self, a, fill_value):
        """
        Create a tensor filled with fill_value with the same shape as a

        :param a: torch.tensor
        :param fill_value: value to fill

        :return: torch.tensor
        """
        return torch.full_like(a, fill_value, dtype=self.dtype, device=self.device)

    def eye_like(self, x):
        """
        Create an identity matrix

        :param x: torch.tensor of floats [..., n]

        :return: torch.tensor of floats [n, n]
        """
        return torch.eye(x.size(-1), dtype=self.dtype, device=self.device)

    def atleast_1d(self, a):
        """
        Ensure tensor is at least 1-dimensional

        :param a: torch.tensor

        :return: torch.tensor
        """
        return torch.atleast_1d(a)

    def abs(self, a):
        """
        Element-wise absolute value

        :param a: torch.tensor

        :return: torch.tensor
        """
        return torch.abs(a)

    def clamp(self, a, min=None, max=None):
        """
        Clamp tensor values

        :param a: torch.tensor
        :param min: minimum value
        :param max: maximum value

        :return: torch.tensor
        """
        return torch.clamp(a, min=min, max=max)

    def where(self, condition, x, y):
        """
        Select elements from x or y based on condition

        :param condition: boolean tensor
        :param x: tensor for True values
        :param y: tensor for False values

        :return: torch.tensor
        """
        return torch.where(condition, x, y)

    def unsqueeze(self, a, dim):
        """
        Add a dimension to tensor

        :param a: torch.tensor
        :param dim: dimension to add

        :return: torch.tensor
        """
        return a.unsqueeze(dim)

    def squeeze(self, a, dim=None):
        """
        Remove dimension from tensor

        :param a: torch.tensor
        :param dim: dimension to remove (None for all size-1 dims)

        :return: torch.tensor
        """
        if dim is None:
            return a.squeeze()
        return a.squeeze(dim)

    def expand(self, a, sizes, dim=None):
        """
        Repeat tensor along specified dimension

        :param a: torch.tensor
        :param sizes: number of repeats
        :param dim: dimension to repeat along

        :return: torch.tensor
        """
        if dim is None:
            return a.expand(*sizes)

        shape = list(a.shape)
        shape[dim] = sizes

        return a.expand(*shape)

    def flatten(self, a):
        """
        Flatten tensor

        :param a: torch.tensor

        :return: torch.tensor
        """
        return a.flatten()

    def diag(self, a):
        """
        Create diagonal matrix from vector

        :param a: torch.tensor (1D)

        :return: torch.tensor (2D diagonal matrix)
        """
        return torch.diag(a)

    def transpose(self, a):
        """
        Transpose tensor

        :param a: torch.tensor

        :return: torch.tensor
        """
        return torch.transpose(a, -2, -1)

    def sum(self, a, dim=None):
        """
        Sum tensor along specified dimension

        :param a: torch.tensor
        :param dim: dimension to sum over (None for all)

        :return: torch.tensor
        """
        return torch.sum(a, dim=dim)

    def prod(self, a, dim=None):
        """
        Product of tensor elements along specified dimension

        :param a: torch.tensor
        :param dim: dimension to compute product over (None for all)

        :return: torch.tensor
        """
        return torch.prod(a, dim=dim)

    def mean(self, a, dim=None):
        """
        Mean of tensor elements along specified dimension

        :param a: torch.tensor
        :param dim: dimension to compute mean over (None for all)

        :return: torch.tensor
        """
        return torch.mean(a, dim=dim)

    def ceil(self, a):
        """
        Element-wise ceiling

        :param a: torch.tensor

        :return: torch.tensor
        """
        return torch.ceil(a)

    def floor(self, a):
        """
        Element-wise floor

        :param a: torch.tensor

        :return: torch.tensor
        """
        return torch.floor(a)

    def allclose(self, a, b, rtol=1e-05, atol=1e-08):
        """
        Check if two tensors are element-wise equal within a tolerance

        :param a: torch.tensor
        :param b: torch.tensor
        :param rtol: relative tolerance
        :param atol: absolute tolerance

        :return: bool
        """
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    def equal(self, a, b):
        """
        Check if two tensors are exactly equal

        :param a: torch.tensor
        :param b: torch.tensor

        :return: bool
        """
        return torch.equal(a, b)
