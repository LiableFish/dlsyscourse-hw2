"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy as np

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return self.scalar * (node.inputs[0] ** (self.scalar - 1)) * out_grad


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            axis1, axis2 = range(a.ndim)[-2:]
        else:
            axis1, axis2 = self.axes

        return array_api.swapaxes(a, axis1, axis2)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.transpose(axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return a.reshape(self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.reshape(shape=node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    @staticmethod
    def _is_broadcast(input_dim: Optional[int], required_dim: int):
        """
        Define if input_dim would be broadcast by comparing with required_dim
        according to https://numpy.org/doc/stable/user/basics.broadcasting.html
        """
        return (input_dim == 1 and input_dim != required_dim) or input_dim is None

    def compute(self, a: NDArray):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        if input_shape == self.shape:
            # no broadcasting <-> identity op
            return out_grad

        broadcast_axes = []
        for i, (input_dim, required_dim) in enumerate(zip_longest(input_shape, self.shape)):
            if self._is_broadcast(input_dim, required_dim):
                broadcast_axes.append(i)

        return out_grad.sum(axes=tuple(broadcast_axes)).reshape(input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        axes = range(len(input_shape)) if self.axes is None else list(self.axes)

        # like when calling np.sum(..., keepdims=True)
        keepdims_shape = np.array(input_shape)
        keepdims_shape[axes] = 1

        return out_grad.reshape(keepdims_shape).broadcast_to(input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        left_grad = out_grad @ rhs.transpose()
        right_grad = lhs.transpose() @ out_grad

        # in case N-D, N > 2 we need to sum along broadcast axes
        # to ensure that shape of grad equals to shape of input
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

        left_broadcast_axes = tuple(range(len(left_grad.shape) - len(lhs.shape)))
        right_broadcast_axes = tuple(range(len(right_grad.shape) - len(rhs.shape)))

        if left_broadcast_axes:
            left_grad = left_grad.sum(left_broadcast_axes)

        if right_broadcast_axes:
            right_grad = right_grad.sum(right_broadcast_axes)

        return left_grad, right_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * Tensor.make_const(np.where(node.realize_cached_data() > 0, 1, 0))


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
