"""The module.
"""
from re import A
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight_init_data = init.kaiming_uniform(
            fan_in=self.in_features,
            fan_out=self.out_features,
        )

        self.weight = Parameter(
            weight_init_data,
            requires_grad=True, 
            device=device,
            dtype=dtype,
        )

        self.bias = None
        if bias:
            bias_init_data = (
                init.kaiming_uniform(fan_in=self.out_features, fan_out=1)
                .reshape((1, self.out_features))
            )
            
            self.bias = Parameter(
                bias_init_data,
                requires_grad=True, 
                device=device,
                dtype=dtype,
            )



    def forward(self, X: Tensor) -> Tensor:
        mul = X @ self.weight
        if self.bias:
            mul += self.bias.broadcast_to(mul.shape)
        
        return mul
       



class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.reshape((X.shape[0], np.prod(X.shape[1:])))


class ReLU(Module):
    def forward(self, X: Tensor) -> Tensor:
        return ops.relu(X)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, X: Tensor) -> Tensor:
        res = X
        for module in self.modules:
            res = module(res)

        return res


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        one_hot = init.one_hot(n=logits.shape[1], i=y, device=y.device)
        return (ops.logsumexp(logits, axes=(1,)) - (logits * one_hot).sum((1,))).sum() / logits.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(
            init.ones(dim),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        self.bias = Parameter(
            init.zeros(dim),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        self.running_mean = init.zeros(
            dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )

        self.running_var = init.ones(
            dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )


    def forward(self, X: Tensor) -> Tensor:
        assert self.dim == X.shape[1]
        
        if self.training:
            mean = X.sum((0,)) / X.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data

            mean = mean.reshape((1, X.shape[1])).broadcast_to(X.shape)

            var = ((X - mean) ** 2).sum((0,)) / X.shape[0]
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            var = var.reshape((1, X.shape[1])).broadcast_to(X.shape)

        else:
            mean = self.running_mean.reshape((1, X.shape[1])).broadcast_to(X.shape)
            var = self.running_var.reshape((1, X.shape[1])).broadcast_to(X.shape)

        normalized_X = (X - mean) / (var + self.eps) ** 0.5

        weight = self.weight.reshape((1, self.dim)).broadcast_to(X.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(X.shape)

        return weight * normalized_X + bias



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.weight = Parameter(
            init.ones(dim),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        self.bias = Parameter(
            init.zeros(dim),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )


    def forward(self, X: Tensor) -> Tensor:
        assert self.dim == X.shape[1]

        mean = X.sum((1,)) / X.shape[1]
        mean = mean.reshape((X.shape[0], 1)).broadcast_to(X.shape)

        var = ((X - mean) ** 2).sum((1,)) / X.shape[1]
        var = var.reshape((X.shape[0], 1)).broadcast_to(X.shape)

        normalized_X = (X - mean) / (var + self.eps) ** 0.5

        weight = self.weight.reshape((1, self.dim)).broadcast_to(X.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(X.shape)

        return weight * normalized_X + bias


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, X: Tensor) -> Tensor:
        if not self.training:
           return X
        
        dropout_probs = init.randb(*X.shape, p=1-self.p, device=X.device)
        return X * dropout_probs / (1 - self.p)
        

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, X: Tensor) -> Tensor:
        return X + self.fn(X)
