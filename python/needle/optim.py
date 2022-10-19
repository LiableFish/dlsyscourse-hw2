"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            u = self.u.setdefault(param, 0)

            grad = param.grad.data + self.weight_decay * param.data

            self.u[param] = self.momentum * u + (1 - self.momentum) * grad

            param.data = param.data - self.lr * self.u[param]


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for param in self.params:
            m = self.m.setdefault(param, 0)
            v = self.v.setdefault(param, 0)

            grad = param.grad.data + self.weight_decay * param.data

            self.m[param] = self.beta1 * m + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * v + (1 - self.beta2) * grad ** 2

            m_corrected = self.m[param] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[param] / (1 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m_corrected / (v_corrected ** 0.5 + self.eps)
