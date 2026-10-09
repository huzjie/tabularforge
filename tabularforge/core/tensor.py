"""A tiny autodiff tensor (educational reference).

The cpu backend and TabFM model deliberately use plain Python lists/floats for
speed, so this module is optional -- it exists to document how a scalar-valued
autodiff over the op set would look, and powers a small end-to-end demo.
"""
import math


class Tensor:
    __slots__ = ("data", "grad", "_op", "_children")

    def __init__(self, data, _op=None, _children=()):
        self.data = data
        self.grad = 0.0
        self._op = _op
        self._children = _children

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, "add", (self, other))

        def _back():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _back
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, "mul", (self, other))

        def _back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _back
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def relu(self):
        out = Tensor(max(0.0, self.data), "relu", (self,))

        def _back():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _back
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Tensor(s, "sigmoid", (self,))

        def _back():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _back
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Tensor(t, "tanh", (self,))

        def _back():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _back
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(n):
            if id(n) not in visited:
                visited.add(id(n))
                for c in getattr(n, "_children", ()):
                    build(c)
                topo.append(n)
        build(self)
        self.grad = 1.0
        for n in reversed(topo):
            if hasattr(n, "_backward"):
                n._backward()

    def __repr__(self):
        return f"Tensor({self.data:.4f})"
