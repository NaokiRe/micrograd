import math
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: 'Value') -> 'Value':
        return self + other
    
    def __neg__(self) -> 'Value':
        return self * -1
    
    def __mul__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other: 'Value') -> 'Value':
        return self * other
    
    def __pow__(self, power) -> 'Value':
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self, ), f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: 'Value') -> 'Value':
        return self * other**-1

    def __sub__(self, other: 'Value') -> 'Value':
       return self + (-other) 
    
    def exp(self) -> 'Value':
        out =  Value(math.exp(self.data), (self,), 'exp', label='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    
    def tanh(self) -> 'Value':
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh', label='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        self.grad = 1.0

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for v in reversed(topo):
            v._backward()

