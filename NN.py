import numpy as np
import math

class Variable:
    """basic variable to store a scalar, perform operations and calculate gradients"""
    def __init__(self, value, _children=()): 
        self.value = value
        self.grad = 0
        self.prev = set(_children)          # children in the reverse computation graph
        self._backward = lambda : None
    
    def __repr__(self):
        """ current state of the object """
        return f"value = {self.value}\ngrad = {self.grad}"
    
    def __str__(self):
        """ Readable repesentation of the object """
        return f"Variable(value={self.value})"
    
   
    def __neg__(self):                      # self * -1
        return self * -1
    
    # fundamental method
    def __pow__(self, power):               # self ^ power
        assert isinstance(power, (int,float))
        out = Variable(self.value**power, (self,))   

        def _backward():
            self.grad += power*(self.value**(power-1)) * out.grad
        
        out._backward = _backward
        return out

    # fundamental method
    def __add__(self, other):               # self + other
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        out = Variable((self.value + other.value), (self, other,))   # returns Variable object

        # closure function , remembers the variables from the scope where it was created
        def _backward():
            self.grad += 1.0 *out.grad
            other.grad += 1.0 *out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        return Variable(self.value + other.value)   # returns Variable object
    
    def __sub__(self, other):               # self + (-other)
        return self + (-other)
    
    def __rsub__(self, other):               # other + (-self) 
        return -self + other
    
    # fundamental method
    def __mul__(self, other):               # self * other
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        out = Variable(self.value * other.value, (self, other,))   # returns Variable object

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):               # other * self
        if not isinstance(other, Variable):
            other = Variable(other)             # only supports numerical value
        out =  Variable(self.value * other.value, (self, other))   # returns Variable object
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)
    
    def exp(self):
        out = Variable(math.exp(self.value), (self,))

        def _backward():
            self.grad += out.value * out.grad     # y = e^x  dy/dx = e^x (-> out.value)  out.grad -> out's grad
        out._backward = _backward
        return out

    def sigmoid(self):
        # e^x/1+e^x
        t = self.exp().value
        d = 1 + t
        out = Variable(t/d, (self,))

        def _backward():
            self.grad += out.grad * out.value*(1-out.value)
        out._backward = _backward
        return out

    def relu(self):
        out = Variable(max(0, self.value), (self,))

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward

        return out


    def backprop(self):
        # backprop from self to all the variables in computation graph
        # using topological sort

        comp_graph = []
        self.grad = 1
        visited = set()
        def topo(z):
            if z not in visited:
                comp_graph.append(z)
                visited.add(z)
                for child in z.prev:
                    topo(child)
        topo(self)
        
        for node in comp_graph:
            node._backward()


class Neuron:
    def __init__(self, n_ip, act = 'linear'):
        self.w = [Variable(np.random.rand() ) for _ in range(n_ip)]
        self.b = Variable(np.random.rand())
        self.act = act

    def __call__(self, inputs):
        # act = ('linear', 'sigmoid', 'relu')
        # x1.w1 + x2w2 + x3w3 ...... + b
        out = 0
        for xi , wi in zip(self.w, inputs):
            out = out+ xi*wi
        print(type(out))
        print(type(self.b))
        out = out + self.b

        if(self.act == 'sigmoid'):
            out = out.sigmoid()
        elif(self.act == 'relu'):
            out = out.relu()
        
        return out
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.act} Neuron({len(self.w)})"

        
class Layer:
    # defines a dense connected layer
    def __init__(self, n_in, n_neurons, act = 'linear'):
        self.neurons = [Neuron(n_in, act=act) for _ in range(n_neurons)]
    
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


            
class MLP:
    def __init__(self, n_in, layers):
        layers = [n_in] + layers
        self.layers = [Layer(layers[i], layers[i+1], act= ('relu' if i!= len(layers)-2 else 'sigmoid')) for i in range(len(layers)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def fit(self,x, y, epochs = 1, lr = 0.1):

        for epoch in range(epochs):
            for i in range(len(x)):
                y_pred = self.__call__(x[i])[0]
                y_true = y[i]

                loss = (y_pred - y_true)**2
                if i == 0 :
                    epoch_loss = loss
                else:
                    epoch_loss = epoch_loss + loss
                

            epoch_loss.backprop()
            for p in self.parameters():
                p.value = p.value - lr*p.grad
            print(f"Epoch {epoch} Loss = {epoch_loss}")

            
            
            self.zero_grad()

        
            
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"