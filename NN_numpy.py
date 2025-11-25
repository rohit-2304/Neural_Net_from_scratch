import numpy as np

# dense layer
class Dense:
    def __init__(self, n_in, n_neurons):
        self.weights = 0.01 * np.random.randn(n_in , n_neurons)   # w^T shape
        self.baises = np.zeros((1, n_neurons))                   # b^T shape

    def __call__(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises
        return self.output
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.baises
        return self.output
         
    def backward(self):
        self.grad = self.inputs




    def __repr__(self):
        return f"Dense Layer ({self.weights.shape[0]} -> {self.weights.shape[1]})"
    
    def __str__(self):
        return self.__repr__()

# activation functions
class ReLU:
    def __call__(self, inputs ):
        self.output = np.maximum(0, inputs)
        return self.output

class Sigmoid:
    def __call__(self, inputs):
        exp = np.exp(inputs)
        self.output = exp/(exp + 1)
        return self.output

class Softmax:
    def __call__(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # to prevent exploding gradients
        norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values
        return self.output

# loss functions
  

# regularization
class Dropout:
    pass

class BatchNorm:
    pass
