import numpy as np

# dense layer
class Dense:
    def __init__(self, n_in, n_neurons, weight_regularizer_l1=0, bias_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l2=0):
        upper = (np.sqrt(6/(n_in + n_neurons)))
        lower = -upper
        # Xavier Initialization
        self.weights = np.random.uniform(lower, upper, (n_in, n_neurons))    # w^T shape
        self.biases = np.zeros((1, n_neurons))              # b^T shape

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2

        self.trainable = True

    def __call__(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def forward(self, inputs, training = True):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
         
    def backward(self, prev_grads):
        self.dweights = np.dot(self.inputs.T, prev_grads)       # inputs.T = (no. of features, no. of samples)
        self.dbiases = np.sum(prev_grads, axis=0, keepdims=True)    # sum over columns (i.e wrt to each neuron)
        # gradients on inputs
        # each dinputi will be sum over dai*wi for all neurons, da is upstream gradient
        # upstream grad shape = (m , self.n_neurons) 
        # self.weights.T shape = (self.n_neurons, self.n_in)
        # dinputs shape = (m, self.n_in)    [n_in are the n_neurons for prev layer]
        self.dinputs = np.dot(prev_grads, self.weights.T)   # required to propogate gradients


        # if l1 regularization used
        if self.weight_regularizer_l1 > 0:
            l1_dweights = np.ones_like(self.weights)
            l1_dweights[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * l1_dweights

        if self.bias_regularizer_l1 > 0:
            l1_dbias = np.ones_like(self.biases)
            l1_dbias[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * l1_dbias

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    def __repr__(self):
        return f"Dense Layer ({self.weights.shape[0]} -> {self.weights.shape[1]})"
    
    def __str__(self):
        return self.__repr__()
