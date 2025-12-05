import numpy as np

# activation functions

class Linear:
    def forward(self, inputs, training = True):
        self.output = inputs
    
    def backward(self, prev_grads):
        self.dinputs =  prev_grads.copy()
    
    def predict(self, outputs):
        return outputs

class ReLU:
    def __call__(self, inputs ):
        self.output = np.maximum(0, inputs)
        return self.output
    
    def forward(self, inputs ,training = True):
        # relu = 0 for x < 0
        #      = x for x >= 0
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, prev_grads):
        #prev_grads shape = (m , self.n_neurons)
        #self.outputs shape = (m , self.n_neurons)
        self.dinputs = prev_grads.copy()
        # dinputs = self.output * prev_grad         ---- element wise multiplication not dot product
        self.dinputs[self.output <= 0] = 0 
    
    def predict(self, outputs):
        return outputs

class Sigmoid:
    def __call__(self, inputs):
        exp = np.exp(inputs)
        self.output = exp/(exp + 1)
        return self.output

    def forward(self, inputs, training = True):
        # sigmoid(x) = 1/1+e^-x = e^x / 1 + e^x
        exp = np.exp(inputs)
        self.output = exp/(exp + 1)
        return self.output
    
    def backward(self, prev_grads):
        # sigmoid(x) = e^x / 1 + e^x
        # sigmoid'(x) = e^x(1 + e^x) * 1/1+e^x
        #             = sigmoid(x) * (1 - sigmoid(x))

        # prev_grads shape = (m , self.n_neurons)
        # self.output shape = (m , self.n_neurons)
        self.dinputs = (self.output*(1-self.output)) * prev_grads   # element-wise multiplication
    
    def predict(self, outputs):
        return (outputs > 0.5)*1

class Softmax:
    def __call__(self, inputs):
        # sofmax(xi) = e^-
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # to prevent exploding gradients by preventing overflow from exponent function
        norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values
        return self.output
    
    def forward(self, inputs, training = True):
        # sofmax(xi) = e^-xi / sum(e^xj) j = 1 to n_neurons
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # to prevent exploding gradients by preventing overflow from exponent function
        norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values
        return self.output
    
    def backward(self, prev_grads):
        # sofmax(xi) = e^-xi / sum(e^xj) j = 1 to n_neurons
        # dsoftmax(xi)/dxi = softmax(xi)*(1-sofmax(xi))
        # dsoftmax(xi)/dxj = -softmax(xi)*softmax(xj)

        # dsoftmax(xi)dxj = softmax(xi)*(1-sofmax(xi))   i = j
        #                 = -softmax(xi)*softmax(xj)    i != j
        # Kronecker delta function dij = 1 if i = j 
        #                              = 0 i!=j
        # dsoftmax(xi)/dxj = softmax(xi)(dij - softmax(xj))
        #                  = softmax(xi)dij - softmax(xi)softmax(xj)

        # but xi = ipi - max
        # dxi = 1

        # prev_grads shape = (m , n_classes)
        # self.output shape = (m, n_classes)
        self.dinputs = np.empty_like(prev_grads)

        # enumerate outputs and gradients
        for index, (single_output, single_grad) in enumerate(zip(self.output, prev_grads)):
            # flatten output array
            single_output = single_output.reshape(-1,1) # col matrix
            # Jacobian matrix is an array of partial derivatives in all of the combinations of both input vectors.
            # Jacobian matrix of a vector-valued function of several variables is the matrix of all its first-order partial derivatives
            # calculate jacobian matrix of the output s
            # comes from jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # calculate sample-wise gradient
            # and add it to the array of sample gradients
            # here single grad is shape (1, )
            # numpy treats it as column vector when it is on the left of dot product
            self.dinputs[index] = np.dot(jacobian_matrix, single_grad)  # shape(n_classes, 1) but numpy gives shape (1, ) which will be row vector in out context
    
    def predict(self, outputs):
        return np.argmax(outputs, axis=1)
