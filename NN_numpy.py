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
         
    def backward(self, prev_grads):
        self.dweights = np.dot(self.inputs.T, prev_grads)       # inputs.T = (no. of features, no. of samples)
        self.dbiases = np.sum(prev_grads, axis=0, keepdims=True)    # sum over columns (i.e wrt to each neuron)
        # gradients on inputs
        # each dinputi will be sum over dai*wi for all neurons, da is upstream gradient
        # upstream grad shape = (m , self.n_neurons) 
        # self.weights.T shape = (self.n_neurons, self.n_in)
        # dinputs shape = (m, self.n_in)    [n_in are the n_neurons for prev layer]
        self.dinputs = np.dot(prev_grads, self.weights.T)   # required to propogate gradients

    def __repr__(self):
        return f"Dense Layer ({self.weights.shape[0]} -> {self.weights.shape[1]})"
    
    def __str__(self):
        return self.__repr__()

# activation functions
class ReLU:
    def __call__(self, inputs ):
        self.output = np.maximum(0, inputs)
        return self.output
    
    def forward(self, inputs ):
        # relu = 0 for x < 0
        #      = x for x >= 0
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, prev_grads):
        #prev_grads shape = (m , self.n_neurons)
        #self.outputs shape = (m , self.n_neurons)
        self.dinputs = prev_grads.copy()
        # dinputs = self.output * prev_grad         ---- element wise multiplication not dot product
        self.dinputs[prev_grads <= 0] = 0 

class Sigmoid:
    def __call__(self, inputs):
        exp = np.exp(inputs)
        self.output = exp/(exp + 1)
        return self.output

    def forward(self, inputs):
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

class Softmax:
    def __call__(self, inputs):
        # sofmax(xi) = e^-
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # to prevent exploding gradients by preventing overflow from exponent function
        norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values
        return self.output
    
    def forward(self, inputs):
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



# loss functions
class Loss:

    # calculates the data and regularization losses
    # given model output and ground truth values

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # to prevent division by zero(log(0))
        clipped_output = np.clip(y_pred, 1e-07, 1-1e-07)

        # Probablities for target values -
        # only if categorical labels    i.e true class index for each sample
        if len(y_true.shape) == 1:
            correct_confidences = clipped_output[range(samples), y_true]    # y_true only contains the incide of true class
        elif len(y_true.shape) ==2:     # contains one-hot encoded labels OR true probability distribution
            # mask the valus
            correct_confidences = np.sum(clipped_output*y_true, axis=1)
        
        #losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)       # number of samples m
        labels = len(y_pred[0])     # number of classes c

        # prev_grad shape = (m, c)

        # if labels are sparse convert to one hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]         #[y_true] reorders the values of I to match the hot encoding
        
        self.dinputs = -y_true / y_pred

        # normalization of gradients
        self.dinputs = self.dinputs/samples
  

# regularization
class Dropout:
    pass

class BatchNorm:
    pass
