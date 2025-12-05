import numpy as np

# regularization
# l1 and l2 regularization already added in dense layer and loss function
class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = 1 - dropout_rate    # stored as inverted rate (ratio of neurons to keep)
    
    def forward(self, inputs,training = True):
        if not training :
            self.output = inputs
            return
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.dropout_rate, inputs.shape)/self.dropout_rate * self.inputs
        self.output = self.inputs * self.binary_mask

    def backward(self, prev_grads):
        self.dinputs = prev_grads  * self.binary_mask

class BatchNorm:
    # normalizes data to 0 mean and unit std deviation
    # adds regulariztion
    # allows for more aggresive learning rates
    def __init__(self, epsilon = 1e-07, ):
        self.epsilon = epsilon
        self.gamma = None       # scaling parameter learnable parameter
        self.beta = None        # shift parameter   both of them neccessary to regain the lost dimensions
        self.trainable = True
    
    def forward(self, inputs,training = False):
        if not training:
            self.output = inputs
            return
        if self.gamma is None:
            _, features = inputs.shape
            self.gamma = np.ones((1, features))
            self.beta  = np.zeros((1, features))

        self.inputs = inputs
        self.samples = len(input)

        self.mean = np.sum(inputs, axis=0,keepdims=True) / self.samples                     # shape( 1,  n_neurons)
        self.var = np.sum((inputs - self.mean)**2, axis=0,keepdims=True)/ self.samples      # shape (1, n_neurons)

        self.x_mu = inputs - self.mean                              # shape (m, n_neurons)
        self.std_inv = 1.0 / np.sqrt(self.var + self.epsilon)       # shape(1, n_neurons)
        self.x_hat = self.x_mu * self.std_inv                       # shape (m, n_neurons)      broadcasting done

        self.output = self.gamma * self.x_hat + self.beta   # shape (m, n_neurons)
        return self.output

    def backward(self, prev_grads):
        self.dgamma = np.sum(self.x_hat * prev_grads, axis = 0, keepdims = True)    #shape (1, n_neurons)
        self.dbeta  = np.sum(prev_grads, axis=0, keepdims=True)                     #shape (1, n_neurons)

        self.dx_hat = self.gamma * prev_grads   

        self.dvar = np.sum(self.dx_hat * self.x_mu* -0.5 * self.std_inv**3, axis=0, keepdims=True)

        self.dmu = np.sum(self.dx_hat * -self.std_inv, axis=0, keepdims=True) + self.dvar * np.mean(-2.0 * self.x_mu, axis=0, keepdims=True)

        self.dinputs = self.dx_hat * self.std_inv + self.dvar * 2 * self.x_mu/self.mean + self.dmu/self.mean
