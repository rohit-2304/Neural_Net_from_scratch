import numpy as np

class Optimizer:
    def pre_update_params(self):
        pass
    def update_params(self):
        pass
    def post_update_params(self):
        pass

class Optimizer_SGD(Optimizer):
    # momentum prevents model beign stuck in local minima
    # learning rate decay prevents exploding gradients
    def __init__(self, learning_rate = 0.001, decay = 0., momentum = 0., epsion = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.steps = 0      # how many steps have taken place so far
        self.momentum_factor = momentum 

    # implement a wrapper here if decay
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.steps))
    
    def update_params(self, layer):

        # if momentum is used
        if self.momentum_factor:
            # per parameter momentums
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                            # fraction from previous updates
            weight_updates = (self.momentum_factor*layer.weight_momentums) - (self.current_learning_rate*layer.dweights)   # everything is a vector here
            layer.weight_momentums = weight_updates

            bias_updates = (self.momentum_factor*layer.bias_momentums) - (self.current_learning_rate*layer.dbiases)   # everything is a vector here
            layer.bias_momentums = bias_updates
        # vanilla sgd
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.steps += 1

class Optimizer_Adagrad(Optimizer):
    """Adagrad optimizer with learning rate decay support"""
    # reduce the update size in ratio to the previous size of updates for each param
    def __init__(self, learning_rate = 1e-3, decay=0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.steps= 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate/( 1 + self.steps*self.decay)
        
    def update_params(self, layer):
        # cache is the history of updates
        # the layer wont have a cache to begin with
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # update cache with current squared gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -(self.current_learning_rate * layer.dweights)/(np.sqrt(layer.weight_cache )+ self.epsilon)
        layer.biases += -(self.current_learning_rate * layer.dbiases)/(np.sqrt(layer.bias_cache) + self.epsilon)

    def post_udpate_params(self):
        self.steps += 1

class Optimizer_RMSProp(Optimizer):
    # Slows down oscillations in dimensions where it high
    # reduce the update size in ratio to the previous size of updates for each param , but in a better way
    # adds a mechanism similar to momentum 
    # calculates per param learning rate 
    # retains global direction and slows changes in direction
    def __init__(self, learning_rate = 1e-03, decay = 0., rho = 0.9, epsilon = 1e-07):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.steps = 0      # how many steps have taken place so far
        self.rho = rho      # cache memory decay rate : factor which decides the proportion of previous cache and current gradients squared
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1. + self.decay*self.steps))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # some proportion of previous gradients and some proportion of current gradients
        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache = self.rho*layer.bias_cache + (1-self.rho)*layer.dbiases**2

        # update like Adagrad
        layer.weights += -self.current_learning_rate* layer.dweights/(np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate* layer.dbiases/(np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.steps += 1

class Optimizer_Adam(Optimizer):
    def __init__(self, learning_rate = 1e-03, decay = 0.,  epsilon = 1e-07, beta_1 = 0.9, beta_2 = 0.9999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.steps = 0              # how many steps have taken place so far
        self.epsilon = epsilon
        self.beta_1 = beta_1    #    momentum factor
        self.beta_2 = beta_2            # cache memory decay rate

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1. + self.decay*self.steps))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_momentums = (self.beta_1*layer.weight_momentums) + ((1-self.beta_1)*layer.dweights)
        layer.bias_momentums = (self.beta_1*layer.bias_momentums) + ((1-self.beta_1)*layer.dbiases)

        # adujst for early small values
        # adaptive momentums - early the momentum will be greater
        weight_momentums_corrected = layer.weight_momentums/(1 - self.beta_1**(self.steps + 1) )# + 1 to avoid div by zero
        bias_momentums_corrected = layer.bias_momentums/(1 - self.beta_1**(self.steps +1))

        layer.weight_cache = self.beta_2*layer.weight_cache + (1-self.beta_2)*layer.dweights**2
        layer.bias_cache = self.beta_2*layer.bias_cache + (1-self.beta_2)*layer.dbiases**2

        # adjust for early small values
        weight_cache_corrected = layer.weight_cache/(1 - self.beta_2**(self.steps + 1) )
        bias_cache_corrected = layer.bias_cache/(1 - self.beta_2**(self.steps +1))

        # param updates : vanilla sgd param update + normalization with sqaure rooted cache
        layer.weights += -self.current_learning_rate* weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+ self.epsilon)
        layer.biases += -self.current_learning_rate*bias_momentums_corrected/(np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.steps += 1
        