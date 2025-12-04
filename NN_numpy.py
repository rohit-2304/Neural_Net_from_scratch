import numpy as np

# dense layer
class Dense:
    def __init__(self, n_in, n_neurons, weight_regularizer_l1=0, bias_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l2=0):
        upper = (np.sqrt(6/(n_in + n_neurons)))
        lower = -upper
        # Xavier Initialization
        self.weights = np.random.uniform(lower, upper, (n_in, n_neurons))    # w^T shape
        self.biases = np.random.uniform(lower, upper, (1, n_neurons))               # b^T shape

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
            l1_dbias = np.ones_like(self.biases)
            l1_dbias[self.biases < 0] = -1
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    def __repr__(self):
        return f"Dense Layer ({self.weights.shape[0]} -> {self.weights.shape[1]})"
    
    def __str__(self):
        return self.__repr__()

# activation functions
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
        self.dinputs[prev_grads <= 0] = 0 
    
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

# loss functions
class Loss:

    # calculates the data and regularization losses
    # given model output and ground truth values

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization =False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
    def reset(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def regularization_loss(self, layer):
        # total data_loss = regularization_loss(layer1) + regularization_loss(layer2) + ..... n

        regularization_loss = 0
        for layer in self.trainable_layers:
        # if l1 regularization used
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularization_l1 * np.sum(np.abs(layer.weights))
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularization_l1 * np.sum(np.abs(layer.biases))

            # if l2_regularization used
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularization_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularization_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss

class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # to prevent division by zero(log(0)) 
        clipped_output = np.clip(y_pred, 1e-07, 1-1e-07)    # 1-1e-07 becuase if there is some epsilon addition the log(x > 1) will be negative instead of 0

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
        clipped_y_pred = np.clip(y_pred, 1e-07, 1-1e-07)

        # prev_grad shape = (m, c)

        # if labels are sparse convert to one hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]         #[y_true] reorders the values of I to match the hot encoding
        
        self.dinputs = -y_true / clipped_y_pred

        # normalization of gradients
        self.dinputs = self.dinputs/samples

class BinaryCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # to prevent division by zero(log(0))   
        y_pred_clipped = np.clip(y_pred, 1e-07, 1-1e-07)    # 1-1e-07 becuase if there is some epsilon addition the log(x > 1) will be negative instead of 0

        sample_losses = -(y_true*np.log(y_pred_clipped) + (1- y_true)*np.log(1- y_pred_clipped))
        sample_losses = -np.mean(sample_losses, axis=-1)
        return sample_losses
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)       # number of samples m
        outputs = len(y_pred[0])     # number of classes c
        clipped_ypred = np.clip(y_pred, 1e-7,1 - 1e-7)
        self.dinputs = -(y_true / clipped_ypred - (1- y_true)/(1- clipped_ypred))/ outputs
        self.dinputs = self.dinputs / samples

class MeanSquaredErrorLoss(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis = -1)
        return sample_losses
    
    def backward(self, y_pred, y_true):

        samples = len(y_pred)
        outputs = len(y_pred[0])

        self.dinputs = -2 * (y_true - y_pred)/ outputs
        # loss over whole batch
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
# no jacobian calculation involved
class Activation_Softmax_Loss_Cross_Entropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CrossEntropyLoss()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.forward(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        """Derivation of combined Softmax + Cross-Entropy gradient:

        1. Softmax converts logits z_j into probabilities p_j.
            p_j = e^(z_j) / sum(e^(z_k))

        2. Cross-entropy loss for one-hot labels:
            L = -log(p_c)      (c = correct class)

        3. We want dL/dz_j (gradient of loss wrt logits).

        4. Use the chain rule:
            dL/dz_j = sum_i( dL/dp_i * dp_i/dz_j )

        5. Derivative of cross-entropy wrt softmax output:
            dL/dp_i = -y_i / p_i
            (non-zero only for the correct class because y is one-hot)

        6. Derivative of softmax wrt logits:
            if i == j: dp_i/dz_j = p_i * (1 - p_i)
            else:      dp_i/dz_j = -p_i * p_j

        7. Multiply them using the chain rule: the complicated terms cancel out.

        8. After simplification the entire derivative collapses to:
            dL/dz_j = p_j - y_j

        9. Meaning:
            - correct class:  p_j - 1
            - other classes:  p_j - 0

        10. This is why the combined backward pass can simply do:
                dinputs = softmax_output
                dinputs[range(samples), true_class] -= 1
                dinputs /= samples
            which implements: (p - y) / batch_size  """

        # in the combined backward pass:
        # dvalues = softmax output (p_j)  -> NOT upstream gradients
        # because the loss layer starts backprop, so it receives the model's predictions
        
        samples = len(dvalues)  # number of samples m
        # dvalues shape = (m, n_classes)

        # if true labels are one-hot encoded convert them to sparse labels
        # because we only need the index where y_j = 1
        # and subtracting 1 at that index will give (p_j - y_j)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)  
            # y_true now shape = (m, ), each entry is correct class index

        # make a copy so we don't modify original softmax output
        # starting gradient = p_j
        # final gradient = p_j - y_j
        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        # normalization
        self.dinputs = self.dinputs / samples

# optimizers
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
            self.current_learning_rate = self.learning_rate*(1/ 1 + self.steps*self.decay)
        
    def update_params(self, layer):
        # cache is the history of updates
        # the layer wont have a cache to begin with
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # update cache with current squared gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cach += layer.dbiases**2

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
        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho)*self.dweights**2
        layer.bias_cache = self.rho*layer.bias_cache + (1-self.rho)*self.dbiases**2

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

class Input:
    def forward(self, inputs):
        self.output = inputs

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy
    
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum/self.accumulated_count

        return accuracy
    
    def reset(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision =None
    
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) /250
    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Classification(Accuracy):
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer, accuracy):          # * ensures that loss and optimizer are required keyword args
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Input()

        layer_count = len(self.layers)

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

        self.trainable_layers = []
        for layer in self.layers:
            if hasattr(layer, 'trainable') :
                self.trainable_layers.append(layer)
        self.loss.remember_trainable_layers(self.trainable_layers)

        # use combined softmax and crossentropy
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CrossEntropyLoss):
            self.softmax_classifier_output = Activation_Softmax_Loss_Cross_Entropy()

    def train(self, X, y, X_val = None, y_val = None,batch_size = None,*, epochs = 1, verbose = 1, print_every = 1):
        self.accuracy.init(y)

        train_steps = 1

        if X_val is not None and y_val is not None:
            val_steps = 1
        
        if batch_size is not None:
            train_steps = len(X)//batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1
            if X_val is not None and y_val is not None:
                val_steps = len(X_val) // batch_size
                if val_steps * batch_size < len(X_val):
                    val_steps += 1
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            self.loss.reset()
            self.accuracy.reset()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size: (step + 1)*batch_size]
                    batch_y = y[step*batch_size: (step + 1)*batch_size]

            output = self.forward(batch_X)
            data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization = True)
            loss = data_loss + regularization_loss  

            predictions = self.output_layer_activation.predict(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
            self.backward(output, batch_y)
            for layer in self.trainable_layers:
                self.optimizer.pre_update_params()
                self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            if not step % print_every or step == train_steps-1:
                print(f'step {step}\tAcc = {accuracy}\tLoss = {loss}')
        
        epoch_data_loss, epoch_reg_loss = self.loss.calculate_accumulated(include_regularization = True)
        epoch_loss = epoch_loss + epoch_reg_loss
        epoch_acc = self.accuracy.calculate_accumulated()

        print(f'Epoch Acc = {epoch_acc}\tLoss = {epoch_loss}')


        if (X_val is not None) and (y_val is not None):
            self.loss.reset()
            self.accuracy.reset()

            for step in range(val_steps):
                if batch_size is None:
                    batch_X = X_val
                    batch_y = y_val
                else:
                    batch_X = X_val[step*batch_size : (step+1)*batch_size]
                    batch_y = y_val[step*batch_size : (step+1)*batch_size]

                val_output = self.forward(X_val, training = False)
                val_loss = self.loss.calculate(val_output, y_val)

                val_predictions = self.output_layer_activation.predict(val_output)
                val_accuracy = self.accuracy.calculate(val_predictions, y_val)
            val_loss = self.loss.calculated_accumulated()
            val_accuracy = self.accuracy.calculate_accumulated()
            print(f'Val Loss = {val_loss}\tAcc = {val_accuracy}')
    def temp(self):
        pass    
    def forward(self, X, training = True):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output, training = training)
        
        return layer.output
    
    def backward(self, outputs, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(outputs, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        self.loss.backward(outputs, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def predict(self,X):
        output = self.forward(X, training = False)
        predictions = np.argmax(output, axis=1)
        return predictions


