import numpy as np
from .activations import Softmax

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
    
    def regularization_loss(self):
        # total data_loss = regularization_loss(layer1) + regularization_loss(layer2) + ..... n

        regularization_loss = 0
        for layer in self.trainable_layers:
        # if l1 regularization used
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # if l2_regularization used
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
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

        sample_losses = y_true*np.log(y_pred_clipped) + (1- y_true)*np.log(1- y_pred_clipped)
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
        sample_losses = np.mean((y_true - y_pred)**2, axis = -1, keepdims=True)
        return sample_losses
    
    def backward(self, y_pred, y_true):

        samples = len(y_pred)
        outputs = len(y_pred[0])
        
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1,1)

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
