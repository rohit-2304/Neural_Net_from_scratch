import numpy as np
from .layers import *
from .loss import *
from .optimizers import *
from .regularization import *
from .activations import *

class Input:
    def forward(self, inputs):
        self.output = inputs

class Accuracy:
    def init(self, y):
        pass

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
        self.softmax_classifier_output = None

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

    # training loop
    def train(self, X, y, X_val = None, y_val = None,batch_size = None,*, epochs = 1, verbose = 1, print_every = 1):
        self.accuracy.init(y)

        train_steps = 1

        # if validation used
        self.val = False
        if X_val is not None and y_val is not None:
            self.val = True
        
        if self.val:
            val_steps = 1

        # if batch size used
        if batch_size is not None:
            # no. of steps for each epoch
            train_steps = len(X)//batch_size

            # as floor div rounds to lower integer
            if train_steps * batch_size < len(X):
                train_steps += 1
            if self.val:
                # no. of steps for each epoch
                val_steps = len(X_val) // batch_size
                # as floor div rounds to lower integer
                if val_steps * batch_size < len(X_val):
                    val_steps += 1

        # actual training
        for epoch in range(1, epochs+1):

            # reset the accumulated loss over epoch to 0
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
                
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
        
            epoch_data_loss, epoch_reg_loss = self.loss.calculate_accumulated(include_regularization = True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.calculate_accumulated()

            print(f'Epoch {epoch} Acc = {epoch_acc}\tLoss = {epoch_loss}')


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

                    val_output = self.forward(batch_X, training = False)
                    val_loss = self.loss.calculate(val_output, batch_y)

                    val_predictions = self.output_layer_activation.predict(val_output)
                    val_accuracy = self.accuracy.calculate(val_predictions, batch_y)
                val_loss = self.loss.calculate_accumulated()
                val_accuracy = self.accuracy.calculate_accumulated()
                print(f'Val Loss = {val_loss}\tAcc = {val_accuracy}')
    
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
        predictions = self.output_layer_activation.predict(output)
        return predictions


