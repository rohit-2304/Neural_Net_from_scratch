import numpy as np
import pandas as pd
from ronet.model import *

training_data = pd.read_csv('data/MNIST/mnist_train.csv')
testing_data = pd.read_csv('data/MNIST/mnist_test.csv')
training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)


val_ratio = 0.2
data_size = training_data.shape[0]
val_size = round(0.2*data_size)
X_train = training_data[:-val_size]
X_val = training_data[-val_size:]
y_train = X_train['label']
y_val = X_val['label']
X_train = X_train.drop(['label'], axis = 1)
X_val = X_val.drop(['label'], axis = 1)

X_train = X_train.to_numpy()/255
y_train = y_train.to_numpy()
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()

print(y_train.shape)


model = Model()

model.add(Dense(784, 256))
model.add(ReLU())
model.add(Dense(256, 128))
model.add(Dropout(0.1))
model.add(ReLU())
model.add(Dense(128, 64))
model.add(Dropout(0.1))
model.add(ReLU())
model.add(Dense(64, 10))
model.add(Softmax())

model.set(loss=CrossEntropyLoss(), optimizer= Optimizer_Adam(learning_rate=0.001,decay=0.001), accuracy= Accuracy_Classification())

model.finalize()

model.train(X_train, y_train, batch_size=512, epochs=50, print_every= 512)