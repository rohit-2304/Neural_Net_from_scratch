import numpy as np
import pandas as pd
from NN_numpy import *
from torch import nn
import torch
import torch.optim as optim

training_data = pd.read_csv('data/MNIST/mnist_train.csv')
testing_data = pd.read_csv('data/MNIST/mnist_test.csv')

val_ratio = 0.2
data_size = training_data.shape[0]
val_size = round(0.2*data_size)
X_train = training_data[:-val_size]
X_val = training_data[-val_size:]
y_train = X_train['label']
y_val = X_val['label']
X_train = X_train.drop(['label'], axis = 1)
X_val = X_val.drop(['label'], axis = 1)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()

print("\tpytorch\n\n")
device = torch.device("cpu")

print(device)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.Softmax()
)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs_torch = torch.from_numpy(X_train).float()
outputs_torch = torch.from_numpy(y_train).long()
model.to(device)
inputs = inputs_torch.to(device)
labels = outputs_torch.to(device)
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(inputs_torch)
    loss = criterion(logits, outputs_torch)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")
