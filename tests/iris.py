import pandas as pd
import numpy as np
from ronet.model import *
from ronet import *
from torch import nn
import torch
import torch.optim as optim

data = pd.read_csv("data/iris/Iris.csv")
data = data.drop(['Id'], axis = 1)

labelEnc = {}
for i, class_ in enumerate(data.Species.unique()):
    labelEnc[class_] = i

data.Species = data.Species.apply(lambda x : labelEnc[x])

X = data.drop(['Species'], axis = 1)
y = data['Species']

X_np = X.to_numpy()
y_np = y.to_numpy()

model = Model()

model.add(Dense(4, 16))
model.add(ReLU())
model.add(Dense(16, 12))
model.add(ReLU())
model.add(Dense(12, 3))
model.add(Softmax())

model.set(loss = CrossEntropyLoss(), optimizer = Optimizer_Adam(learning_rate=0.001),accuracy=Accuracy_Classification())

model.finalize()

model.train(X_np, y_np, epochs=100)

print("\n\n\tpytorch\n\n")
model = nn.Sequential(
    nn.Linear(4,16),
    nn.ReLU(),
    nn.Linear(16,12),
    nn.ReLU(),
    nn.Linear(12,3),
    nn.Softmax()
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs_torch = torch.from_numpy(X_np).float()
outputs_torch = torch.from_numpy(y_np).long()

epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(inputs_torch)
    loss = criterion(logits, outputs_torch)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")
