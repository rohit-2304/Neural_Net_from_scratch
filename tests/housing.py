import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ronet
from ronet.model import *

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
data = pd.read_csv('data/housing/cleaned.csv')

X = data.drop(['price'], axis=1).to_numpy()
y = data['price'].to_numpy().reshape(-1, 1)

# -------------------------------------------------------------
# 2. TRANSFORM TARGET (log transform improves regression A LOT)
# -------------------------------------------------------------
y_log = np.log1p(y)    # log(price + 1)

# -------------------------------------------------------------
# 3. SCALE INPUT FEATURES
# -------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------------------------------------
# 4. BUILD MODEL (Better architecture for nonlinear regression)
# -------------------------------------------------------------
model = Model()

model.add(ronet.layers.Dense(X.shape[1], 32))
model.add(ronet.activations.ReLU())

model.add(ronet.layers.Dense(32, 16))
model.add(ronet.activations.ReLU())

model.add(ronet.layers.Dense(16, 8))
model.add(ronet.activations.ReLU())

model.add(ronet.layers.Dense(8, 1))
model.add(ronet.activations.Linear())   # Regression output

# -------------------------------------------------------------
# 5. SET LOSS + OPTIMIZER + METRIC
# -------------------------------------------------------------
model.set(
    loss = MeanSquaredErrorLoss(),
    optimizer = Optimizer_Adam(learning_rate=0.001),
    accuracy = Accuracy_Regression(),   # useless but required by RONet
)

model.finalize()

# -------------------------------------------------------------
# 6. TRAIN
# -------------------------------------------------------------
model.train(X, y_log, epochs=500, print_every=20)

# -------------------------------------------------------------
# 7. EVALUATE ON TRAINING DATA (Compute RMSE in REAL PRICE units)
# -------------------------------------------------------------
y_pred_log = model.predict(X)
y_pred = np.expm1(y_pred_log)   # reverse log1p()

rmse = np.sqrt(np.mean((y_pred - y)**2))
print("\nFinal RMSE:", rmse)
y_pred_log = model.predict(X)
y_pred = np.expm1(y_pred_log)

print("Pred mean:", y_pred.mean())
print("True mean:", y.mean())
print("First 10 predictions:", y_pred[:10].flatten())
print("First 10 true:", y[:10].flatten())
