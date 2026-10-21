import numpy as np
import matplotlib.pyplot as plt
from RandomForestRegressor import RandomForestRegressor

# Synthetic Dataset: sin(x) + Noise
rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 10, 200)).reshape(-1, 1)
y = np.sin(X).ravel() + rng.normal(0, 0.3, 200)

# Train / Test split
idx = rng.permutation(len(X))
split = int(0.8 * len(X))
X_train, y_train = X[idx[:split]], y[idx[:split]]
X_test, y_test = X[idx[split:]], y[idx[split:]]

# Training
model = RandomForestRegressor(n_trees=50, max_depth=5)
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
mse = np.mean((y_test - pred) ** 2)
r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"MSE = {mse:.4f}")
print(f"R2  = {r2:.4f}")

# Plotting
X_line = np.linspace(0, 10, 300).reshape(-1, 1)
plt.figure(figsize=(9, 5))
plt.scatter(X_train, y_train, s=15, color="gray", alpha=0.6, label="Train data")
plt.scatter(X_test, y_test, s=25, color="orange", label="Test data")
plt.plot(X_line, model.predict(X_line), color="red", linewidth=2, label="Random Forest")
plt.plot(X_line, np.sin(X_line), "g--", label="True sin(x)")
plt.title("Random Forest Regression from Scratch")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("result.png", dpi=120)
plt.show()