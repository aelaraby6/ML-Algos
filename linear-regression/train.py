import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

df = pd.read_csv("house_pricing.csv")

X = df[['sqft', 'bedrooms', 'bathrooms']].values
y = df['price'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(lr=0.0000001, n_iters=10000)  

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred)**2)
print("MAE:", mae)
print("MSE:", mse)

X_feature_test = X_test[:, 0]  # sqft
X_feature_train = X_train[:, 0]

plt.figure(figsize=(8,6))
plt.scatter(X_feature_train, y_train, color='blue', alpha=0.6, label='Training data')
plt.scatter(X_feature_test, y_test, color='green', alpha=0.6, label='Test data')

y_line = model.weights[0] * X_feature_test + model.bias
plt.plot(X_feature_test, y_line, color='red', linewidth=2, label='Regression line (sqft)')

plt.xlabel("Square Footage (sqft)")
plt.ylabel("Price")
plt.title("Linear Regression Prediction")
plt.legend()
plt.show()