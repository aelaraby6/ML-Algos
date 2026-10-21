from DecisionTreeRegressor import DecisionTreeRegressor

# [area in square meters, number of rooms]
X = [
    [50, 1],
    [60, 2],
    [70, 2],
    [80, 3],
    [100, 3],
    [120, 4],
    [150, 4],
    [180, 5],
]

# Target: house price (in thousands)
y = [100, 130, 150, 200, 260, 330, 420, 500]

# Train the model
model = DecisionTreeRegressor(max_depth=2)
model.fit(X, y)

# Draw the tree
model.plot()

# Predict on new houses
new_houses = [[65, 2], [110, 3], [170, 5]]
predictions = model.predict(new_houses)

print("\nPredictions:")
for house, price in zip(new_houses, predictions):
    print(f"Area={house[0]}, Rooms={house[1]} -> Price={price:.1f}")