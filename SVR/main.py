from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SVR import SimpleSVR

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_excel(BASE_DIR / "data.xlsx")
x = df["hours"].values
y = df["score"].values

# Create and train the model
model = SimpleSVR(epsilon=3.0, C=10.0, learning_rate=0.0005, steps=50000)
model.fit(x, y)

print(f"Line found:  score = {model.w:.2f} * hours + {model.b:.2f}")
print(f"Points outside the tube (the support vectors): {model.n_outside} of {len(x)}")

new_hours = np.array([2.5, 5.5, 8.0])
print("Predictions:")
for h, p in zip(new_hours, model.predict(new_hours)):
    print(f"  {h} hours -> {p:.1f}")

# Plot 
line_x = np.linspace(x.min(), x.max(), 100)
line_y = model.predict(line_x)
plt.scatter(x, y, label="Data")
plt.plot(line_x, line_y, "r", label="SVR line")
plt.fill_between(line_x, line_y - model.epsilon, line_y + model.epsilon,
                 color="r", alpha=0.15, label="epsilon tube")
plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.legend()
plt.savefig(BASE_DIR / "svr_simple_plot.png", dpi=120)
plt.show()