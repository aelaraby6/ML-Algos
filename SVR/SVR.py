# The idea in plain words:
#   We want a straight line  y = w * x + b  that fits the data.
#   - Points INSIDE a tube of width epsilon around the line cost nothing.
#   - Points OUTSIDE the tube are penalized (the further out, the bigger the penalty).
#   - We also keep the slope w small (a "flatter" line is simpler).
#
# Loss = 0.5 * w^2  +  C * sum( max(0, |error| - epsilon) )
#
# We find w and b with gradient descent: repeatedly nudge them downhill on the loss.

import numpy as np


class SimpleSVR:
    def __init__(self, epsilon=3.0, C=10.0, learning_rate=0.0005, steps=50000):
        self.epsilon = epsilon              # half-width of the tube: smaller errors are ignored
        self.C = C                          # how strongly we punish points outside the tube
        self.learning_rate = learning_rate  # size of each nudge
        self.steps = steps                  # how many nudges

        self.w = 0.0                        # slope (learned in fit)
        self.b = 0.0                        # intercept (learned in fit)
        self.n_outside = 0                  # points outside the tube (support vectors)

    def fit(self, x, y):
        # Start with a flat line
        self.w = 0.0
        self.b = 0.0

        # Gradient descent
        for step in range(self.steps):
            prediction = self.w * x + self.b
            error = y - prediction                        # positive: point is above the line

            outside = np.abs(error) > self.epsilon        # True for points outside the tube
            direction = np.sign(error) * outside          # +1 above, -1 below, 0 inside

            # Slope of the loss with respect to w and b
            grad_w = self.w - self.C * np.sum(direction * x)
            grad_b = -self.C * np.sum(direction)

            # Move a small step downhill
            self.w = self.w - self.learning_rate * grad_w
            self.b = self.b - self.learning_rate * grad_b

        # Count the points outside the tube for the final line
        final_error = y - self.predict(x)
        self.n_outside = int(np.sum(np.abs(final_error) > self.epsilon))
        return self

    def predict(self, x):
        return self.w * x + self.b