class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth = max_depth      # maximum depth of the tree
        self.min_samples = min_samples  # minimum rows needed to split a node
        self.tree = None

    def _mean(self, y):
        return sum(y) / len(y)

    def _error(self, y):
        # Sum of squared differences from the mean.
        # Smaller value = the group is more similar.
        m = self._mean(y)
        return sum((v - m) ** 2 for v in y)

    def _best_split(self, X, y):
        best_error = float("inf")
        best_feature, best_threshold = None, None

        n_features = len(X[0])
        for f in range(n_features):                      # try every column
            for threshold in set(row[f] for row in X):  # try every value as a split point
                left_y = [y[i] for i in range(len(X)) if X[i][f] <= threshold]
                right_y = [y[i] for i in range(len(X)) if X[i][f] > threshold]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                error = self._error(left_y) + self._error(right_y)
                if error < best_error:
                    best_error = error
                    best_feature, best_threshold = f, threshold

        return best_feature, best_threshold

    # build the tree
    def _build(self, X, y, depth):
        # Stopping rules: return a leaf (the mean of the targets)
        if depth >= self.max_depth or len(y) < self.min_samples or len(set(y)) == 1:
            return {"value": self._mean(y)}

        f, t = self._best_split(X, y)
        if f is None:
            return {"value": self._mean(y)}

        left_idx = [i for i in range(len(X)) if X[i][f] <= t]
        right_idx = [i for i in range(len(X)) if X[i][f] > t]

        return {
            "feature": f,
            "threshold": t,
            "left": self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1),
            "right": self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    # ---------- prediction ----------
    def _predict_one(self, row, node):
        if "value" in node:  # reached a leaf
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        return self._predict_one(row, node["right"])

    def predict(self, X):
        return [self._predict_one(row, self.tree) for row in X]

    # ---------- draw the tree with matplotlib ----------
    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        self._leaf_count = 0  # each leaf gets its own x position (0, 1, 2, ...)
        self._draw(self.tree, 0, ax)

        ax.set_xlim(-0.7, self._leaf_count - 0.3)
        ax.set_ylim(-self.max_depth - 0.7, 0.7)
        ax.axis("off")
        ax.set_title("Decision Tree Regressor")
        plt.show()

    def _draw(self, node, depth, ax):
        # Draws the node and returns its x position
        y = -depth  # deeper nodes are lower on the plot

        # Leaf: draw a green box with the predicted value
        if "value" in node:
            x = self._leaf_count
            self._leaf_count += 1
            ax.text(x, y, f"price\n{node['value']:.1f}", ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="lightgreen"))
            return x

        # Internal node: draw children first, then place this node between them
        left_x = self._draw(node["left"], depth + 1, ax)
        right_x = self._draw(node["right"], depth + 1, ax)
        x = (left_x + right_x) / 2

        # Lines to the children, with yes/no labels
        ax.plot([x, left_x], [y, y - 1], "k-", zorder=0)
        ax.plot([x, right_x], [y, y - 1], "k-", zorder=0)
        ax.text((x + left_x) / 2, y - 0.5, "yes", ha="center", color="green")
        ax.text((x + right_x) / 2, y - 0.5, "no", ha="center", color="red")

        # Draw the question in a blue box
        ax.text(x, y, f"feature {node['feature']}\n<= {node['threshold']} ?",
                ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="lightblue"))
        return x