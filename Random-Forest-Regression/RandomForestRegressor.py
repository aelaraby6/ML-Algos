import numpy as np


class RandomForestRegressor:
    """Random Forest Regressor from scratch using NumPy."""

    def __init__(self, n_trees=50, max_depth=5, min_samples_split=5,
                 max_features=None, seed=42):
        self.n_trees = n_trees                      # Number of trees in the forest
        self.max_depth = max_depth                  # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.max_features = max_features            # Number of features to consider per split (None = all)
        self.rng = np.random.default_rng(seed)
        self.trees = []

    # Tree Building
    def _best_split(self, X, y):
        n_features = X.shape[1]
        k = self.max_features or n_features
        features = self.rng.choice(n_features, size=min(k, n_features), replace=False)

        best_feature, best_thr, best_sse = None, None, np.inf
        for f in features:
            values = np.unique(X[:, f])
            thresholds = (values[:-1] + values[1:]) / 2   # Midpoints between unique values
            for thr in thresholds:
                left = y[X[:, f] <= thr]
                right = y[X[:, f] > thr]
                # Sum of Squared Errors (SSE) for both splits
                sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
                if sse < best_sse:
                    best_feature, best_thr, best_sse = f, thr, sse
        return best_feature, best_thr

    def _build_tree(self, X, y, depth):
        # Stopping criteria: return node mean value (leaf node)
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return float(y.mean())

        f, thr = self._best_split(X, y)
        if f is None:
            return float(y.mean())

        mask = X[:, f] <= thr
        return {
            "feature": f,
            "threshold": thr,
            "left": self._build_tree(X[mask], y[mask], depth + 1),
            "right": self._build_tree(X[~mask], y[~mask], depth + 1),
        }

    #  Single Tree Prediction 
    def _predict_one(self, tree, x):
        while isinstance(tree, dict):
            tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
        return tree

    #  Main Interface
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.trees = []
        n = len(y)
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, size=n)   # Bootstrap sampling
            self.trees.append(self._build_tree(X[idx], y[idx], depth=0))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        # Average of predictions across all trees
        preds = np.array([[self._predict_one(t, x) for x in X] for t in self.trees])
        return preds.mean(axis=0)