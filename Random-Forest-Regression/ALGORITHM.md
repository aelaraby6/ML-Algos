# Random Forest Regression — How It Works

## 1. The Big Idea

A single decision tree is easy to understand but **overfits**: it memorizes noise in the training data.
A **Random Forest** fixes this by training many different trees and **averaging** their predictions.
Individual trees make different mistakes, and averaging cancels much of that error out
(lower variance, similar bias).

```
 Training data
      |
      |-- bootstrap sample 1 --> Tree 1 --> prediction 1 --+
      |-- bootstrap sample 2 --> Tree 2 --> prediction 2 --+--> average --> final prediction
      |-- ...                                              |
      '-- bootstrap sample N --> Tree N --> prediction N --+
```

## 2. Project Files

| File | Purpose |
|------|---------|
| `random_forest.py` | The `RandomForestRegressor` class (numpy only) |
| `experiment.py` | Generates toy data, trains the model, evaluates it, and plots results |
| `result.png` | Forest prediction vs. data vs. true function |
| `trees.png` | Two individual trees drawn from the forest |

Run it with:

```bash
python experiment.py
```

## 3. Training (`fit`)

For each of the `n_trees` trees:

1. **Bootstrap sampling** — draw `n` rows *with replacement* from the training set.
   About 63% of the unique rows appear in each sample; the rest are left out.
   This makes every tree see slightly different data.
2. **Build a regression tree** on that sample (see below).

## 4. Building One Tree (`_build_tree`)

The tree is built recursively, top to bottom:

1. **Stop** and create a *leaf* if any of these is true:
   - `depth >= max_depth`
   - fewer than `min_samples_split` samples remain
   - all target values in the node are identical

   The leaf stores the **mean** of the target values in it.
2. Otherwise, **find the best split** (see next section).
3. Split the data into `left` (`feature <= threshold`) and `right` (`feature > threshold`)
   and build each child recursively with `depth + 1`.

Each internal node is stored as a dictionary:

```python
{"feature": 0, "threshold": 2.68, "left": <subtree or leaf>, "right": <subtree or leaf>}
```

## 5. Finding the Best Split (`_best_split`)

For every candidate feature and every candidate threshold:

- Candidate thresholds are the **midpoints between consecutive unique values** of the feature.
- Split `y` into `left` and `right`.
- Compute the cost, the **sum of squared errors (SSE)**:

```
SSE = Σ (y_left − mean(y_left))²  +  Σ (y_right − mean(y_right))²
```

The split with the **lowest SSE** wins, because it makes both children as "pure" (low variance) as possible.

**Feature randomness:** if `max_features` is set, only a random subset of features is considered at each split.
This decorrelates the trees even further (it has no effect with one feature, as in the toy example).

## 6. Prediction (`predict`)

1. For each input `x`, walk down every tree:
   at each node go left if `x[feature] <= threshold`, otherwise right, until a leaf is reached.
2. The leaf value is that tree's prediction.
3. The forest's prediction is the **average over all trees**.

## 7. Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_trees` | More trees → smoother, more stable predictions (diminishing returns) |
| `max_depth` | Deeper trees fit finer detail but overfit more; shallow trees underfit |
| `min_samples_split` | Larger values make trees simpler and reduce overfitting |
| `max_features` | Smaller values → more diverse trees (useful with many features) |
| `seed` | Makes results reproducible |

## 8. Reading the Tree Drawing (`plot_tree`)

- **Blue boxes** are decision nodes: `x <= threshold`.
  The `yes` edge goes left (condition true), the `no` edge goes right.
- **Green boxes** are leaves and show the predicted value (the mean of training targets that reached it).

## 9. Evaluation Metrics Used

- **MSE** = mean of `(y_true − y_pred)²` — lower is better.
- **R²** = `1 − SS_res / SS_tot` — 1.0 is perfect, 0 means no better than predicting the mean.

## 10. Simplifications vs. scikit-learn

This implementation is made for learning, so it is intentionally simple:

- Pure-Python loops, so it is slow on large datasets (scikit-learn uses optimized Cython).
- No out-of-bag score, feature importances, or parallel training.
- Thresholds are tried exhaustively instead of with histogram binning.
