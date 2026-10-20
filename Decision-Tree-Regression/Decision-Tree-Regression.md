# Decision Tree Regression, Explained Simply

This guide explains the algorithm used in `decision_tree.py` and walks through
real numbers from `test_tree.py`.

---

## 1. What is a decision tree regressor?

It predicts a **number** (for example a house price) by asking a series of
yes/no questions about the input.

```
Is area <= 100 ?
 ├── yes → Is area <= 70 ?
 │          ├── yes → predict 126.7
 │          └── no  → predict 230.0
 └── no  → Is area <= 120 ?
            ├── yes → predict 330.0
            └── no  → predict 460.0
```

Every house ends up in exactly one **leaf**, and the leaf's prediction is the
**average price** of the training houses that landed there.

---

## 2. Key words

| Word | Meaning |
|------|---------|
| **Node** | One question in the tree (e.g. `area <= 100 ?`) |
| **Leaf** | The end of a branch. It holds the predicted number |
| **Split** | Dividing rows into a left group and a right group |
| **Threshold** | The value used in the question (e.g. `100`) |
| **Depth** | How many questions deep the tree goes |
| **Error (SSE)** | How "mixed up" a group is. Lower is better |

---

## 3. The dataset used in the examples

| Area (m²) | Rooms | Price (thousands) |
|-----------|-------|-------------------|
| 50  | 1 | 100 |
| 60  | 2 | 130 |
| 70  | 2 | 150 |
| 80  | 3 | 200 |
| 100 | 3 | 260 |
| 120 | 4 | 330 |
| 150 | 4 | 420 |
| 180 | 5 | 500 |

In the code: `X` is the list of `[area, rooms]` rows and `y` is the list of prices.

---

## 4. The algorithm step by step

1. Start with **all rows** in one group.
2. **Try every possible split**: for each column and for each value in that
   column, split into `<= value` (left) and `> value` (right).
3. Measure how good each split is using the **error** (see below).
4. **Pick the split with the lowest total error.**
5. Repeat steps 2 to 4 **separately on the left group and on the right group**.
6. **Stop** when:
   - the maximum depth is reached, or
   - the group is too small (`min_samples`), or
   - all prices in the group are identical.
7. At each stopping point, create a **leaf** with the **mean** of its prices.

Step 5 is called **recursion**: the function `_build` calls itself on the
smaller groups.

---

## 5. How do we measure a good split?

For one group, the error is the sum of squared differences from the group's mean:

```
error = (price1 - mean)² + (price2 - mean)² + ...
```

- If all prices in a group are close to each other, the error is **small**.
- If prices are very different, the error is **large**.

For a split, we add the error of the left group and the error of the right group.

### Example: a tiny group

Group `[100, 130, 150]`
- mean = (100 + 130 + 150) / 3 = **126.7**
- error = (100 − 126.7)² + (130 − 126.7)² + (150 − 126.7)² ≈ **1266.7**

Group `[100, 500]`
- mean = 300
- error = (100 − 300)² + (500 − 300)² = **80000**

The first group is much more "similar", so it has a much smaller error.

---

## 6. Worked example: finding the first split

We try every area value as the threshold (feature 0). The first column shows
the question, and the other columns show the result:

| Question | Left prices | Right prices | Left error | Right error | **Total** |
|----------|-------------|--------------|-----------:|------------:|----------:|
| area <= 50  | 100 | 130 … 500 | 0 | 116,571 | 116,571 |
| area <= 60  | 100, 130 | 150 … 500 | 450 | 88,800 | 89,250 |
| area <= 70  | 100, 130, 150 | 200 … 500 | 1,267 | 58,080 | 59,347 |
| area <= 80  | 100 … 200 | 260 … 500 | 5,300 | 32,875 | 38,175 |
| **area <= 100** | **100 … 260** | **330, 420, 500** | **15,880** | **14,467** | **30,347** |
| area <= 120 | 100 … 330 | 420, 500 | 37,750 | 3,200 | 40,950 |
| area <= 150 | 100 … 420 | 500 | 81,143 | 0 | 81,143 |

For comparison, with **no split at all** the error is **146,287**.

The winner is **`area <= 100`** because its total error (30,347) is the smallest.
The code does exactly this in `_best_split`. It also tries feature 1 (rooms) but
area gives a better result here.

After this split we have two groups:

- Left: prices `[100, 130, 150, 200, 260]`
- Right: prices `[330, 420, 500]`

---

## 7. Growing the tree (recursion)

The same process is repeated on each group.

**Left group** `[100, 130, 150, 200, 260]` → best split is `area <= 70`
- Left leaf: `[100, 130, 150]` → mean = **126.7**
- Right leaf: `[200, 260]` → mean = **230.0**

**Right group** `[330, 420, 500]` → best split is `area <= 120`
- Left leaf: `[330]` → mean = **330.0**
- Right leaf: `[420, 500]` → mean = **460.0**

With `max_depth=2` the tree stops here. That gives the tree from section 1.

---

## 8. Making a prediction

To predict, follow the questions from the top until you reach a leaf.

**House: 65 m², 2 rooms**
1. `area <= 100` → yes
2. `area <= 70` → yes
3. Leaf → **126.7**

**House: 110 m², 3 rooms**
1. `area <= 100` → no
2. `area <= 120` → yes
3. Leaf → **330.0**

**House: 170 m², 5 rooms**
1. `area <= 100` → no
2. `area <= 120` → no
3. Leaf → **460.0**

Note that rooms was never used. The tree only asks questions that help.

---

## 9. The effect of `max_depth`

Predictions for the 8 training houses with different depths:

| max_depth | Predictions |
|-----------|-------------|
| 1 | 168, 168, 168, 168, 168, 416.7, 416.7, 416.7 |
| 2 | 126.7, 126.7, 126.7, 230, 230, 330, 460, 460 |
| 3 | 100, 140, 140, 200, 260, 330, 420, 500 |
| Real prices | 100, 130, 150, 200, 260, 330, 420, 500 |

- **Too shallow (depth 1)**: the tree is too simple and predictions are rough.
  This is called **underfitting**.
- **Too deep**: the tree memorizes the training data (every house gets its own
  leaf). It looks perfect on training data but can be bad on new houses. This
  is called **overfitting**.
- A middle value (like 2 or 3 here) is usually a good balance.

---

## 10. Map: algorithm step → code

| Step | Method in `decision_tree.py` |
|------|------------------------------|
| Mean of a group | `_mean` |
| Error of a group | `_error` |
| Try all splits and pick the best | `_best_split` |
| Build the tree recursively | `_build` |
| Train the model | `fit` |
| Follow the questions to a leaf | `_predict_one` |
| Predict many rows | `predict` |
| Draw the tree | `plot` and `_draw` |

---

## 11. How to run

```bash
pip install matplotlib
python test_tree.py
```

```python
from decision_tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=2)
model.fit(X, y)
print(model.predict([[65, 2]]))   # [126.66...]
model.plot()                      # draws the tree
```

---

## 12. Things to try

1. Change `max_depth` to 1, 3, and 5 and look at the plot each time.
2. Add more houses to `X` and `y`.
3. Make the prices depend on **rooms** instead of area and see the tree use feature 1.
4. Set `min_samples=4` and see how the tree gets smaller.

---

## 13. Limitations of this simple version

- It is slow on big datasets, because it tries every value of every column.
- It has no handling for missing values.
- A single tree can overfit. Real projects often use many trees together
  (Random Forest or Gradient Boosting).

For learning, though, this version shows exactly how the idea works.