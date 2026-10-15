# Regression Fundamentals: Notes with Diagrams

## 1. The Machine Learning Process

```mermaid
flowchart LR; A[Data Pre-Processing] --> B[Modelling] --> C[Evaluation]; A --- A1[Import, clean, split train/test, scale]; B --- B1[Build, train, predict]; C --- C1[Metrics and verdict]
```

## 2. Feature Scaling

```
Normalization:   X' = (X - Xmin) / (Xmax - Xmin)   -> [0, 1]
Standardization: X' = (X - mu) / sigma             -> about [-3, +3]
```

The same data points on three scales. The shape is preserved; only the units change.

## 3. Simple Linear Regression

```
y_hat = b0 + b1 * X1
```

b0 is where the line crosses the y-axis, b1 is the slope, and the residual is the gap between a point and the line.

## 4. R Squared

```
SS_res = SUM (y_i - y_hat_i)^2
SS_tot = SUM (y_i - y_avg)^2
R^2 = 1 - SS_res / SS_tot
```

Left: errors around the fitted line. Right: errors around the plain average. The smaller the left errors are compared with the right, the closer R squared is to 1.

| R squared | Meaning |
| --- | --- |
| 1.0 | Perfect (suspicious) |
| about 0.9 | Very good |
| below 0.7 | Not great |
| below 0.4 | Terrible |
| below 0 | Model makes no sense |

## 5. Adjusted R Squared

```
Adj R^2 = 1 - (1 - R^2) * (n - 1) / (n - k - 1)
k = number of independent variables, n = sample size
```

R squared keeps rising as variables are added, even useless ones. Adjusted R squared drops when added variables do not help.

## 6. Assumptions of Linear Regression

How each assumption looks when it is violated. A healthy dataset shows a clean straight band of points, equal spread, bell-shaped errors, and no pattern in order.

## 7. Dummy Variables

| State | New York | California |
| --- | --- | --- |
| New York | 1 | 0 |
| California | 0 | 1 |
| Florida | 0 | 0 |

```
y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*D1
```

```mermaid
flowchart LR; A[Categorical column with m categories] --> B[One-hot encode into m columns] --> C[Drop one column] --> D[Use m-1 dummy variables]; B -. keeping all .-> E[Dummy variable trap: perfect multicollinearity]
```

## 8. Building a Model

```mermaid
flowchart TD; M[5 methods] --> A[1. All-in]; M --> S[Stepwise Regression]; M --> E[5. Score Comparison]; S --> B[2. Backward Elimination]; S --> F[3. Forward Selection]; S --> D[4. Bidirectional Elimination]
```

### Backward elimination

```mermaid
flowchart TD; A[Choose significance level SL, e.g. 0.05] --> B[Fit model with all predictors]; B --> C[Find predictor with highest p-value]; C --> D{{p greater than SL?}}; D -- Yes --> E[Remove that predictor] --> B; D -- No --> F[Final model]
```

### Forward selection

```mermaid
flowchart TD; A[Choose SL to enter] --> B[Fit all simple regressions, keep the lowest p-value]; B --> C[Add each remaining variable to current model]; C --> D[Pick the one with lowest p-value]; D --> E{{p less than SL?}}; E -- Yes --> C; E -- No --> F[Keep previous model]
```

### Bidirectional elimination

```mermaid
flowchart TD; A[Choose SL enter and SL stay] --> B[Forward step: add best new variable if p below SL enter]; B --> C[Backward steps: remove old variables with p above SL stay]; C --> D{{Any variable can enter or leave?}}; D -- Yes --> B; D -- No --> E[Final model]
```

### Score comparison

```mermaid
flowchart LR; A[Choose criterion, e.g. Akaike] --> B[Build all 2^N - 1 models] --> C[Pick the best score]
```

## 9. Polynomial Regression

```
y = b0 + b1*x + b2*x^2 + ... + bn*x^n
```

Linear in its coefficients (the b terms), but nonlinear in its input variable x. The coefficients can still be found with least squares, while the fitted curve bends.

Too low a degree underfits, a suitable degree captures the trend, and a very high degree chases noise (overfitting).