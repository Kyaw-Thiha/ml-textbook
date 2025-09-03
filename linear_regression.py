r"""°°°
# Linear Regression
This is basically trying to find a line of best fit in a multi-dimenensional data.
°°°"""
# |%%--%%| <f2zYnEOzAl|YskWEeMKcA>

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# |%%--%%| <YskWEeMKcA|WJdsVDiTKZ>
r"""°°°
## 1D-Case
$e_i$ = $y_i - (w.x_i + b)$
where (w.x_i + b) is the predicted y'_i

Then, the energy function is 
E(w, b) = summation of (e_i)^2
°°°"""
# |%%--%%| <WJdsVDiTKZ|w753NQfnjj>

# 1D Case
N = 10
low = 0
high = 5
x = np.random.randint(low=low, high=high, size=N)
y = np.random.randint(low=low, high=high, size=N)

# Average over x
x_hat: float = 0
for x_i in x:
    x_hat += x_i
x_hat = x_hat / N

# Average over y
y_hat: float = 0
for y_i in y:
    y_hat += y_i
y_hat = y_hat / N

# Finding the weight (gradient)
numerator: float = 0
denominator: float = 0
for i in range(N):
    numerator = numerator + ((y[i] - y_hat) * (x[i] - x_hat))
    denominator = denominator + ((x[i] - x_hat) ** 2)
w: float = numerator / denominator

# Finding the bias (y-intercept)
b: float = y_hat - w * x_hat

print("Points")
for x_i, y_i in zip(x, y):
    print(f"({x_i}, {y_i})")

print(f"Weight: {w}")
print(f"Bias: {b}")

# |%%--%%| <w753NQfnjj|FGtuKmSZdK>

x_best = list(range(low, high))
y_best = [w * x_i + b for x_i in x_best]
df = pd.DataFrame({"x": x, "y": y})

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x_best,
        y=y_best,
        mode="lines",
        name=f"Best fit: y={w:.3f}x+{b:.3f}",
        line=dict(color="red"),
    )
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Points",
        marker=dict(size=10, color="blue"),
    )
)

fig.update_layout(title="Linear Regression (1-Dimension)")
fig.show()

# |%%--%%| <FGtuKmSZdK|VMzPBVIIkw>
r"""°°°
Multi-dimensional input
°°°"""
# |%%--%%| <VMzPBVIIkw|zRSljPcpa9>
