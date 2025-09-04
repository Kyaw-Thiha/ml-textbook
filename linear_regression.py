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
where $(w.x_i + b)$ is the predicted $y'_i$

Then, the energy function is 
E(w, b) = $\sum_{i=1} (e_i)^2$
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
# |%%--%%| <VMzPBVIIkw|T5ovjlz7SQ>

input_dimension = 2
points_num = 10
low = 0
high = 5
x = np.random.randint(low=low, high=high, size=(points_num, input_dimension))
# y = np.random.randint(low=low, high=high, size=(points_num, 1))
y = np.random.randint(low=low, high=high, size=points_num)
print(f"x: {x}")
print(f"y: {y}")

new_column = np.ones((points_num, 1), dtype=int)
x_tilde = np.hstack([x, new_column])
# y_tilde = np.hstack([y, new_column])
print(f"x_tilde: {x_tilde}")
# print(f"y_tilde: {y_tilde}")

x_transpose = np.transpose(x_tilde)
x_pseudoinverse = np.linalg.inv((x_transpose @ x_tilde)) @ x_transpose
w_tilde = x_pseudoinverse @ y
print(f"w_tilde: {w_tilde}")

w_0 = w_tilde[0]
w_1 = w_tilde[1]
b = w_tilde[-1]

# |%%--%%| <T5ovjlz7SQ|3vKRjQy6EZ>

x1_coord = list(range(low, high))
x2_coord = list(range(low, high))
x1_best, x2_best = np.meshgrid(x1_coord, x2_coord)
y_best = w_0 * x1_best + w_1 * x2_best + b

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=y, mode="markers", name="points"))
fig.add_trace(
    go.Surface(
        x=x1_best,
        y=x2_best,
        z=y_best,
        name=f"fit: y={w_0:.2f}+{w_1:.2f}x1+{b:.2f}x2",
        opacity=0.6,
    )
)
fig.update_layout(
    scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
    title="3D points + fitted plane",
)
fig.show()

# |%%--%%| <3vKRjQy6EZ|zRSljPcpa9>
