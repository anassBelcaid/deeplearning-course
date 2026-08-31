"""Generate the motivating target figure for Task 8 using a small NumPy MLP."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_moons(n=120, noise=0.10, seed=4):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    upper = np.c_[np.cos(t1), np.sin(t1)]
    lower = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    data = np.vstack([upper, lower]) + rng.normal(0, noise, (n, 2))
    labels = np.r_[np.zeros(n1), np.ones(n - n1)]
    order = rng.permutation(n)
    return data[order], labels[order]


rng = np.random.default_rng(42)
X, y = make_moons()
X = (X - X.mean(axis=0)) / X.std(axis=0)
W1 = rng.normal(0, 0.65, (2, 16))
b1 = np.zeros((1, 16))
W2 = rng.normal(0, 0.35, (16, 16))
b2 = np.zeros((1, 16))
W3 = rng.normal(0, 0.35, (16, 1))
b3 = np.zeros((1, 1))
history = []

for epoch in range(700):
    h1 = np.maximum(0, X @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    eps = 1e-9
    loss = -np.mean(y[:, None] * np.log(probabilities + eps) + (1-y[:, None]) * np.log(1-probabilities + eps))
    history.append(loss)

    dz3 = (probabilities - y[:, None]) / len(X)
    dW3, db3 = h2.T @ dz3, dz3.sum(axis=0, keepdims=True)
    dz2 = (dz3 @ W3.T) * (h2 > 0)
    dW2, db2 = h1.T @ dz2, dz2.sum(axis=0, keepdims=True)
    dz1 = (dz2 @ W2.T) * (h1 > 0)
    dW1, db1 = X.T @ dz1, dz1.sum(axis=0, keepdims=True)
    lr = 0.12 * (1 - 0.75 * epoch / 700)
    W1 -= lr*dW1; b1 -= lr*db1
    W2 -= lr*dW2; b2 -= lr*db2
    W3 -= lr*dW3; b3 -= lr*db3

xs = np.linspace(X[:, 0].min() - .45, X[:, 0].max() + .45, 180)
ys = np.linspace(X[:, 1].min() - .45, X[:, 1].max() + .45, 180)
xx, yy = np.meshgrid(xs, ys)
grid = np.c_[xx.ravel(), yy.ravel()]
gh1 = np.maximum(0, grid @ W1 + b1)
gh2 = np.maximum(0, gh1 @ W2 + b2)
scores = (1 / (1 + np.exp(-np.clip(gh2 @ W3 + b3, -40, 40)))).reshape(xx.shape)
accuracy = np.mean((probabilities.ravel() >= .5) == y)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
axes[0].contourf(xx, yy, scores, levels=np.linspace(0, 1, 13), cmap="coolwarm", alpha=.7)
axes[0].contour(xx, yy, scores, levels=[.5], colors="#17212b", linewidths=2.5)
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="white", linewidth=.8, s=43)
axes[0].set_title(f"A nonlinear boundary · accuracy {accuracy:.0%}", weight="bold")
axes[0].set_xlabel("feature 1"); axes[0].set_ylabel("feature 2")
axes[1].plot(history, color="#2377c9", linewidth=3)
axes[1].set_title("Training makes BCE fall", weight="bold")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("binary cross-entropy")
fig.suptitle("The destination: a classifier powered by your Value class", fontsize=16, weight="bold")
fig.savefig(Path(__file__).with_name("expected-two-moons.png"), dpi=180, facecolor="white")
