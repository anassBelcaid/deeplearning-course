"""Generate reproducible expected optimizer trajectories for the lab page."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)
START = np.array([-3.6, 3.1])
A = np.array([[1.4, 1.25], [1.25, 1.4]])


class Landscape:
    def __init__(self, n=64, seed=17):
        rng = np.random.default_rng(seed)
        self.offsets = rng.normal(size=(n, 2)) @ np.diag([0.48, 0.16])
        self.offsets -= self.offsets.mean(axis=0, keepdims=True)
        self.n = n

    def value(self, theta):
        x, y = theta
        return 0.5 * theta @ A @ theta + 0.08 * (1 - np.cos(3 * x))

    def grad(self, theta, indices):
        noise = self.offsets[indices].mean(axis=0)
        return A @ theta + np.array([0.24 * np.sin(3 * theta[0]), 0.0]) + noise


class GD:
    def __init__(self, lr): self.lr = lr
    def reset(self): pass
    def query(self, theta): return theta
    def step(self, theta, grad): return theta - self.lr * grad


class Momentum(GD):
    def __init__(self, lr, beta=.9): super().__init__(lr); self.beta = beta
    def reset(self): self.v = np.zeros(2)
    def step(self, theta, grad):
        self.v = self.beta * self.v + grad
        return theta - self.lr * self.v


class Nesterov(Momentum):
    def query(self, theta): return theta - self.lr * self.beta * self.v


class AdaGrad(GD):
    def reset(self): self.s = np.zeros(2)
    def step(self, theta, grad):
        self.s += grad * grad
        return theta - self.lr * grad / (np.sqrt(self.s) + 1e-8)


class RMSProp(GD):
    def __init__(self, lr, rho=.9): super().__init__(lr); self.rho = rho
    def reset(self): self.s = np.zeros(2)
    def step(self, theta, grad):
        self.s = self.rho * self.s + (1 - self.rho) * grad * grad
        return theta - self.lr * grad / (np.sqrt(self.s) + 1e-8)


class Adam(GD):
    def reset(self): self.m = np.zeros(2); self.v = np.zeros(2); self.t = 0
    def step(self, theta, grad):
        self.t += 1
        self.m = .9 * self.m + .1 * grad
        self.v = .999 * self.v + .001 * grad * grad
        mh = self.m / (1 - .9 ** self.t)
        vh = self.v / (1 - .999 ** self.t)
        return theta - self.lr * mh / (np.sqrt(vh) + 1e-8)


landscape = Landscape()
rng = np.random.default_rng(2026)
batches = [rng.choice(landscape.n, size=8, replace=False) for _ in range(70)]
optimizers = {
    "Gradient descent": GD(.18),
    "Momentum": Momentum(.075),
    "Nesterov": Nesterov(.075),
    "AdaGrad": AdaGrad(.65),
    "RMSProp": RMSProp(.09),
    "Adam": Adam(.12),
}


def run(opt):
    theta = START.copy()
    opt.reset()
    path = [theta.copy()]
    for batch in batches:
        grad = landscape.grad(opt.query(theta), batch)
        theta = opt.step(theta, grad)
        path.append(theta.copy())
    return np.asarray(path)


axis = np.linspace(-4.2, 4.2, 260)
X, Y = np.meshgrid(axis, axis)
Z = 0.5 * (A[0, 0]*X**2 + 2*A[0, 1]*X*Y + A[1, 1]*Y**2) + .08*(1-np.cos(3*X))

for slug, (name, opt) in zip(
    ["gradient-descent", "momentum", "nesterov", "adagrad", "rmsprop", "adam"],
    optimizers.items(),
):
    path = run(opt)
    fig, ax = plt.subplots(figsize=(5.2, 4.35))
    ax.contourf(X, Y, Z, levels=28, cmap="magma_r", alpha=.96)
    ax.contour(X, Y, Z, levels=17, colors="white", linewidths=.35, alpha=.35)
    ax.plot(path[:, 0], path[:, 1], color="#22d3ee", lw=2.1)
    ax.scatter(path[::5, 0], path[::5, 1], s=13, color="#fb7185", zorder=3)
    ax.plot(*START, "o", color="#fb7185", ms=7)
    ax.plot(0, 0, marker="*", color="#facc15", ms=13, mec="#172033")
    ax.set(xlim=(-4.2, 4.2), ylim=(-4.2, 4.2), title=name,
           xlabel=r"$\theta_1$", ylabel=r"$\theta_2$")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / f"{slug}.png", dpi=150, facecolor="white")
    plt.close(fig)

print(f"Generated {len(optimizers)} previews in {OUT}")
