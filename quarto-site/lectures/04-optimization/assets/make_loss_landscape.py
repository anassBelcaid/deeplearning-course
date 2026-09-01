"""Generate the two-parameter classifier visuals used in Section 1."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
BLUE, CORAL, INK, MUTED = "#2377c9", "#ef5b3f", "#17212b", "#657687"

X = np.array([
    [-1.8, 1.2], [-1.4, .25], [-1.05, 1.65], [-.55, .8], [-.15, 1.55],
    [.20, -1.45], [.55, -.55], [1.0, -1.5], [1.45, -.25], [1.8, -1.1],
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def loss(w):
    p = sigmoid(X @ np.asarray(w))
    return -np.mean(y*np.log(p + 1e-10) + (1-y)*np.log(1-p + 1e-10))


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.edgecolor": "#b9c7d3", "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })


def draw_data(ax, w):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=75, c=BLUE, edgecolors="white", linewidth=1.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=75, c=CORAL, edgecolors="white", linewidth=1.5)
    xx = np.array([-2.2, 2.2])
    if abs(w[1]) > 1e-8:
        ax.plot(xx, -w[0]/w[1]*xx, color=INK, lw=3)
    else:
        ax.axvline(0, color=INK, lw=3)
    ax.set(xlim=(-2.2, 2.2), ylim=(-2.05, 2.05), xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.grid(False)


def boundaries():
    candidates = [((.25, .35), "poor"), ((.9, -.35), "better"), ((1.6, -1.5), "strong")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.15), facecolor="white")
    for ax, (w, label) in zip(axes, candidates):
        draw_data(ax, w)
        ax.set_title(label.upper(), color=MUTED, fontsize=12, fontweight="bold", pad=10)
        ax.text(.5, -.13, rf"$w=({w[0]:.2g},{w[1]:.2g})$  ·  $\mathcal{{L}}={loss(w):.2f}$",
                transform=ax.transAxes, ha="center", va="top", color=INK, fontsize=14)
    fig.subplots_adjust(left=.025, right=.985, top=.88, bottom=.17, wspace=.12)
    fig.savefig(OUT / "boundary-candidates.svg", transparent=True)
    plt.close(fig)
    for index, (w, label) in enumerate(candidates, 1):
        fig, ax = plt.subplots(figsize=(4.4, 4.25), facecolor="white")
        draw_data(ax, w)
        ax.set_title(label.upper(), color=MUTED, fontsize=12, fontweight="bold", pad=10)
        ax.text(.5, -.13, rf"$w=({w[0]:.2g},{w[1]:.2g})$  ·  $\mathcal{{L}}={loss(w):.2f}$",
                transform=ax.transAxes, ha="center", va="top", color=INK, fontsize=14)
        fig.subplots_adjust(left=.04, right=.96, top=.87, bottom=.18)
        fig.savefig(OUT / f"boundary-{index}-{label}.svg", transparent=True)
        plt.close(fig)


grid = np.linspace(-2.6, 2.6, 150)
W1, W2 = np.meshgrid(grid, grid)
Z = np.empty_like(W1)
for i in range(len(grid)):
    for j in range(len(grid)):
        Z[i, j] = loss((W1[i, j], W2[i, j]))


def contour_base(ax):
    levels = np.linspace(Z.min(), min(2.0, Z.max()), 13)
    cf = ax.contourf(W1, W2, Z, levels=levels, cmap="Blues_r", alpha=.95, extend="max")
    ax.contour(W1, W2, Z, levels=levels, colors="#426b8a", linewidths=.65, alpha=.65)
    ax.set(xlabel="$w_1$", ylabel="$w_2$", xlim=(-2.6, 2.6), ylim=(-2.6, 2.6))
    ax.set_aspect("equal")
    return cf


def contours():
    fig, ax = plt.subplots(figsize=(7.1, 5.8), facecolor="white")
    contour_base(ax)
    current = np.array([.25, .35])
    ax.scatter(*current, s=150, c=CORAL, edgecolors="white", linewidth=2.2, zorder=5)
    fig.tight_layout()
    fig.savefig(OUT / "loss-contours.svg", transparent=True)
    plt.close(fig)


def sampled_landscape():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), facecolor="white")
    samples = np.array([[-2, -2], [-2, 0], [-2, 2], [0, -2], [0, 0], [0, 2], [2, -2], [2, 0], [2, 2]])
    axes[0].scatter(samples[:, 0], samples[:, 1], c=[loss(p) for p in samples], cmap="Blues_r", s=180,
                    edgecolors=INK, linewidth=1.2)
    for p in samples:
        axes[0].text(p[0], p[1]-.28, f"{loss(p):.2f}", ha="center", va="top", fontsize=10, color=INK)
    axes[0].set(xlabel="$w_1$", ylabel="$w_2$", xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), title="SAMPLE PARAMETER SETTINGS")
    axes[0].set_aspect("equal")
    contour_base(axes[1])
    axes[1].set_title("CONNECT EQUAL-LOSS REGIONS", fontsize=13, color=MUTED, fontweight="bold")
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "landscape-construction.svg", transparent=True)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8, 5.25), facecolor="white")
    ax.scatter(samples[:, 0], samples[:, 1], c=[loss(p) for p in samples], cmap="Blues_r", s=180,
               edgecolors=INK, linewidth=1.2)
    for p in samples:
        ax.text(p[0], p[1]-.28, f"{loss(p):.2f}", ha="center", va="top", fontsize=10, color=INK)
    ax.set(xlabel="$w_1$", ylabel="$w_2$", xlim=(-2.6, 2.6), ylim=(-2.6, 2.6))
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "landscape-samples.svg", transparent=True)
    plt.close(fig)


def choices():
    fig, ax = plt.subplots(figsize=(7.1, 5.8), facecolor="white")
    contour_base(ax)
    start = np.array([.25, .35])
    ax.scatter(*start, s=150, c=CORAL, edgecolors="white", linewidth=2.2, zorder=6)
    moves = {"A": np.array([-.9, .65]), "B": np.array([.25, 1.45]), "C": np.array([1.15, -.7])}
    for label, end in moves.items():
        delta = end - start
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", lw=3.2, color="#7258a8", mutation_scale=19), zorder=7)
        ax.text(*(end + np.array([.08, .08])), label, fontsize=16, fontweight="bold", color="#7258a8", zorder=8)
    fig.tight_layout()
    fig.savefig(OUT / "candidate-moves.svg", transparent=True)
    plt.close(fig)


def surface():
    fig = plt.figure(figsize=(8.2, 5.7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    step = 3
    ax.plot_surface(W1[::step, ::step], W2[::step, ::step], Z[::step, ::step], cmap="Blues_r",
                    linewidth=.2, edgecolor="#315774", alpha=.96)
    ax.set(xlabel="$w_1$", ylabel="$w_2$", zlabel="loss")
    ax.view_init(elev=34, azim=-57)
    ax.set_zlim(0, 2.2)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "loss-surface.svg", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    setup()
    boundaries()
    contours()
    sampled_landscape()
    choices()
    surface()
