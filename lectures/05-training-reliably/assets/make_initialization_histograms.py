"""Generate the three activation-distribution panels used in Section 1."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"small": "#2377c9", "moderate": "#29845a", "large": "#ef5b3f"}


def draw_activation_histogram(ax, values, title, color):
    """Draw one normalized activation histogram on shared tanh axes."""
    bins = np.linspace(-1, 1, 22)
    weights = np.ones_like(values) / len(values)
    ax.hist(values, bins=bins, weights=weights, color=color, edgecolor="white", linewidth=1.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 0.46)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=13)
    ax.set_xlabel("layer 8 activation", fontsize=11)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, .2, .4])
    ax.set_yticklabels(["0", ".2", ".4"])
    ax.tick_params(labelsize=10, colors="#657687")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#758594")
    ax.grid(axis="y", color="#dce5eb", linewidth=.8, alpha=.8)
    ax.set_axisbelow(True)


rng = np.random.default_rng(2026)
n = 6000
collapsed = np.clip(rng.normal(0, .035, n), -1, 1)
useful = np.clip(rng.normal(0, .36, n), -1, 1)
signs = rng.choice([-1, 1], size=n)
saturated = np.clip(signs * rng.normal(.91, .075, n), -1, 1)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
fig.patch.set_facecolor("white")
specs = [
    (collapsed, "Very small weight scale", COLORS["small"]),
    (useful, "Moderate weight scale", COLORS["moderate"]),
    (saturated, "Very large weight scale", COLORS["large"]),
]
for ax, (values, title, color) in zip(axes, specs):
    draw_activation_histogram(ax, values, title, color)
axes[0].set_ylabel("relative frequency", fontsize=11)
fig.suptitle("Same 8-layer tanh network · activation distribution before training", fontsize=18, y=1.01)
fig.tight_layout(w_pad=2.4)

target = Path(__file__).with_name("initialization-activation-histograms.svg")
fig.savefig(target, bbox_inches="tight", facecolor="white")
plt.close(fig)

# A text-free compact version for the one-layer mechanism slide.
fig, axes = plt.subplots(3, 1, figsize=(5.0, 6.7), sharex=True, sharey=True)
fig.patch.set_facecolor("white")
compact_specs = [
    (collapsed, COLORS["small"]),
    (useful, COLORS["moderate"]),
    (saturated, COLORS["large"]),
]
for ax, (values, color) in zip(axes, compact_specs):
    bins = np.linspace(-1, 1, 22)
    weights = np.ones_like(values) / len(values)
    ax.hist(values, bins=bins, weights=weights, color=color, edgecolor="white", linewidth=1.1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, .46)
    ax.axhline(0, color="#9aa9b6", linewidth=1.2)
    ax.axis("off")
fig.tight_layout(h_pad=.55, pad=.1)

compact_target = Path(__file__).with_name("one-layer-activation-cases.svg")
fig.savefig(compact_target, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {target}\nWrote {compact_target}")
