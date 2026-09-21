"""Draw slide 5 as one stable, self-contained figure."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


fig, ax = plt.subplots(figsize=(13.4, 5.3))
ax.set_xlim(0, 13.4)
ax.set_ylim(0, 5.3)
ax.axis("off")
fig.patch.set_facecolor("#f4f0e7")


def panel(x, width):
    ax.add_patch(FancyBboxPatch(
        (x, .18), width, 4.90,
        boxstyle="round,pad=.025,rounding_size=.18",
        facecolor="white", edgecolor="#d2e0ea", linewidth=1.7,
    ))


def node(x, latex):
    ax.add_patch(FancyBboxPatch(
        (x, 1.78), 1.35, .90,
        boxstyle="round,pad=.02,rounding_size=.13",
        facecolor="#edf6ff", edgecolor="#c7ddeb", linewidth=1.6,
    ))
    ax.text(x + .675, 2.23, latex, ha="center", va="center",
            fontsize=23, color="#245d8d")


def edge(x1, x2, label):
    ax.add_patch(FancyArrowPatch(
        (x1, 2.23), (x2, 2.23), arrowstyle="-|>", mutation_scale=13,
        linewidth=2.2, color="#ef5b3f", shrinkA=3, shrinkB=3,
    ))
    ax.text((x1 + x2) / 2, 2.55, label, ha="center", va="bottom",
            fontsize=16, color="#ef5b3f", fontweight="bold")


panel(.08, 6.30)
panel(6.58, 6.74)

ax.text(.35, 4.65, r"LAYER $\ell$", fontsize=17, color="#263440")
ax.text(3.23, 3.72,
        r"$z^{(\ell)}=W^{(\ell)}h^{(\ell-1)}$     $h^{(\ell)}=\tanh(z^{(\ell)})$",
        ha="center", fontsize=21, color="#17212b")
node(.48, r"$h^{(\ell-1)}$")
node(2.56, r"$z^{(\ell)}$")
node(4.64, r"$h^{(\ell)}$")
edge(1.83, 2.56, r"$W^{(\ell)}$")
edge(3.91, 4.64, r"tanh")

ax.text(6.88, 4.65, "ASK OF EVERY LAYER", fontsize=17, color="#263440")
ax.text(6.88, 4.15, "Does its output retain a useful spread?",
        fontsize=18, color="#17212b", fontweight="bold")

def mini_histogram(x, y, heights, color):
    """Draw a tiny distribution with a shared baseline and fixed bin geometry."""
    width, gap, scale = .17, .035, .82
    ax.plot([x - .03, x + len(heights) * (width + gap)], [y, y],
            color="#a5b2bd", linewidth=1.0)
    for index, height in enumerate(heights):
        ax.add_patch(Rectangle(
            (x + index * (width + gap), y), width, height * scale,
            facecolor=color, edgecolor="white", linewidth=.55,
        ))


rows = [
    (3.22, .14, "#2377c9", "collapsed", [.01, .02, .05, .18, 1, .18, .05, .02, .01]),
    (2.02, .27, "#29845a", "useful spread", [.06, .16, .34, .58, .78, .58, .34, .16, .06]),
    (.82, .40, "#ef5b3f", "saturated", [1, .48, .16, .05, .02, .05, .16, .48, 1]),
]
for y, radius, color, label, heights in rows:
    ax.add_patch(Circle((7.35, y + .25), radius + .08, facecolor=color, alpha=.14, edgecolor="none"))
    ax.add_patch(Circle((7.35, y + .25), radius, facecolor=color, edgecolor="white", linewidth=1.5))
    mini_histogram(7.95, y - .08, heights, color)
    ax.text(10.05, y + .25, label, va="center", fontsize=15, color=color, fontweight="bold")

target = Path(__file__).with_name("one-layer-rescaling.svg")
fig.savefig(target, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Wrote {target}")
