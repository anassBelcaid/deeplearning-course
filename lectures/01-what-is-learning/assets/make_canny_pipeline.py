"""Build the six-stage Canny teaching figure used in Chapter 1."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    result = np.zeros_like(magnitude)
    direction = (np.rad2deg(angle) + 180) % 180
    for row in range(1, magnitude.shape[0] - 1):
        for col in range(1, magnitude.shape[1] - 1):
            theta = direction[row, col]
            if theta < 22.5 or theta >= 157.5:
                before, after = magnitude[row, col - 1], magnitude[row, col + 1]
            elif theta < 67.5:
                before, after = magnitude[row - 1, col + 1], magnitude[row + 1, col - 1]
            elif theta < 112.5:
                before, after = magnitude[row - 1, col], magnitude[row + 1, col]
            else:
                before, after = magnitude[row - 1, col - 1], magnitude[row + 1, col + 1]
            if magnitude[row, col] >= before and magnitude[row, col] >= after:
                result[row, col] = magnitude[row, col]
    return result


image = cv2.imread(str(HERE / "canny-source.jpg"))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(gx, gy)
magnitude = np.uint8(255 * magnitude / max(magnitude.max(), 1))
nms = non_maximum_suppression(magnitude, np.arctan2(gy, gx))

high = np.percentile(nms[nms > 0], 82)
low = high * 0.42
thresholds = np.zeros_like(nms, dtype=np.uint8)
thresholds[nms >= high] = 255
thresholds[(nms >= low) & (nms < high)] = 105
edges = cv2.Canny(blurred, low, high)

panels = [gray, blurred, magnitude, nms, thresholds, edges]
titles = [
    "1 · grayscale",
    "2 · Gaussian blur",
    "3 · gradient magnitude",
    "4 · non-max suppression",
    "5 · double threshold",
    "6 · hysteresis",
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.5), facecolor="#f5f1e8")
for axis, panel, title in zip(axes.flat, panels, titles):
    axis.imshow(panel, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title, color="#17212b", fontweight="bold", pad=8)
    axis.axis("off")
fig.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.02, wspace=0.06, hspace=0.18)
fig.savefig(HERE / "canny-six-stages.png", dpi=180, facecolor=fig.get_facecolor())
