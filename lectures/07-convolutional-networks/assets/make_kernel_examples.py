"""Create the fixed-kernel examples used in the convolution lecture."""

from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).parent
SOURCE = HERE / "kernel-source-lighthouse.jpg"


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply one 3x3 kernel independently to each image channel."""
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    for row in range(3):
        for col in range(3):
            result += kernel[row, col] * padded[
                row : row + image.shape[0], col : col + image.shape[1]
            ]
    return result


photo = Image.open(SOURCE).convert("RGB")
width, height = photo.size
crop_height = int(width * 0.72)
top = max(0, (height - crop_height) // 2 - 80)
photo = photo.crop((0, top, width, top + crop_height)).resize((720, 520))
pixels = np.asarray(photo, dtype=np.float32)

kernels = {
    "smooth": np.ones((3, 3), dtype=np.float32) / 9,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
}

for name, kernel in kernels.items():
    output = np.clip(convolve(pixels, kernel), 0, 255).astype(np.uint8)
    Image.fromarray(output).save(HERE / f"kernel-{name}.jpg", quality=94)

gray = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
gray_rgb = np.repeat(gray[:, :, None], 3, axis=2)
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
edges = np.abs(convolve(gray_rgb, edge_kernel))
edges = np.clip(edges * 1.7, 0, 255).astype(np.uint8)
Image.fromarray(edges).save(HERE / "kernel-edges.jpg", quality=94)
