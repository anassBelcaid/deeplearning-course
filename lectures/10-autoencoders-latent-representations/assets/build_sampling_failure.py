"""Decode unsupported locations from the trained 2-D autoencoder latent space."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from build_latent_sweep import Autoencoder, seed_everything


ROOT = Path(__file__).parent
GEOMETRY = ROOT / "latent-geometry"
OUTPUT = ROOT / "sampling-failure"


def sprite(images: torch.Tensor, columns: int, path: Path) -> None:
    array = images.detach().cpu().squeeze(1).clamp(0, 1).mul(255).round().byte().numpy()
    rows = int(np.ceil(len(array) / columns))
    canvas = np.zeros((rows * 28, columns * 28), dtype=np.uint8)
    for index, image in enumerate(array):
        row, column = divmod(index, columns)
        canvas[row * 28:(row + 1) * 28, column * 28:(column + 1) * 28] = image
    Image.fromarray(canvas, mode="L").save(path, optimize=True)


def main() -> None:
    seed_everything(73)
    manifest = json.loads((GEOMETRY / "manifest.json").read_text())
    encoded = torch.tensor([point["z"] for point in manifest["points"]])
    low = encoded.quantile(0.01, dim=0)
    high = encoded.quantile(0.99, dim=0)

    model = Autoencoder(2)
    model.load_state_dict(torch.load(GEOMETRY / "autoencoder-d2.pt", map_location="cpu", weights_only=True))
    model.eval()

    # A regular atlas exposes what the decoder produces throughout the bounding box.
    side = 10
    xs = torch.linspace(float(low[0]), float(high[0]), side)
    ys = torch.linspace(float(high[1]), float(low[1]), side)
    grid = torch.cartesian_prod(ys, xs)[:, [1, 0]]
    with torch.inference_mode():
        decoded_grid = model.decoder(grid)

    # Uniform random samples are legal decoder inputs but need not be supported by data.
    generator = torch.Generator().manual_seed(73)
    random_codes = low + torch.rand((250, 2), generator=generator) * (high - low)
    distances = torch.cdist(random_codes, encoded).min(dim=1).values
    ranked = torch.argsort(distances)
    chosen_indices = torch.cat([ranked[:4], ranked[-4:]])
    chosen_codes = random_codes[chosen_indices]
    with torch.inference_mode():
        chosen_images = model.decoder(chosen_codes)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sprite(decoded_grid, side, OUTPUT / "decoder-atlas.png")
    sprite(chosen_images, 8, OUTPUT / "random-probes.png")

    def normalise(codes: torch.Tensor) -> list[dict[str, float]]:
        scaled = (codes - low) / (high - low)
        return [
            {"x": round(float(point[0]), 6), "y": round(float(point[1]), 6)}
            for point in scaled.clamp(0, 1)
        ]

    output = {
        "bounds": {"low": low.tolist(), "high": high.tolist()},
        "encoded_points": normalise(encoded),
        "random_points": [
            {**position, "distance": round(float(distance), 5)}
            for position, distance in zip(normalise(random_codes), distances)
        ],
        "chosen": [
            {
                **position,
                "distance": round(float(distances[index]), 5),
                "support": "near" if rank < 4 else "far",
            }
            for rank, (index, position) in enumerate(zip(chosen_indices.tolist(), normalise(chosen_codes)))
        ],
        "atlas": {"side": side, "x": xs.tolist(), "y": ys.tolist()},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"nearest chosen distances: {[round(float(distances[i]), 3) for i in chosen_indices[:4]]}")
    print(f"farthest chosen distances: {[round(float(distances[i]), 3) for i in chosen_indices[4:]]}")


if __name__ == "__main__":
    main()
