"""Build interactive latent-space evidence for the autoencoder lecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from build_latent_sweep import Autoencoder, seed_everything, train_one


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def make_sprite(images: torch.Tensor, columns: int, path: Path) -> None:
    images = images.detach().cpu().squeeze(1).clamp(0, 1).mul(255).round().byte().numpy()
    rows = int(np.ceil(len(images) / columns))
    sprite = np.zeros((rows * 28, columns * 28), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sprite[row * 28:(row + 1) * 28, column * 28:(column + 1) * 28] = image
    Image.fromarray(sprite, mode="L").save(path, optimize=True)


def select_balanced(targets: torch.Tensor, per_class: int) -> list[int]:
    indices: list[int] = []
    for label in range(10):
        matches = (targets == label).nonzero(as_tuple=True)[0][:per_class]
        indices.extend(int(index) for index in matches)
    return indices


def normalise_for_canvas(z: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    low = np.quantile(z, 0.01, axis=0)
    high = np.quantile(z, 0.99, axis=0)
    scaled = (z - low) / np.maximum(high - low, 1e-8)
    scaled = np.clip(scaled, 0, 1)
    return scaled, {"low": low.tolist(), "high": high.tolist()}


def save_interpolation(
    model: Autoencoder,
    first: torch.Tensor,
    second: torch.Tensor,
    device: torch.device,
    path: Path,
    steps: int = 9,
) -> list[list[float]]:
    model.eval()
    with torch.inference_mode():
        endpoints = model.encoder(torch.stack([first, second]).to(device))
        weights = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
        codes = (1 - weights) * endpoints[0] + weights * endpoints[1]
        decoded = model.decoder(codes).cpu()
    make_sprite(decoded, steps, path)
    return codes.cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--points-per-class", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/fashion-mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "latent-geometry")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    seed_everything(args.seed)
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        persistent_workers=True,
    )
    model = train_one(2, train_loader, args.epochs, device, args.seed)
    model.eval()

    selected_indices = select_balanced(test_data.targets, args.points_per_class)
    images = torch.stack([test_data[index][0] for index in selected_indices])
    labels = torch.tensor([int(test_data.targets[index]) for index in selected_indices])
    with torch.inference_mode():
        codes = model.encoder(images.to(device)).cpu()
        reconstructions = model.decoder(codes.to(device)).cpu()
    scaled, bounds = normalise_for_canvas(codes.numpy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = 40
    make_sprite(images, columns, args.output_dir / "inputs-sprite.png")
    make_sprite(reconstructions, columns, args.output_dir / "reconstructions-sprite.png")

    # Interpolate between a sneaker and an ankle boot, then between a shirt and a coat.
    selected_labels = labels.tolist()
    sneaker = selected_labels.index(7)
    ankle_boot = selected_labels.index(9)
    shirt = selected_labels.index(6)
    coat = selected_labels.index(4)
    paths = [
        {
            "name": "sneaker → ankle boot",
            "start": sneaker,
            "end": ankle_boot,
            "file": "interpolation-footwear.png",
            "codes": save_interpolation(
                model, images[sneaker], images[ankle_boot], device,
                args.output_dir / "interpolation-footwear.png",
            ),
        },
        {
            "name": "shirt → coat",
            "start": shirt,
            "end": coat,
            "file": "interpolation-tops.png",
            "codes": save_interpolation(
                model, images[shirt], images[coat], device,
                args.output_dir / "interpolation-tops.png",
            ),
        },
    ]

    points = []
    per_sample_mse = (reconstructions - images).square().flatten(1).mean(1)
    for position, (dataset_index, label) in enumerate(zip(selected_indices, labels.tolist())):
        points.append({
            "id": position,
            "dataset_index": dataset_index,
            "label": label,
            "class": CLASS_NAMES[label],
            "x": round(float(scaled[position, 0]), 6),
            "y": round(float(scaled[position, 1]), 6),
            "z": [round(float(value), 6) for value in codes[position]],
            "mse": round(float(per_sample_mse[position]), 7),
        })

    manifest = {
        "dataset": "Fashion-MNIST test split",
        "latent_dimension": 2,
        "seed": args.seed,
        "epochs": args.epochs,
        "points_per_class": args.points_per_class,
        "sprite_columns": columns,
        "sprite_cell_size": 28,
        "normalisation": bounds,
        "classes": CLASS_NAMES,
        "points": points,
        "interpolations": paths,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(model.state_dict(), args.output_dir / "autoencoder-d2.pt")
    print(f"wrote {args.output_dir} with {len(points)} latent points", flush=True)


if __name__ == "__main__":
    main()
