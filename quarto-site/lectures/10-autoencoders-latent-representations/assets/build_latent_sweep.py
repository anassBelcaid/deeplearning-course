"""Train a controlled Fashion-MNIST autoencoder latent-width sweep.

This script is intentionally self-contained and runs unchanged in Colab after
installing PyTorch and TorchVision.  It trains the same MLP autoencoder for each
latent width, evaluates every model on the test split, and exports small PNG
assets plus a JSON manifest for the lecture's browser-based explorer.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


LATENT_DIMS = (2, 4, 8, 16, 32, 64, 128)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def representative_indices(targets: torch.Tensor) -> list[int]:
    """Return the first test example for each Fashion-MNIST class."""
    return [int((targets == label).nonzero(as_tuple=True)[0][0]) for label in range(10)]


def save_gray(tensor: torch.Tensor, path: Path) -> None:
    pixels = tensor.detach().cpu().squeeze().clamp(0, 1).mul(255).round().byte().numpy()
    Image.fromarray(pixels, mode="L").save(path, optimize=True)


def save_error(original: torch.Tensor, reconstruction: torch.Tensor, path: Path) -> None:
    error = (original - reconstruction).abs().detach().cpu().squeeze().clamp(0, 1).numpy()
    # A dark-purple to warm-yellow map keeps zero error quiet and large errors visible.
    stops = np.array([[28, 18, 59], [95, 44, 128], [214, 73, 92], [252, 214, 99]])
    position = error * (len(stops) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(stops) - 1)
    mix = (position - low)[..., None]
    rgb = stops[low] * (1 - mix) + stops[high] * mix
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path, optimize=True)


def train_one(
    latent_dim: int,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    seed: int,
) -> Autoencoder:
    # Reset the seed so latent width is the controlled architectural change.
    seed_everything(seed)
    model = Autoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(images)
            loss = criterion(reconstruction, images)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        print(
            f"d={latent_dim:>3} epoch={epoch + 1}/{epochs} "
            f"train_mse={running / len(train_loader.dataset):.6f}",
            flush=True,
        )
    return model


@torch.inference_mode()
def evaluate(model: Autoencoder, loader: DataLoader, device: torch.device) -> float:
    squared_error = 0.0
    pixels = 0
    model.eval()
    for images, _ in loader:
        images = images.to(device)
        reconstruction = model(images)
        squared_error += (reconstruction - images).square().sum().item()
        pixels += images.numel()
    return squared_error / pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-limit", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/fashion-mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "latent-sweep")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}")

    seed_everything(args.seed)
    transform = transforms.ToTensor()
    full_train = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)
    train_data = (
        full_train
        if args.train_limit >= len(full_train)
        else Subset(full_train, range(args.train_limit))
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        persistent_workers=True,
    )
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = representative_indices(test_data.targets)
    sample_images = torch.stack([test_data[index][0] for index in sample_indices])
    class_names = list(test_data.classes)
    for sample, image in enumerate(sample_images):
        save_gray(image, args.output_dir / f"sample-{sample}-input.png")

    manifest: dict[str, object] = {
        "dataset": "Fashion-MNIST test split",
        "seed": args.seed,
        "epochs": args.epochs,
        "train_examples": len(train_data),
        "input_dimensions": 784,
        "latent_dimensions": list(LATENT_DIMS),
        "samples": [
            {"id": sample, "class": class_names[label], "test_index": index}
            for sample, (label, index) in enumerate(zip(range(10), sample_indices))
        ],
        "models": {},
    }

    for latent_dim in LATENT_DIMS:
        model = train_one(latent_dim, train_loader, args.epochs, device, args.seed)
        test_mse = evaluate(model, test_loader, device)
        model.eval()
        with torch.inference_mode():
            reconstructions = model(sample_images.to(device)).cpu()
        sample_mse = (reconstructions - sample_images).square().flatten(1).mean(1)
        for sample, (original, reconstruction) in enumerate(zip(sample_images, reconstructions)):
            stem = f"d-{latent_dim}-sample-{sample}"
            save_gray(reconstruction, args.output_dir / f"{stem}-reconstruction.png")
            save_error(original, reconstruction, args.output_dir / f"{stem}-error.png")
        manifest["models"][str(latent_dim)] = {
            "test_mse": round(test_mse, 7),
            "coordinate_ratio": round(latent_dim / 784, 6),
            "sample_mse": [round(float(value), 7) for value in sample_mse],
        }
        print(f"d={latent_dim:>3} test_mse={test_mse:.7f}", flush=True)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
