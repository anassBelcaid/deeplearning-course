"""Generate real denoising and anomaly-detection evidence for Section 05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from build_latent_sweep import Autoencoder, seed_everything


def save_sprite(images: torch.Tensor, path: Path) -> None:
    array = images.detach().cpu().squeeze(1).clamp(0, 1).mul(255).round().byte().numpy()
    canvas = np.zeros((28, 28 * len(array)), dtype=np.uint8)
    for column, image in enumerate(array):
        canvas[:, column * 28:(column + 1) * 28] = image
    Image.fromarray(canvas, mode="L").save(path, optimize=True)


def corrupt(images: torch.Tensor, generator: torch.Generator, sigma: float = 0.45) -> torch.Tensor:
    noise = torch.randn(images.shape, generator=generator) * sigma
    return (images + noise).clamp(0, 1)


def train_denoiser(loader: DataLoader, epochs: int, device: torch.device, seed: int) -> Autoencoder:
    seed_everything(seed)
    model = Autoencoder(16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    noise_generator = torch.Generator().manual_seed(seed + 1)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for clean, _ in loader:
            noisy = corrupt(clean, noise_generator).to(device)
            clean = clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy)
            loss = loss_fn(prediction, clean)
            loss.backward()
            optimizer.step()
            total += loss.item() * clean.size(0)
        print(f"denoiser epoch={epoch + 1}/{epochs} mse={total / len(loader.dataset):.6f}", flush=True)
    return model.eval()


def train_footwear_model(loader: DataLoader, epochs: int, device: torch.device, seed: int) -> Autoencoder:
    seed_everything(seed)
    model = Autoencoder(16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(images)
            loss = loss_fn(reconstruction, images)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
        print(f"footwear epoch={epoch + 1}/{epochs} mse={total / len(loader.dataset):.6f}", flush=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/fashion-mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "useful-reconstruction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    train = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=2)
    denoiser = train_denoiser(loader, args.epochs, device, args.seed)

    # One representative from six visually distinct classes.
    labels = [0, 1, 4, 5, 8, 9]
    indices = [int((test.targets == label).nonzero(as_tuple=True)[0][2]) for label in labels]
    clean = torch.stack([test[index][0] for index in indices])
    noisy = corrupt(clean, torch.Generator().manual_seed(args.seed + 99))
    with torch.inference_mode():
        restored = denoiser(noisy.to(device)).cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_sprite(clean, args.output_dir / "denoise-clean.png")
    save_sprite(noisy, args.output_dir / "denoise-noisy.png")
    save_sprite(restored, args.output_dir / "denoise-restored.png")

    # Train only on footwear, then score both familiar footwear and unseen garments.
    footwear_indices = torch.isin(train.targets, torch.tensor([5, 7, 9])).nonzero(as_tuple=True)[0].tolist()
    footwear_loader = DataLoader(Subset(train, footwear_indices), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=2)
    anomaly_model = train_footwear_model(footwear_loader, args.epochs, device, args.seed + 7)

    candidates = []
    for label in range(10):
        for index in (test.targets == label).nonzero(as_tuple=True)[0][:40].tolist():
            candidates.append((index, label))
    candidate_images = torch.stack([test[index][0] for index, _ in candidates])
    with torch.inference_mode():
        candidate_recon = anomaly_model(candidate_images.to(device)).cpu()
    scores = (candidate_recon - candidate_images).square().flatten(1).mean(1)

    normal_positions = [i for i, (_, label) in enumerate(candidates) if label in (5, 7, 9)]
    anomaly_positions = [i for i, (_, label) in enumerate(candidates) if label not in (5, 7, 9)]
    normal_positions = sorted(normal_positions, key=lambda i: float(scores[i]))[::max(1, len(normal_positions) // 6)][:6]
    anomaly_positions = sorted(anomaly_positions, key=lambda i: float(scores[i]), reverse=True)[:6]
    chosen = normal_positions + anomaly_positions
    save_sprite(candidate_images[chosen], args.output_dir / "anomaly-inputs.png")
    save_sprite(candidate_recon[chosen], args.output_dir / "anomaly-reconstructions.png")

    normal_scores = scores[torch.tensor([label in (5, 7, 9) for _, label in candidates])]
    anomaly_scores = scores[torch.tensor([label not in (5, 7, 9) for _, label in candidates])]
    threshold = float(torch.quantile(normal_scores, 0.95))
    manifest = {
        "seed": args.seed,
        "epochs": args.epochs,
        "denoising": {
            "sigma": 0.45,
            "classes": [test.classes[label] for label in labels],
            "noisy_mse": [round(float(v), 5) for v in (noisy - clean).square().flatten(1).mean(1)],
            "restored_mse": [round(float(v), 5) for v in (restored - clean).square().flatten(1).mean(1)],
        },
        "anomaly": {
            "normal_classes": ["Sandal", "Sneaker", "Ankle boot"],
            "threshold_95pct": round(threshold, 5),
            "normal_score_median": round(float(normal_scores.median()), 5),
            "other_score_median": round(float(anomaly_scores.median()), 5),
            "gallery": [
                {"class": test.classes[candidates[i][1]], "score": round(float(scores[i]), 5), "anomaly": candidates[i][1] not in (5, 7, 9)}
                for i in chosen
            ],
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
