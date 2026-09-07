"""Reproduce the reliable-training case study from Lecture 05.

The experiment trains the same six-layer ReLU classifier on Fashion-MNIST
under progressively better training recipes and saves loss curves plus a CSV.

Examples
--------
Quick classroom check (small subset, three epochs):
    python compare_training_recipes.py --quick

Full comparison:
    python compare_training_recipes.py --epochs 30

Dependencies:
    pip install torch torchvision matplotlib numpy
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class Recipe:
    name: str
    label: str
    good_init: bool
    batch_norm: bool = False
    dropout: float = 0.0
    weight_decay: float = 0.0
    schedule: bool = False
    early_stopping: bool = False


RECIPES = (
    Recipe("baseline", "Bad init; no BN/dropout", good_init=False),
    Recipe("he", "+ He initialization", good_init=True),
    Recipe("he_bn", "+ BatchNorm", good_init=True, batch_norm=True),
    Recipe(
        "he_bn_dropout",
        "+ Dropout",
        good_init=True,
        batch_norm=True,
        dropout=0.25,
    ),
    Recipe(
        "controlled",
        "Full controlled recipe",
        good_init=True,
        batch_norm=True,
        dropout=0.25,
        weight_decay=1e-4,
        schedule=True,
        early_stopping=True,
    ),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SixLayerClassifier(nn.Module):
    """Five hidden affine layers and one output layer."""

    def __init__(self, recipe: Recipe, width: int = 256) -> None:
        super().__init__()
        dimensions = [28 * 28, width, width, width, width, width]
        blocks: list[nn.Module] = []

        for in_features, out_features in zip(dimensions[:-1], dimensions[1:]):
            blocks.append(nn.Linear(in_features, out_features))
            if recipe.batch_norm:
                blocks.append(nn.BatchNorm1d(out_features))
            blocks.append(nn.ReLU())
            if recipe.dropout > 0:
                blocks.append(nn.Dropout(recipe.dropout))

        blocks.append(nn.Linear(dimensions[-1], 10))
        self.network = nn.Sequential(*blocks)
        self._initialize(good_init=recipe.good_init)

    def _initialize(self, good_init: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if good_init:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    # Deliberately ignores fan-in. At depth, activation and
                    # gradient scales become poorly controlled.
                    nn.init.normal_(module.weight, mean=0.0, std=0.15)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images.flatten(1))


def make_loaders(
    data_dir: Path,
    batch_size: int,
    train_examples: int | None,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    full_train = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )

    train_set, val_set = random_split(
        full_train,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(seed),
    )
    if train_examples is not None:
        train_set = Subset(train_set, range(min(train_examples, len(train_set))))
        val_size = min(max(train_examples // 5, 500), len(val_set))
        val_set = Subset(val_set, range(val_size))

    common = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    # This switch is essential: BatchNorm uses running statistics and dropout
    # becomes the identity map during evaluation.
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        examples += targets.size(0)
    return loss_sum / examples, correct / examples


def learning_rate_multiplier(epoch: int, epochs: int, warmup: int) -> float:
    """Linear warmup followed by cosine decay."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_recipe(
    recipe: Recipe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    # Reset before every run so differences come from the recipe, not random
    # data order or an unrelated random initialization draw.
    seed_everything(seed)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)
    model = SixLayerClassifier(recipe).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=recipe.weight_decay,
    )
    scheduler = None
    if recipe.schedule:
        warmup = max(1, min(3, epochs // 5))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: learning_rate_multiplier(epoch, epochs, warmup),
        )

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int | str]] = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        examples = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)

            if not torch.isfinite(loss):
                print(f"  epoch {epoch + 1:02d}: non-finite loss; stopping run")
                return history

            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.size(0)
            examples += targets.size(0)

        train_loss = loss_sum / examples
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "recipe": recipe.name,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )
        print(
            f"  epoch {epoch + 1:02d} | train {train_loss:9.4f} | "
            f"val {val_loss:9.4f} | accuracy {val_accuracy:6.2%}"
        )

        if scheduler is not None:
            scheduler.step()

        if recipe.early_stopping:
            if val_loss < best_loss - 1e-3:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"  early stop; restoring epoch {epoch + 1 - patience}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_results(
    histories: dict[str, list[dict[str, float | int | str]]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for history in histories.values() for row in history]
    if not rows:
        raise RuntimeError("Every run diverged before completing an epoch.")

    csv_path = output_dir / "training_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(RECIPES)))
    labels = {recipe.name: recipe.label for recipe in RECIPES}
    for color, recipe in zip(colors, RECIPES):
        history = histories[recipe.name]
        if not history:
            continue
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(
            epochs,
            [float(row["train_loss"]) for row in history],
            marker="o",
            markersize=3,
            color=color,
            label=labels[recipe.name],
        )
        axes[1].plot(
            epochs,
            [float(row["val_loss"]) for row in history],
            marker="o",
            markersize=3,
            color=color,
            label=labels[recipe.name],
        )

    for axis, title in zip(axes, ("Training loss", "Validation loss")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Same data, architecture, seed, and optimizer—different training recipes")
    figure_path = output_dir / "training_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"\nSaved {csv_path}\nSaved {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("training-results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 5,000 training examples and three epochs for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    epochs = 3 if args.quick else args.epochs
    train_examples = 5_000 if args.quick else None
    seed_everything(args.seed)
    train_loader, val_loader = make_loaders(
        args.data_dir, args.batch_size, train_examples, args.seed
    )
    print(f"Device: {device}; epochs: {epochs}; recipes: {len(RECIPES)}")

    histories = {}
    for recipe in RECIPES:
        print(f"\n[{recipe.name}] {recipe.label}")
        histories[recipe.name] = train_one_recipe(
            recipe,
            train_loader,
            val_loader,
            epochs,
            args.learning_rate,
            args.patience,
            args.seed,
            device,
        )
    save_results(histories, args.output_dir)


if __name__ == "__main__":
    main()
