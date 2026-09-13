"""Build the two-session CIFAR-10 PyTorch vision-system notebook."""

import json
from pathlib import Path
from textwrap import dedent


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(True),
    }


def code(source, *, provided=False):
    metadata = {"tags": ["provided-infrastructure"]} if provided else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": dedent(source).strip().splitlines(True),
    }


cells = [
md(r"""
# Project 2 — The Vision Training System

**Two sessions · CIFAR-10 · PyTorch · MLP → CNN · checkpoint and resume**

You will build one reusable training system and use it twice. Session 1 practices PyTorch with an MLP. Session 2 begins after the convolution lecture and replaces only the model with a CNN.

The notebook starts with a small subset so every experiment finishes during class. Increase `TRAIN_LIMIT` only after the complete pipeline works.
"""),
md(r"""
## Experimental contract

| Fixed across runs | Allowed to change |
|---|---|
| split seed, transforms, loaders, metrics | Session 1 optimizer exploration |
| validation protocol, epoch accounting | model family: MLP then CNN |
| checkpoint format | device in the timing experiment |

Do not claim that GPU is always faster from one measurement. Report the hardware, dataset size, model, warm-up policy, and elapsed time.
"""),
code(r"""
from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2

SEED = 17
DATA_DIR = Path(os.environ.get("COURSE_DATA_DIR", "data"))
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Classroom-sized defaults. Set to None for the complete split.
TRAIN_LIMIT = 12_000
VAL_LIMIT = 2_000
TEST_LIMIT = 2_000
BATCH_SIZE = 128

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
""", provided=True),
md(r"""
## Session 1 — PyTorch fluency with an MLP

### 1 · Define evaluation and augmentation transforms

Evaluation should be deterministic. Training may apply label-preserving stochastic augmentation. For CIFAR-10, a small padded crop and horizontal flip are reasonable starting choices.

Implement both functions. Use the modern `torchvision.transforms.v2` API.
"""),
code(r"""
def make_eval_transform():
    # Return image -> float tensor in [0, 1].
    # TODO: v2.ToImage(), then v2.ToDtype(torch.float32, scale=True)
    raise NotImplementedError


def make_train_transform():
    # Return stochastic augmentation followed by float conversion.
    # TODO: ToImage(), RandomCrop(32, padding=4), RandomHorizontalFlip(),
    #       and ToDtype(torch.float32, scale=True), in that order.
    raise NotImplementedError
"""),
code(r"""
eval_transform = make_eval_transform()
train_transform = make_train_transform()

assert isinstance(eval_transform, v2.Compose)
assert isinstance(train_transform, v2.Compose)

probe = np.zeros((32, 32, 3), dtype=np.uint8)
eval_probe = eval_transform(probe)
train_probe = train_transform(probe)
assert eval_probe.shape == (3, 32, 32)
assert train_probe.shape == (3, 32, 32)
assert eval_probe.dtype == torch.float32
assert 0.0 <= eval_probe.min() <= eval_probe.max() <= 1.0
print("✓ transforms return the expected tensor contract")
""", provided=True),
md(r"""
### Inspect stochastic augmentation

Run the next cell twice. The source image is unchanged; the training transform samples a new view each time. Decide whether every transformation preserves the class label.
"""),
code(r"""
raw_train = datasets.CIFAR10(DATA_DIR, train=True, download=True)
image, label = raw_train[0]

fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
axes[0, 0].imshow(image)
axes[0, 0].set_title(f"original · {raw_train.classes[label]}")
axes[0, 0].axis("off")

for ax in axes.flat[1:]:
    transformed = train_transform(image)
    ax.imshow(transformed.permute(1, 2, 0))
    ax.set_title("random training view")
    ax.axis("off")

fig.tight_layout()
plt.show()
"""),
md(r"""
**Interpretation.** Which pixels can change? Which object property should remain invariant? Name one augmentation that would be inappropriate for some CIFAR-10 classes.

Answer: …
"""),
md(r"""
### 2 · Build reproducible datasets and loaders

We create separate dataset objects so training and validation can use different transforms. The index split is generated once and reused.
"""),
code(r"""
def limited(indices, limit):
    return indices if limit is None else indices[:limit]


generator = torch.Generator().manual_seed(SEED)
permutation = torch.randperm(50_000, generator=generator).tolist()
val_indices = limited(permutation[:5_000], VAL_LIMIT)
train_indices = limited(permutation[5_000:], TRAIN_LIMIT)

train_base = datasets.CIFAR10(
    DATA_DIR, train=True, download=True, transform=train_transform
)
val_base = datasets.CIFAR10(
    DATA_DIR, train=True, download=True, transform=eval_transform
)
test_base = datasets.CIFAR10(
    DATA_DIR, train=False, download=True, transform=eval_transform
)

train_set = Subset(train_base, train_indices)
val_set = Subset(val_base, val_indices)
test_indices = limited(list(range(len(test_base))), TEST_LIMIT)
test_set = Subset(test_base, test_indices)

CLASS_NAMES = train_base.classes
print(len(train_set), len(val_set), len(test_set), CLASS_NAMES)
""", provided=True),
code(r"""
def make_loaders(batch_size=BATCH_SIZE, num_workers=0):
    # TODO: training loader shuffles; validation and test loaders do not.
    # Return (train_loader, val_loader, test_loader).
    raise NotImplementedError
"""),
code(r"""
train_loader, val_loader, test_loader = make_loaders()
images, labels = next(iter(train_loader))

assert images.ndim == 4 and images.shape[1:] == (3, 32, 32)
assert labels.ndim == 1 and labels.dtype == torch.long
assert images.shape[0] == labels.shape[0] <= BATCH_SIZE
assert len(train_loader.dataset) == len(train_set)
print("✓ loader batch contract:", images.shape, labels.shape)
""", provided=True),
md(r"""
### 3 · Declare an MLP and count its parameters

The MLP deliberately discards spatial structure by flattening every image. Implement `CifarMLP` with:

```text
Flatten → Linear(3072, 512) → ReLU → Linear(512, 128) → ReLU → Linear(128, 10)
```
"""),
code(r"""
class CifarMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: assign the requested layers to self.network
        raise NotImplementedError

    def forward(self, images):
        # TODO: return the logits
        raise NotImplementedError
"""),
code(r"""
model = CifarMLP()
probe = torch.randn(7, 3, 32, 32)
logits = model(probe)

assert logits.shape == (7, 10)
assert not torch.allclose(logits.softmax(dim=1), logits)
print(model)
print("✓ output contract:", logits.shape)
""", provided=True),
md(r"""
Compute the parameter count *before* asking PyTorch.

For `Linear(in, out)`, the weight has `out × in` values and the bias has `out` values.

$$
P_{\text{MLP}} = \underline{\hspace{8cm}}
$$
"""),
code(r"""
def count_trainable_parameters(model):
    # TODO: sum numel() for parameters that require gradients
    raise NotImplementedError


mlp_parameters = count_trainable_parameters(model)
expected = (3072 * 512 + 512) + (512 * 128 + 128) + (128 * 10 + 10)
assert mlp_parameters == expected
print(f"✓ CifarMLP parameters: {mlp_parameters:,}")
"""),
md(r"""
### 4 · Loss, optimizer, and scheduler

Define a classification criterion and an optimizer factory. The factory lets us compare existing optimizers without rewriting the training loop.
"""),
code(r"""
criterion = ...  # TODO: nn.CrossEntropyLoss()


def make_optimizer(name, model, learning_rate, weight_decay=0.0):
    name = name.lower()
    if name == "sgd":
        # TODO: SGD with momentum=0.9
        raise NotImplementedError
    if name == "adam":
        # TODO: Adam
        raise NotImplementedError
    if name == "adamw":
        # TODO: AdamW
        raise NotImplementedError
    raise ValueError(f"unknown optimizer: {name}")
"""),
code(r"""
assert isinstance(criterion, nn.CrossEntropyLoss)
for name in ["sgd", "adam", "adamw"]:
    candidate = make_optimizer(name, model, 1e-3, weight_decay=1e-4)
    assert isinstance(candidate, torch.optim.Optimizer)
print("✓ criterion and optimizer factory")
""", provided=True),
md(r"""
Use `StepLR` for the first session. It multiplies the learning rate by `gamma` every `step_size` epochs. The scheduler advances **after** the training epoch.
"""),
code(r"""
optimizer = make_optimizer("adamw", model, 1e-3, weight_decay=1e-4)

# TODO: decay the learning rate by 0.5 every 3 epochs.
scheduler = ...

assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
assert scheduler.step_size == 3
assert scheduler.gamma == 0.5
print("✓ scheduler configured")
"""),
md(r"""
### 5 · Provided reusable training infrastructure

Read this code by responsibility. It is supplied so the experiments measure your model and configuration rather than repeated loop-writing mistakes.
"""),
code(r"""
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs, start_epoch=1, history=None):
    history = [] if history is None else list(history)
    model.to(device)

    for epoch in range(start_epoch, start_epoch + epochs):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )
        elapsed = time.perf_counter() - started
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": lr,
            "seconds": elapsed,
        })
        scheduler.step()

        print(
            f"epoch {epoch:02d} · "
            f"train {train_metrics['loss']:.3f}/{train_metrics['accuracy']:.1%} · "
            f"val {val_metrics['loss']:.3f}/{val_metrics['accuracy']:.1%} · "
            f"lr {lr:.2e} · {elapsed:.1f}s"
        )

    return history


def plot_history(history, title):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(frame.epoch, frame.train_loss, marker="o", label="train")
    axes[0].plot(frame.epoch, frame.val_loss, marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="epoch"); axes[0].legend()
    axes[1].plot(frame.epoch, 100 * frame.train_accuracy, marker="o", label="train")
    axes[1].plot(frame.epoch, 100 * frame.val_accuracy, marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="percent"); axes[1].legend()
    axes[2].plot(frame.epoch, frame.learning_rate, marker="o")
    axes[2].set(title="Learning rate", xlabel="epoch")
    fig.suptitle(title, weight="bold")
    fig.tight_layout()
    plt.show()
""", provided=True),
md(r"""
### 6 · Train the MLP

Choose one optimizer. Record the choice and why its learning rate is reasonable. Three to five classroom epochs are enough to verify the system; longer runs belong after class.
"""),
code(r"""
seed_everything()
mlp = CifarMLP()
mlp_optimizer = make_optimizer(
    "adamw", mlp, learning_rate=1e-3, weight_decay=1e-4
)
mlp_scheduler = torch.optim.lr_scheduler.StepLR(
    mlp_optimizer, step_size=3, gamma=0.5
)

MLP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_history = fit(
    mlp, train_loader, val_loader, criterion,
    mlp_optimizer, mlp_scheduler, MLP_DEVICE, epochs=5,
)
plot_history(mlp_history, "CIFAR-10 MLP")
"""),
md(r"""
**Interpretation.** Is the dominant problem underfitting, overfitting, or inconclusive evidence from a short run? Cite both training and validation curves.

Answer: …
"""),
md(r"""
### 7 · CPU versus GPU: a controlled timing experiment

Timing must use the same initial weights, batches, optimizer, and number of epochs. GPU work is asynchronous, so synchronize before stopping the clock.
"""),
code(r"""
def timed_training(device, initial_state, epochs=2):
    # Recreate randomness and the loader so both devices see the same
    # shuffled indices and stochastic augmentations (num_workers=0).
    seed_everything(SEED + 99)
    timing_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + 99),
        num_workers=0,
    )
    candidate = CifarMLP().to(device)
    candidate.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    history = fit(
        candidate, timing_loader, val_loader, criterion,
        optimizer, scheduler, device, epochs=epochs,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    return time.perf_counter() - started, history[-1]["val_accuracy"]


seed_everything()
initial_state = copy.deepcopy(CifarMLP().state_dict())
timings = []

cpu_seconds, cpu_accuracy = timed_training(
    torch.device("cpu"), initial_state
)
timings.append({"device": "CPU", "seconds": cpu_seconds,
                "val_accuracy": cpu_accuracy})

if torch.cuda.is_available():
    gpu_seconds, gpu_accuracy = timed_training(
        torch.device("cuda"), initial_state
    )
    timings.append({"device": torch.cuda.get_device_name(0),
                    "seconds": gpu_seconds, "val_accuracy": gpu_accuracy})

pd.DataFrame(timings)
""", provided=True),
md(r"""
Report:

1. CPU model and GPU model name;
2. number of training examples, batch size, and epochs;
3. elapsed times and speedup `CPU seconds / GPU seconds`;
4. why this result does **not** prove that every model is faster on GPU.

Answer: …
"""),
md(r"""
### 8 · Save a resumable checkpoint

Weights are enough for inference. Continuing training also requires optimizer, scheduler, epoch, and history state.
"""),
code(r"""
def save_checkpoint(path, model, optimizer, scheduler, epoch, history):
    # TODO: torch.save a dictionary containing all five states.
    raise NotImplementedError


def load_checkpoint(path, model, optimizer, scheduler, device):
    # TODO: load with map_location=device, restore all state_dict objects,
    # and return (next_epoch, history).
    raise NotImplementedError
"""),
code(r"""
checkpoint_path = CHECKPOINT_DIR / "mlp_resume.pt"
save_checkpoint(
    checkpoint_path, mlp, mlp_optimizer, mlp_scheduler,
    epoch=mlp_history[-1]["epoch"], history=mlp_history,
)
assert checkpoint_path.exists()

restored_mlp = CifarMLP().to(MLP_DEVICE)
restored_optimizer = torch.optim.AdamW(restored_mlp.parameters(), lr=1e-3)
restored_scheduler = torch.optim.lr_scheduler.StepLR(
    restored_optimizer, step_size=3, gamma=0.5
)
next_epoch, restored_history = load_checkpoint(
    checkpoint_path, restored_mlp, restored_optimizer,
    restored_scheduler, MLP_DEVICE,
)

for original, restored in zip(mlp.parameters(), restored_mlp.parameters()):
    assert torch.equal(original, restored)
assert next_epoch == mlp_history[-1]["epoch"] + 1
print("✓ model and training state restored")
""", provided=True),
md(r"""
## Session 2 — Convolutional continuation

> **LOCKED UNTIL THE CONVOLUTION LECTURE.** In Session 1, this section is provided future infrastructure, not assessed work. After convolution is introduced, implement and explain it.

### 9 · Declare a compact CNN

Implement this shape path:

```text
(B,3,32,32)
→ Conv(3,32,3,pad=1) → ReLU → MaxPool(2)     = (B,32,16,16)
→ Conv(32,64,3,pad=1) → ReLU → MaxPool(2)    = (B,64,8,8)
→ Flatten → Linear(4096,128) → ReLU → Linear(128,10)
```
"""),
code(r"""
class CifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO after the convolution lecture:
        # define self.features and self.classifier.
        raise NotImplementedError

    def forward(self, images):
        # TODO: features -> flatten -> classifier
        raise NotImplementedError
"""),
code(r"""
cnn = CifarCNN()
probe = torch.randn(7, 3, 32, 32)
assert cnn(probe).shape == (7, 10)

cnn_parameters = count_trainable_parameters(cnn)
print(cnn)
print(f"CNN parameters: {cnn_parameters:,}")
print(f"MLP parameters: {mlp_parameters:,}")
""", provided=True),
md(r"""
Compute the CNN count analytically. For `Conv2d(in_channels, out_channels, k)`, the weight contains

$$
\text{out channels}\times\text{in channels}\times k_H\times k_W
$$

values, plus one bias per output channel.

Calculation: …
"""),
md(r"""
### 10 · Train the CNN with a fixed protocol

To isolate the architectural comparison, everyone uses AdamW with `lr=3e-4`, `weight_decay=1e-4`, and cosine decay. Train directly on GPU when available.
"""),
code(r"""
CNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_CNN_EPOCHS = 8
TOTAL_CNN_EPOCHS = 10

seed_everything()
cnn = CifarCNN().to(CNN_DEVICE)
cnn_optimizer = torch.optim.AdamW(
    cnn.parameters(), lr=3e-4, weight_decay=1e-4
)
cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    cnn_optimizer, T_max=TOTAL_CNN_EPOCHS
)

cnn_history = fit(
    cnn, train_loader, val_loader, criterion,
    cnn_optimizer, cnn_scheduler, CNN_DEVICE, epochs=INITIAL_CNN_EPOCHS,
)
plot_history(cnn_history, "CIFAR-10 CNN")
"""),
md(r"""
### 11 · Save, reload, and continue

Save the CNN after the first run, construct fresh objects, restore all states, and continue for two epochs. The learning rate and epoch number must continue rather than restart.
"""),
code(r"""
cnn_checkpoint = CHECKPOINT_DIR / "cnn_resume.pt"
save_checkpoint(
    cnn_checkpoint, cnn, cnn_optimizer, cnn_scheduler,
    epoch=cnn_history[-1]["epoch"], history=cnn_history,
)

continued_cnn = CifarCNN().to(CNN_DEVICE)
continued_optimizer = torch.optim.AdamW(
    continued_cnn.parameters(), lr=3e-4, weight_decay=1e-4
)
continued_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    continued_optimizer, T_max=TOTAL_CNN_EPOCHS
)

next_epoch, continued_history = load_checkpoint(
    cnn_checkpoint, continued_cnn, continued_optimizer,
    continued_scheduler, CNN_DEVICE,
)

continued_history = fit(
    continued_cnn, train_loader, val_loader, criterion,
    continued_optimizer, continued_scheduler, CNN_DEVICE,
    epochs=2, start_epoch=next_epoch, history=continued_history,
)

assert continued_history[-1]["epoch"] == cnn_history[-1]["epoch"] + 2
print("✓ training continued from the saved epoch")
""", provided=True),
md(r"""
### 12 · Fair MLP–CNN comparison

Create a table with:

- trainable parameters;
- best validation accuracy;
- final validation loss;
- median seconds per epoch;
- device used.
"""),
code(r"""
def summarize(name, model, history, device):
    frame = pd.DataFrame(history)
    return {
        "model": name,
        "parameters": count_trainable_parameters(model),
        "best_val_accuracy": frame.val_accuracy.max(),
        "final_val_loss": frame.val_loss.iloc[-1],
        "median_seconds_per_epoch": frame.seconds.median(),
        "device": str(device),
    }


comparison = pd.DataFrame([
    summarize("MLP", mlp, mlp_history, MLP_DEVICE),
    summarize("CNN", cnn, cnn_history, CNN_DEVICE),
])
comparison
""", provided=True),
md(r"""
### Final technical conclusion

Answer with evidence from your executed notebook.

1. What did augmentation change in the *data distribution seen during training*?
2. Which optimizer did you use for the MLP, and what did the scheduler change over epochs?
3. On your hardware, when did GPU execution help? What limits the claim?
4. Which model used more parameters? Show the calculations.
5. Did the CNN improve validation performance? Why is spatial structure a plausible explanation?
6. Which exact states had to be restored to continue training faithfully?
7. Name one uncontrolled factor that prevents a stronger MLP–CNN causal claim.

**Submission:** restart, run all unlocked cells, keep outputs visible, and export this completed notebook.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "execute": {"enabled": False},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"cell-{index:03d}"

target = Path(__file__).parent / "starter" / "vision_training_system.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {target} with {len(cells)} cells")
