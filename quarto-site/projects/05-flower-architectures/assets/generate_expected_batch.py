"""Reproduce the expected Flowers102 batch image used by the homework."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision.transforms import v2

NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]

transform = v2.Compose([
    v2.Resize(256), v2.CenterCrop(224), v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
dataset = Flowers102("/tmp/course-flowers102", split="train", download=True, transform=transform)
loader = DataLoader(dataset, batch_size=12, shuffle=True,
                    generator=torch.Generator().manual_seed(23))
images, labels = next(iter(loader))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(3, 4, figsize=(12, 8.2))
fig.patch.set_facecolor("#f6f3ec")
for ax, image, label in zip(axes.flat, images, labels):
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(NAMES[int(label)], color="#19242d", fontweight="semibold", pad=7)
    ax.axis("off")
fig.suptitle("A batch is evidence: inspect images and labels before training",
             fontsize=17, fontweight="bold", color="#19242d", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.965), pad=1.2)
fig.savefig(Path(__file__).with_name("expected-flower-batch.png"), dpi=160,
            facecolor=fig.get_facecolor(), bbox_inches="tight")
