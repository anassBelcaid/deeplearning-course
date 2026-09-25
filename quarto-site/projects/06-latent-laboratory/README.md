# Authoring the Latent Laboratory

`build_notebook.py` is the canonical cell source. `python build_notebook.py`
regenerates `starter/latent_laboratory.ipynb` and `reading.qmd`, preserving all
student TODOs. Expected images are embedded as notebook attachments.

To regenerate measured expectations, install torch, torchvision, numpy,
matplotlib, and ipywidgets, then run:

```sh
COURSE_DATA_DIR=/tmp/fashion-mnist python build_notebook.py --reference
```

The reference mode replaces the explicitly defined task cells with maintainer
solutions, executes the same provided training and checks, exercises widget
callbacks, saves figures/metrics under `assets/expected/`, and rebuilds both
student artifacts. This is author tooling, not a linked student solution.

Use a GPU if available; otherwise the code runs on CPU. The reference figures
use seed 41, six epochs, four widths, and a fixed 50k/10k split. Do not substitute
lecture figures: their training protocol differs.

Quarto rendering is non-executing by project metadata:

```sh
quarto render projects/06-latent-laboratory/index.qmd
quarto render projects/06-latent-laboratory/reading.qmd
```

The starter download uses a same-origin link with the HTML download attribute;
the readable version is an explicit `reading.html` destination. Colab uses the
published course repository's main branch and becomes available after publishing.
