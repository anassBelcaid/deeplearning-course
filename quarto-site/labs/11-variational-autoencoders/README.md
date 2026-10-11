# VAE session authoring

`build_notebook.py` is the shared task source for the student notebook and Quarto reading page. It embeds all PNG/SVG expected figures as notebook attachments and packages the notebook plus four tensor-only checkpoints into `assets/vae-session.zip`.

- `support.py`: fully provided preparation, model assembly, training, plots, and widgets; embedded verbatim as a folded notebook cell.
- `reference_model.py`: solutions to A/B/C, also revealed behind details in the reading page.
- `build_evidence.py`: explicit training and evidence generation. It uses the same support and reference implementations as the student path. Never run during Quarto rendering.
- `assets/checkpoints/manifest.json`: exact configuration, measured histories/timing, selected dataset IDs, and SHA-256 checksums.
- `assets/expected/metrics.json`: validation, latent-use probe, and short-run measurements.

Regenerate evidence with `COURSE_DATA_DIR=/path/to/data python build_evidence.py`; regenerate galleries from saved weights with `--reuse`. Then `python build_notebook.py` (nbformat required). Rebuild the notebook whenever support or solutions change.

Reference protocol: seed 117, split/loader seed 41; 12,000 training and 2,000 validation from a fixed permutation of Fashion-MNIST's official training split. No test set. Four 20-epoch runs: deterministic AE, stochastic beta=0, VAE beta=1, and beta=20. Adam .001, batch 256; matched architecture and initialization. Labels never train the model. The stochastic runs share their batch/noise sequence. The AE's unused log-variance head is retained only for common checkpoint structure.

The 60-second student training cap is checked between batches, with a final validation pass. It does not cover downloads. Provide dataset caching before class. Session ZIP removes reliance on the course host for checkpoints but does not redistribute the dataset.

All garment images are Fashion-MNIST (Xiao, Rasul, Vollgraf); outputs, plots, and the ownership diagram were created for this session. The mean-code shuffling probe is deterministic and distinct from the stochastic reconstruction estimator. The colored latent maps distinguish means from samples; the radius sqrt(-2 log(.05)) circle denotes 95% mass of the two-dimensional standard-normal prior. One seed, two dimensions, selected galleries, and limited training do not establish broad model rankings or complete posterior collapse.
