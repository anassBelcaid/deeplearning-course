# Chapter 11 notes: two reading paths

`index.qmd` includes `_chapter.qmd` only for HTML and `_review.qmd` only for PDF.
The full chapter preserves the six approved sections, all six exercises, visual
evidence and eight interactive experiments. The PDF is a five-part technical
review, with no simulations or repeated galleries.

`build_notes_assets.py` rebuilds the original text-free SVG cover, standalone
demo pages, evidence-gallery includes, and source-synchronized code snippets.
Run it after rendering the lecture, using Python with BeautifulSoup installed:

```sh
python quarto-site/notes/11-variational-autoencoders/assets/build_notes_assets.py
quarto render quarto-site/notes/11-variational-autoencoders/index.qmd --to html
quarto render quarto-site/notes/11-variational-autoencoders/index.qmd --to pdf
```

The chapter and demos reuse local lecture images, manifests, styles and control
scripts. No network training or model inference occurs during rendering or in the
browser. Standalone HTML pages are published as project resources. Embedded
frames resize to their content and link to a full-window alternative. The
scripts retain the lecture's honest labeling of recorded outputs and toy arithmetic.

## Preservation map

1. Reconstruction gallery; random pixels; AE map and all 16 samples; generative
   path; all 16 VAE preview draws; exact training-neighbor gallery; A/B/C exercise.
2. Encoder module comparison; Gaussian parameters; all 3 inputs × 8 draws and
   mean-code mode; actual prior/posterior map; Gaussian-interval exercise; code.
3. Controlled ablation and both full galleries; three-draw costs; Gaussian-KL
   derivation/control; proposal exercise; exact discrete evidence-bound example.
4. Detached and reparameterized probes; operation graph with recorded noise;
   scalar gradient exercise and log-variance extension; manual/rsample results.
5. Full encoder, model, loss and train-step code; tensor shapes; batch reduction
   exercise; measured smoke check; all three output inspection paths.
6. Both 21-position latent paths and map; all 16 prior draws; reassignment gallery
   and scores; collapse interpretation; closing generation task and solution.

Lecture assets and generation scripts are the canonical evidence. Reusable
citations remain in `quarto-site/references.bib`. Detailed seeds, checkpoint hashes,
software versions and selection rules are in the lecture manifests/README.
