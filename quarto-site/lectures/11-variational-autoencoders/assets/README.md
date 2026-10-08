# Section 1: real generation evidence

The figures and 16 recorded samples in `evidence/` are actual model outputs,
not generated illustrations. `vae-hero.svg` is explicitly conceptual artwork.

From the repository root, with Python, torch, torchvision, numpy and matplotlib:

```sh
python quarto-site/lectures/11-variational-autoencoders/assets/build_generation_evidence.py --data /tmp/fashion-mnist
```

Run within the repository: the script imports the established Chapter 10
`Autoencoder` implementation. It reuses `/tmp/latent-laboratory-checkpoints/ae-d2.pt`
if present, otherwise recreates it. The VAE checkpoint is cached in the same
temporary directory. Use `--checkpoints` with a fresh directory for a fresh run.
No checkpoint is published with the site. Quarto does not train either network.

Protocol and selections are recorded in `evidence/manifest.json`:

- Fashion-MNIST: fixed seed-41 split, 50,000 train / 10,000 validation.
- AE: two-dimensional MLP, six epochs, pixel MSE (Project 3 protocol).
- VAE: two-dimensional MLP, ten epochs, seed 113, Adam at 0.001.
- VAE objective: sum-pixel soft-target BCE plus Gaussian KL per example,
  then batch mean. Grayscale-target BCE is a common surrogate, not an exact
  continuous-image likelihood.
- AE probes: uniform draws in the coordinate-wise 1st–99th percentile box
  of the first 2,000 validation codes. The map uses actual coordinates, not PCA.
- VAE draws: standard normal; first sixteen recorded draws, no selection by
  appearance. The page displays decoder means rather than Bernoulli pixel draws.
- Neighbors: exact pixel-MSE search over all 50,000 training images for the
  first eight VAE means. This is not a proof against memorization.

These models are motivation experiments, not a controlled quality benchmark:
their objectives and epoch counts differ. All imperfect samples are retained.

Sources: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
[Kingma and Welling (2014)](https://arxiv.org/abs/1312.6114).
Reusable bibliography keys: `xiao2017fashion`, `kingma2014vae`.

The browser viewer uses only local PNGs and the manifest. Serve the rendered
site over HTTP. Keyboard focus + Enter activates the Next draw button. The
sixteen samples cycle without contacting a model service.

## Section 2: image-conditioned latent distributions

`build_posterior_evidence.py --data /path/to/dataset-parent` trains the same VAE
architecture and protocol in a **separate run**, cached at `/tmp/vae-section2.pt`.
It requires the Fashion-MNIST data to be present locally; it does not download.
No model training occurs during site rendering. To regenerate from scratch, pass
`--checkpoint` with a new path. The checkpoint itself is not published.

`evidence/posterior/manifest.json` records the checkpoint SHA-256, PyTorch version,
training losses, seeds, selected dataset indices, means, log variances, standard
deviations, and all recorded posterior and prior draws. Training uses the seed-41
50k/10k split, initialization seed 113, ten epochs, Adam at 0.001, batch size 256,
and sum-pixel soft-target BCE + Gaussian KL, then batch mean. This is the same
soft-target surrogate described above, not a continuous-image likelihood.

The interactive viewer takes the **first validation item** of classes 0, 1, 9
(T-shirt/top, Trouser, Ankle boot), chosen before inspecting outputs. It retains
all eight draws per input from seed 307. The decoder output is its mean image.
The mean-code button decodes μ; it does not average the eight output images.
The map uses actual latent coordinates in a labeled μ ± 0.30 local window,
with identical scale across inputs. Ellipses mark Mahalanobis radii 1 and 2,
not hard support boundaries or marginal 68%/95% probability contours.
The comparison figure uses global coordinates and the actual, narrow spreads.
`gaussian-exercise.svg` is explicitly a hand-chosen numerical example, not data.

Controls: choose input, cycle draws, decode the mean, resume draws. All operate
on local recorded assets and work by keyboard. A fetch failure disables controls
and reports a static preview instead of silently inventing results.

## Section 3: reconstruction and the prior penalty

Run `build_loss_evidence.py --data /path/to/dataset-parent` after generating the
section-2 checkpoint. Optional `--vae-checkpoint` and `--unregularized-checkpoint`
paths select the caches. Rendering never trains. The beta=1 condition reuses
section 2; beta=0 uses identical initialization, architecture, data split,
minibatch order, sampling protocol, optimizer and epoch count, with the KL
coefficient changed to zero. Checkpoint hashes, software version, seeds and
validation indices are in `evidence/loss/manifest.json`.

The comparison shows the first four validation inputs, one posterior draw each,
and the first four fixed-seed prior draws. Common noise is used across conditions;
model weights differ after training. Eight prior outputs per model are retained.
The displayed validation reconstruction score averages eight posterior draws
for each of the first 1,024 held-out images. Pixel costs are summed, then draws
and images averaged. This soft-target BCE on grayscale data is a surrogate,
not an exact continuous-data negative log likelihood. One seed and an illustrative
image gallery do not establish a general generative-quality ranking.

The reconstruction slide shows the first three seeded posterior draws for the
first validation image and averages their measured scalar costs. It does not
score an averaged image. The Gaussian exercise and the two-proposal loss exercise
are explicitly pedagogical numerical illustrations.

The KL controls perform exact 1D Gaussian calculations in the browser; they do
not run model inference. The ELBO controls use an exact discrete toy model:
prior `(0.5, 0.5)`, likelihood for observed x=1 `(0.8, 0.2)`, evidence `0.5`, true
posterior `(0.8, 0.2)`. For proposal `(w,1-w)`, all expectations are exact sums.
The positive total cost is an upper bound on negative log evidence; its negative
is the ELBO. The prior KL in the objective and posterior KL in the bound gap are
distinct. At w=0.8 the gap is zero; at w=0.5 it is approximately 0.223144 nats.
Primary reference: `kingma2014vae`, §2.2 and Appendix B.

## Section 4: the reconstruction gradient path

Run `python check_reparameterization.py` with PyTorch installed. This executes
three independent CPU probes (seed 7, fresh leaves in each): `Normal.sample()`,
manual `mu + sigma * eps`, and `Normal.rsample()`. Each uses a scalar decoder
`a*z`, target 2, and half squared reconstruction error, without KL. The JSON in
`evidence/reparameterization/manifest.json` records forward values, gradients,
software version, and eight standard-normal draws from seed 73 for the viewer.
No training or model inference runs in the browser.

The script verifies equal seeded forward values in this tested configuration,
missing encoder gradients for `sample()`, matching manual/`rsample()` gradients,
and double-precision central finite differences with common fixed noise.
It also verifies the worked example mu=1, sigma=0.5, epsilon=-1, including the
log-variance derivative. The interactive graph uses recorded draws; its explicit
“Worked value” button chooses -1 rather than presenting it as a seeded draw.
Moving the sliders never changes the current noise. A new training forward pass
normally draws fresh noise; the fixed-noise view represents one backward pass.

References: `kingma2014vae` §2.4 and the official PyTorch distributions docs.
The scalar probes are diagnostic examples, not an image-generation benchmark.


## Section 5: runnable training path

`vae_training.py` assembles the encoder, reparameterization, decoder, per-image
loss, and Adam update shown on the slides. With torch, torchvision, and local
Fashion-MNIST data available, run:

```sh
python vae_training.py --data /path/to/data --check
python vae_training.py --data /path/to/data --epochs 10 --output /tmp/course-vae.pt
```

The check records results in `evidence/training/checks.json`: shapes, finite
loss, per-image reductions, duplicate-output batch invariance, reconstruction
gradients to both heads, parameter updates, and stochastic forward in eval mode.
It uses initialization seed 113 and noise seed 811 on 32 training images.
The simplified constructor consumes initialization RNG differently from the
original evidence generator; equal seeds do not imply identical starting weights.
The inspector reuses section-2 recorded outputs, including their first posterior
and prior draws; it does not run inference in the browser. The composed module
can load the original checkpoint by renaming encoder and head parameter keys.

## Section 6: paths, samples, and latent use

Run `build_latent_evidence.py --data /path/to/data` with the section-2 checkpoint
(default `/tmp/vae-section2.pt`). It loads unchanged weights into the section-5
module with equivalent parameter names. It never trains.

- Paths: first validation T-shirt, trouser, and boot; 21 linear interpolations
  between posterior means for T-shirt/trouser and T-shirt/boot. All are retained.
- Map: first 1,024 validation means, actual coordinates, labels only for display.
  The viewport is [-4,4] in both coordinates and reports omitted points.
- Gallery: first 16 prior draws, seed 619, no filtering. Mean pixels, not pixel draws.
- Latent diagnostic: first 1,024 validation images, eight posterior draws each,
  noise seed 620. Cyclic reassignment (+1 image) preserves the decoded samples
  and breaks their pairing with targets. Pixel-summed BCE averaged over draws
  and images: paired 255.4250, reassigned 836.5840; mean analytic KL 6.2336.
  First four inputs and first draw shown. This is an intervention on matching,
  not a trained collapse condition or a quality benchmark.

The manifest stores all indices, path codes, map coordinates, seeds, costs,
software version, and checkpoint SHA-256. PNGs use grayscale uint8 rounding.
The browser only changes local recorded images and the corresponding SVG marker.
Slider and path selector support keyboard input. A manifest failure disables them.
References: `kingma2014vae`, `xiao2017fashion`, and `bowman2016generating` §3.1.
