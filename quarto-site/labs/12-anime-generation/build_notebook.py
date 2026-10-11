"""Single source for an open-ended student notebook and its Quarto reading companion.
Embeds infrastructure and figures so the notebook has no course-server dependency.
No model solution or hidden trained generator is included.
"""
from pathlib import Path
from textwrap import dedent
import base64,json,zipfile
import nbformat
R=Path(__file__).resolve().parent
cells=[]
def md(text):cells.append(('markdown',dedent(text).strip(),False))
def code(text,provided=False):cells.append(('code',dedent(text).strip(),provided))
def picture(file,caption):md(f'![{caption}](assets/{file})')
measure=json.loads((R/'assets/preview/measurements.json').read_text())['pairwise_pixel_mse']
md(r'''
# Anime faces: design, train, defend

**Final coding homework · independent take-home investigation · Chapters 10–12**

Create an unconditional model that generates **new RGB anime-face images from random codes**.
Choose a **VAE or GAN**, justify the choice, and train your own weights. You may adapt the course's
ideas and cite code you reuse, but you must understand and explain every model and training step.
A deterministic autoencoder with arbitrary normal sampling does not by itself meet the requirement.

This assignment deliberately leaves the architecture, objective implementation, training procedure,
and interpretation to you. There is no fill-in-the-blanks model or supplied trained solution.
Data auditing, split construction, plotting, diagnostic calculations, logging, and report rendering
are provided so your effort goes into the generative experiment.

**The scientific question:** what specific design decision will you test, and what evidence could show
that your initial expectation was wrong?

## Scope and deliverables

Build one working baseline and **one controlled variant**. Change one substantive factor, state a
prediction first, and compare under a declared matched budget. A small, carefully interpreted run is
better than an unexplained long run. Train from scratch; a pretrained face generator is not a submission.

Planning estimate: **4–6 hours of active work**, plus hardware-dependent training. This is not a
90-minute lab or a promise about convergence. Start with 32 × 32 or 64 × 64 pixels and a reproducible
subset. Reserve a modest budget before launching runs; there is no required number of epochs or GPU.
If compute is limited, use 32 × 32, fewer channels and fewer images for both runs, and report the limits.
You are assessed on design, correctness, evidence and reasoning—not a leaderboard score.

Submit:

1. The executed notebook with code, decisions, visible outputs, and cited assistance.
2. A self-contained **report.html** (roughly 1,200–1,800 words plus figures).
3. The data manifest, both run configurations and loss logs, evaluation JSON, and final model weights.
4. A short reproduction instruction: environment, dataset handle/path, selected run and sampling command.

Do not bundle the dataset. Due date and submission channel follow the instructor's announcement.
''')
picture('preview/dataset.png','Actual files from Kaggle Anime Face Dataset (splcher/animefacedataset). First 12 file-list entries, displayed in lexical order; cropped and resized to 64 × 64. This is a small data preview, not generated output or a representative sample.')
md(r'''
## 1 · Write the experiment plan before training

Record your answers in Markdown:

- **Question and hypothesis:** name one change and predict its effect and possible downside.
- **Model:** VAE or GAN? Why does its objective suit generation from fresh codes?
- **Architecture:** resolution, latent dimension, channel widths, activations, normalization and output range.
- **Controlled comparison:** what changes between A and B? What stays fixed: split, resolution, batch size,
  budget definition, initialization seed, data-order seed, sample codes and evaluation protocol?
- **Budget:** maximum steps or wall time per run, machine, checkpoint interval, and stopping rule.
- **Selection:** use validation evidence to select the model/checkpoint; reserve test images for final reporting.

Possible factors include latent dimension, channel capacity, G/D learning-rate ratio, or the VAE KL weight.
Choose one you can explain. Architecture and learning-rate changes together are not a single-factor study.
Do not introduce an unfamiliar loss merely because a tutorial uses it. Advanced methods are optional,
not assessed prerequisites; VAE and non-saturating GAN objectives are sufficient.

**Decision record:** write your hypothesis now, then append revisions with reasons. Do not rewrite your
initial prediction after seeing the outcomes. A result contradicting it can be a strong report.
''')
code('''
PLAN = {
    'student': '[WRITE your name]',
    'family': '[CHOOSE VAE or GAN]',
    'question': '[WRITE one testable question]',
    'hypothesis': '[WRITE predicted effect and risk]',
    'changed_factor': '[WRITE one factor]',
    'budget_rule': '[WRITE equal steps or equal wall time and why]',
    'selection_rule': '[WRITE validation-based checkpoint rule]',
}
''')
picture('ownership.svg','Ownership and information flow. Provided objects hold paths and decode tensors; students own the model parameters, objectives and updates. Only the training split is used for optimization.')
md(r'''
## 2 · Prepare and audit the data (infrastructure provided)

Use [Anime Face Dataset on Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset), handle
`splcher/animefacedataset`. The uploader attributes the source to
[Mckinsey666/Anime-Face-Dataset](https://github.com/Mckinsey666/Anime-Face-Dataset), collected from Getchu.
Record the version/download date, source attribution, and listed terms in your report. The metadata
currently lists DbCL-1.0; check the page when downloading. **Measure your actual file counts** instead
of copying a count from the description. A dataset of these selected faces is not representative of all
anime art, characters or styles; inspect and discuss its selection biases.

Choose an access route:

- **Kaggle:** attach `splcher/animefacedataset` through Add Input, then point `DATA_ROOT` at its images.
- **Colab/local:** download and extract the dataset yourself, or use the optional `kagglehub` cell below.
- **Offline:** use a previously extracted copy and set its path. No model download is required.

Install missing packages once with `%pip install torch torchvision pillow numpy matplotlib kagglehub`.
Only `kagglehub` needs network access for the optional download; the assignment itself uses PyTorch,
Pillow, NumPy and Matplotlib. Keep API tokens in the platform's secret mechanism, not the notebook.
Authentication, when needed, is handled by Kaggle's normal login flow.
''')
code('''
# OPTIONAL DOWNLOAD: leave False if the dataset is already attached/extracted.
DOWNLOAD_DATA = False
if DOWNLOAD_DATA:
    import kagglehub
    downloaded_root = kagglehub.dataset_download('splcher/animefacedataset')
    print('Dataset folder:', downloaded_root)
''',True)
md(r'''
The next cell defines the provided utilities. `FaceDataset` owns a root path, a list of manifest rows,
and a resolution. A row refers to an image file; indexing decodes a fresh tensor. A DataLoader batches
these tensors and shuffles **only training**. Model parameters are not stored in the dataset.

Default preprocessing: apply EXIF orientation, convert to RGB, center-crop a square and bicubic-resize.
The returned tensor is **(B, 3, H, W), float, [0,1]**. Adapt the range explicitly if your generator uses
Tanh. Exact duplicates after preprocessing are removed **before** the seeded 80/10/10 split; near
copies and related characters can remain. Augmentation is your choice, and must affect training only.
''')
code((R/'support.py').read_text(),True)
code('''
DATA_ROOT = Path('/kaggle/input/animefacedataset')  # EDIT to the actual image folder
OUTPUT = Path('anime_results')
OUTPUT.mkdir(exist_ok=True)
ENVIRONMENT = seed_everything(120)
print(ENVIRONMENT)

# Suggested starting scale, not mandatory design choices.
IMAGE_SIZE = 64
MAX_IMAGES = 8000  # deterministic subset BEFORE duplicate removal and splitting
manifest = prepare_data(DATA_ROOT, OUTPUT, size=IMAGE_SIZE, max_images=MAX_IMAGES, seed=120)
datasets, loaders = make_loaders(DATA_ROOT, manifest, batch_size=64, seed=121)
print({name: len(ds) for name, ds in datasets.items()})
print('Rejected:', len(manifest['rejected']), 'Exact duplicates removed:', len(manifest['duplicates']))
real_batch = next(iter(loaders['train']))
assert real_batch.ndim == 4 and real_batch.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
assert torch.isfinite(real_batch).all() and 0 <= real_batch.min() <= real_batch.max() <= 1
save_grid(real_batch[:32], OUTPUT/'dataset_preview.png', 'Training subset preview')
from IPython.display import display
from IPython.display import Image as DisplayImage
display(DisplayImage(filename=str(OUTPUT/'dataset_preview.png')))
''',True)
md(r'''
**Dataset report:** give discovered/scanned/retained counts, corrupt/duplicate exclusions, split counts,
original size range, preprocessing and rationale. Inspect the preview: what information does center
cropping discard? Do backgrounds, repeated characters or style concentration create easy shortcuts?
State the limitation of the duplicate check. Keep this manifest fixed across A/B. The helper caps data
before splitting, so MAX_IMAGES is a candidate-subset cap, not the final training count.

**Expected checks:** nonempty disjoint manifests; RGB batches at the selected size; finite [0,1]
values; no shared retained pixel hash between splits. These checks do not establish dataset diversity.
''')
code('''
hashes = {k: {r['pixel_sha256'] for r in rows} for k, rows in manifest['splits'].items()}
assert not hashes['train'] & hashes['validation']
assert not hashes['train'] & hashes['test']
assert not hashes['validation'] & hashes['test']
# Your audit summary and any justified changes to preparation go here.
''')
md(r'''
## 3 · Design and implement your model

Draw your architecture **before** training: tensor shapes, operations, channels, activations and parameter
counts. Explain how resolution grows in the generator/decoder and shrinks in the discriminator/encoder.
Keep the notation separate from measured activations. Include the diagram in the report.

You decide the module organization. Implement your model(s), objective and optimizers below. Provide
an adapter `sample_from_codes(models, z)` that accepts `(N, latent_dim)` standard-normal codes and
returns `(N, 3, IMAGE_SIZE, IMAGE_SIZE)` images in **[0,1]**. This adapter should use evaluation mode and
`no_grad`; reshape codes for a convolutional G as needed. It must generate without consuming a real image.

For a **GAN**, explain source targets, raw logits, non-saturating G loss, gradient routing, separate
optimizer ownership, and normalization-buffer behavior. For a **VAE**, explain the posterior parameters,
reparameterization, reconstruction likelihood/surrogate, KL and loss reductions; sample from the prior
through the decoder. A deterministic mean-code reconstruction is not a fresh generated sample.

You may consult the course code and primary papers. Cite adapted code and identify what you changed.
Do not copy an entire online training notebook and present it as your own design.
''')
code('''
# STUDENT WORK: modules, objective functions and optimizer construction.
# Add as many cells as you need; no architecture or loss is supplied here.

def build_models(config):
    raise NotImplementedError('Design and return a dictionary of your PyTorch models.')

def sample_from_codes(models, z):
    raise NotImplementedError('Generate RGB [0,1] images from codes, with eval mode and no_grad.')
''')
md(r'''
**Before a long run:** verify generation at batch sizes 1 and 4; output range and finiteness; nonzero
parameter counts; intended gradients and parameter changes after one update. For a GAN, verify each
turn changes only its owned parameters. For a VAE, test the reconstruction gradient path separately
from the KL term. Verify a saved/reloaded model reproduces evaluation-mode output for fixed codes.
These checks are your responsibility; document the evidence, not only “the code ran.”
''')
code('''
# EDIT config values and include every choice needed to reconstruct the model.
BASE_CONFIG = {
    'family': PLAN['family'], 'image_size': IMAGE_SIZE, 'latent_dim': 64,
    'seed': 122, 'data_order_seed': 121, 'batch_size': 64,
    'max_steps': 1000, 'max_seconds': 1800,  # starting caps; justify your actual budget
    # Add widths, activations, optimizer settings, losses/reductions and mode policy.
}
models = build_models(BASE_CONFIG)
print(parameter_counts(models))
for n in [1, 4]:
    # Your adapter is responsible for moving z to the model's device.
    z = torch.randn(n, BASE_CONFIG['latent_dim'])
    samples = checked_images(sample_from_codes(models, z))
    assert samples.shape == (n, 3, IMAGE_SIZE, IMAGE_SIZE)
# Add your one-update ownership/gradient checks and checkpoint round-trip check.
''')
md(r'''
## 4 · Train a baseline and one controlled variant

Implement `train_model(config, train_loader, validation_loader, folder)` yourself. Return the trained
model dictionary. You own the training loop, objectives, optimizer steps, mode transitions and stopping.
The helper `RunLog` records scalars; it does not train. Use the same step or time convention in both runs.

Your loop should: respect both caps after each batch; record actual elapsed time and updates; save
checkpoints and fixed-code sample grids at declared intervals; keep validation free of optimizer updates;
and stop to investigate nonfinite losses. Restore training mode after making evaluation snapshots.
A wall-time cap is checked between batches and cannot interrupt a slow operation already in progress.

**What to log:** GAN D/G losses separately with their reductions, or VAE reconstruction/KL/total losses;
learning rate(s), examples seen, update count, elapsed time, and validation observations used for selection.
Do not call adversarial source accuracy an image-quality metric. Do not compare a VAE objective numerically
with a GAN objective as though lower meant better images.

Example use of provided logging (integrate into your own loop):

```python
log = RunLog(folder, config)
# after your update:
log.record(step, loss_D=loss_D.item(), loss_G=loss_G.item())  # GAN example
# or log reconstruction, KL and total for a VAE
log.plot()
save_checkpoint(Path(folder)/'checkpoint.pt', models, optimizers, config, step)
```

`save_checkpoint` stores model/optimizer states and configuration. Exact mid-epoch resumption also
needs your data-order and all relevant RNG states; do not claim exact resume from this helper alone.
''')
code('''
# STUDENT WORK: implement a bounded training loop. No training recipe is hidden in support.py.
def train_model(config, train_loader, validation_loader, folder):
    raise NotImplementedError('Implement training, validation selection, checkpoints and logging.')

VARIANT_CONFIG = dict(BASE_CONFIG)
# Change ONE justified factor here and add it to BASE_CONFIG too.
# Example structure only: VARIANT_CONFIG['chosen_factor'] = your_alternative
assert VARIANT_CONFIG != BASE_CONFIG, 'Define your controlled variant before launching it.'

runs = {}
for label, config in [('A', BASE_CONFIG), ('B', VARIANT_CONFIG)]:
    seed_everything(config['seed'])
    _, run_loaders = make_loaders(DATA_ROOT, manifest, config['batch_size'], config['data_order_seed'])
    runs[label] = train_model(config, run_loaders['train'], run_loaders['validation'], OUTPUT/label)
# Record actual compute, stopped step, checkpoint selection and failures for each run.
''')
md(r'''
If the hardware cannot reach convincing faces within your declared budget, retain the results and
analyze the failure. A smaller matched comparison with correct mechanics and honest evidence is acceptable.
Do not extend only the better-looking run without reporting that the budget has changed. One run per
configuration is exploratory evidence, not a statistically established improvement; a second seed is
welcome if it fits the same budget.

## 5 · Evaluate without selecting only attractive samples

For A and B, show the same first 64 evaluation draws and fixed-code histories. If latent dimension is
the changed factor, use seeded draws in each space and explain why columns are not the same codes.
For final reporting, use a new draw seed and the untouched test reference **after** selecting your run
using validation. Keep all generated samples, including failures. Mark any curated gallery separately.

Required evidence:

1. Data preview and an architecture diagram with shapes/parameter counts.
2. Loss curves with step axes, reductions and compute budgets.
3. At least two fixed-code checkpoints and uncurated 64-sample grids for A/B.
4. A metrics table with sample counts, resolution, preprocessing, reference selection and limitations.
5. First eight query/nearest-training-image pairs; inspect failures and suspicious resemblance.

The supplied numerical diagnostics use concepts already covered in the course: squared pixel distance,
means, and nearest neighbors. **They are not fidelity metrics or a GAN leaderboard.**

For $N$ images with $P=3HW$ pixel values, mean pairwise MSE is

$$D_{\mathrm{pixel}}=\frac{2}{N(N-1)}\sum_{i<j}\frac{\|x_i-x_j\|_2^2}{P}.$$

It detects literal repetition but ignores semantic quality; random noise can score highly. Nearest-image
MSE can reveal suspicious copying but is sensitive to alignment and cannot prove memorization or novelty.
Use equal-sized train and held-out reference pools to avoid a larger pool automatically having more
opportunities for close matches. The search is exhaustive **within the declared pools**, not necessarily
the whole dataset. Report their sizes. Increasing sample count changes the evidence, not just precision.

FID/KID or other unfamiliar metrics are **optional external analyses**, not assessed requirements.
If used, cite the implementation, feature extractor/domain, preprocessing, counts and limitations; do
not compare scores obtained under different protocols. A generic photo feature space may poorly reflect
anime-specific visual quality. No extra credit comes from an unexplained metric name.
''')
picture('preview/repeated.png','Constructed control from a real dataset image: 12 exact repeats. This is not a trained model output or observed GAN collapse.')
picture('preview/noise.png','Constructed uniform-noise control: substantial pixel variation without faces. A larger diversity diagnostic is not automatically better.')
md(f'''
**Measured diagnostic sanity check** on the provided 12-image preview at 64 × 64 RGB:

| Input | Mean pairwise pixel MSE |
|---|---:|
| Actual file-list preview | {measure['real_preview']:.6f} |
| Exact repeated-image control | {measure['repeated_control']:.6f} |
| Seeded uniform-noise control | {measure['noise_control']:.6f} |

These values validate the calculation and expose its limitations. They are **not targets for student
models** and do not evaluate a trained anime generator. The preview is too small and non-random to be a
dataset benchmark. Your reference pools must come from your saved split.
''')
code('''
# Provided evaluation infrastructure. Run only after completing your model/training cells.
# These are DEVELOPMENT comparisons; test images remain unused here.
reference_n = min(512, len(datasets['train']), len(datasets['validation']))
train_ref = collect(datasets['train'], reference_n)
validation_ref = collect(datasets['validation'], reference_n)
configs = {'A': BASE_CONFIG, 'B': VARIANT_CONFIG}
metrics = {}
for label, model_dict in runs.items():
    z = torch.randn(256, configs[label]['latent_dim'], generator=torch.Generator().manual_seed(125))
    generated = checked_images(sample_from_codes(model_dict, z))
    metrics[label], near_train, near_val = evaluate_samples(generated, train_ref, validation_ref)
    save_grid(generated[:64], OUTPUT/label/'fresh.png', f'{label}: first 64 of 256 development draws')
    save_neighbors(generated[:8], train_ref,
                   {'mse':near_train['mse'][:8], 'index':near_train['index'][:8]}, OUTPUT/label/'nearest.png')
    (OUTPUT/label/'evaluation.json').write_text(json.dumps(metrics[label], indent=2))
print(json.dumps(metrics, indent=2))
# Interpret the numbers together with the images. Do not select by maximizing pixel diversity.
''',True)
code('''
# STUDENT DECISION: use your predeclared validation evidence and explain your choice.
SELECTED_RUN = '[CHOOSE A or B]'
SELECTION_REASON = '[WRITE the evidence and trade-off]'
assert SELECTED_RUN in runs and '[WRITE' not in SELECTION_REASON
n = min(512, len(datasets['train']), len(datasets['test']))
final_z = torch.randn(256, configs[SELECTED_RUN]['latent_dim'], generator=torch.Generator().manual_seed(126))
final_images = checked_images(sample_from_codes(runs[SELECTED_RUN], final_z))
final_metrics, final_nn, _ = evaluate_samples(final_images, collect(datasets['train'], n), collect(datasets['test'], n))
save_grid(final_images[:64], OUTPUT/'final_samples.png', 'Selected model: first 64 fresh final draws')
(OUTPUT/'final_evaluation.json').write_text(json.dumps({'selected_run':SELECTED_RUN,
    'reason':SELECTION_REASON,'seed':126,'metrics':final_metrics}, indent=2))
# Do not return to model tuning after viewing test results without disclosing that reuse.
''')
md(r'''
## 6 · Write and render the report

The report must be readable without running the notebook. Use this structure:

| Section | Evidence and explanation expected |
|---|---|
| Question | Hypothesis, changed factor, reason for model family, and what would falsify the prediction |
| Dataset | Kaggle source/version/date, attribution, measured counts, split, preprocessing, exclusions and biases |
| Architecture | Your diagram, tensor shapes, parameter counts, output range, objective formulas and reductions |
| Training | Hardware/software, seeds, optimizer(s), learning rates, batch size, updates/epochs/time, modes, checkpoint selection and failures |
| Results and metrics | A/B evidence under the same protocol, sample counts, curves, uncurated images, nearest neighbors, and interpretation |
| Limitations and conclusion | What changed, alternative explanations, missing evidence, compute limits and one justified next experiment |
| References | Dataset, papers, reused code, and disclosed external/AI assistance |

Captions should identify the run/checkpoint, sample-selection rule, and what a reader should inspect.
Distinguish observations from explanations: “the mouths repeat in 20 of 64 draws” is different from
“the discriminator is too strong.” What extra experiment would support the latter claim?

Fill the report sections below in your own words. The renderer only formats your writing and embeds
figures; it does not write the analysis. Add your architecture PNG and checkpoint-history figure to the
figure list. `report.html` is self-contained and can be opened offline. Use your browser's Print → Save
as PDF if a PDF is requested; use a figure appendix if needed rather than shrinking the evidence.
''')
code('''
REPORT = {
    'Question': '[WRITE your question, hypothesis, family choice and controlled change]',
    'Dataset': '[WRITE measured audit, source/date, split and preprocessing rationale]',
    'Architecture': '[WRITE model structure, dimensions, objective and parameter counts]',
    'Training': '[WRITE actual configuration, hardware, timing, modes and checkpoint rule]',
    'Results and metrics': '[WRITE quantitative and visual comparison, including failures]',
    'Limitations and conclusion': '[WRITE supported conclusions, caveats and next experiment]',
    'References': '[WRITE dataset/papers/code URLs and assistance attribution]',
}
FIGURES = [
    (OUTPUT/'dataset_preview.png', 'Replace with your informative dataset caption.'),
    (OUTPUT/'architecture.png', 'Your architecture, tensor dimensions and parameter counts.'),
    (OUTPUT/'A'/'curves.png', 'Baseline loss curves: state units and reduction.'),
    (OUTPUT/'B'/'curves.png', 'Variant loss curves under the declared matched budget.'),
    (OUTPUT/'fixed_history.png', 'Your fixed-code checkpoint comparison and selection rule.'),
    (OUTPUT/'A'/'fresh.png', 'Baseline: first 64 development draws, seed 125.'),
    (OUTPUT/'B'/'fresh.png', 'Variant: first 64 development draws, seed 125.'),
    (OUTPUT/SELECTED_RUN/'nearest.png', 'First eight development queries and nearest training references; state pool size.'),
    (OUTPUT/'final_samples.png', 'Selected model: first 64 final draws, seed 126; no quality filtering.'),
]
report_path = render_report(OUTPUT/'report.html', REPORT,
    {'plan':PLAN,'environment':ENVIRONMENT,'A':BASE_CONFIG,'B':VARIANT_CONFIG},
    {**metrics,'selected_test':final_metrics}, FIGURES)
print('Open:', report_path.resolve())
# In Colab: from google.colab import files; files.download(str(report_path))
''')
md(r'''
**Other export routes.** The ZIP also contains `report_template.qmd`. Fill it with the same evidence,
then use `quarto render report_template.qmd --to html` (or `--to pdf` with a LaTeX installation).
To export your executed notebook as a technical appendix, run
`jupyter nbconvert --to html --embed-images anime_faces_homework.ipynb`.
The notebook appendix does not replace the focused experimental report.

**Submission check:** restart and run your finished notebook in order, or clearly mark expensive training
cells and provide checkpoint-loading cells for reproducing evaluation. Include exact preprocessing,
architecture/config, model weights and a fixed-code sampling check. Open report.html in a fresh browser
and verify every figure is embedded. Remove credentials and dataset files from the submission bundle.
Do not claim that a blank starter notebook or a supplied diagnostic control is your model's result.

## Assessment (100 points)

| Criterion | Points |
|---|---:|
| Dataset audit, split discipline and preprocessing justification | 15 |
| Model/objective correctness and architecture explanation | 25 |
| Training implementation, bounded experiment, logs and reproducibility | 20 |
| Controlled comparison, complementary evidence and metric interpretation | 25 |
| Clear report, limitations, citations and ownership of decisions | 15 |

There is no required fidelity score. More GPU time, more epochs, a fancier model name, or a successful-looking
single image does not substitute for the assessed evidence. Report failures honestly and diagnose them.

## Reading path

- [Course VAE notes](https://anassbelcaid.github.io/deeplearning-course/notes/11-variational-autoencoders/): posterior, reparameterization, likelihood and KL.
- [Course GAN notes](https://anassbelcaid.github.io/deeplearning-course/notes/12-generative-adversarial-networks/): logits, update ownership, convolutional shapes and evaluation.
- [Kingma & Welling](https://arxiv.org/abs/1312.6114): VAE objective and inference.
- [Goodfellow et al.](https://arxiv.org/abs/1406.2661): adversarial objectives and alternating optimization.
- [Radford et al.](https://arxiv.org/abs/1511.06434): convolutional GAN design choices.
- [Theis et al.](https://arxiv.org/abs/1511.01844): evaluation depends on the claim and intended use.
- [Kaggle dataset](https://www.kaggle.com/datasets/splcher/animefacedataset), [KaggleHub](https://github.com/Kaggle/kagglehub), and [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): source access and implementation contracts.
''')

def build():
    nb=[];reading=[]
    import re
    for kind,text,provided in cells:
        if kind=='markdown':
            attachments={}
            def replace(m):
                rel=m.group(2);p=R/rel;mime='image/svg+xml' if p.suffix=='.svg' else 'image/png'
                key=p.name;attachments[key]={mime:base64.b64encode(p.read_bytes()).decode()}
                return f'![{m.group(1)}](attachment:{key})'
            embedded=re.sub(r'!\[([^]]*)\]\((assets/[^)]+)\)',replace,text)
            c=nbformat.v4.new_markdown_cell(embedded)
            if attachments:c['attachments']=attachments
            nb.append(c);reading.append(text)
        else:
            c=nbformat.v4.new_code_cell(text)
            if provided:c.metadata={'tags':['provided-infrastructure']}
            nb.append(c)
            block='```python\n'+text+'\n```'
            if provided and len(text)>3000:block='<details>\n<summary>Provided infrastructure · expand code</summary>\n\n'+block+'\n\n</details>'
            reading.append(block)
    for i,c in enumerate(nb):c['id']=f'anime-final-{i:03d}'
    notebook=nbformat.v4.new_notebook(cells=nb,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbformat.validate(notebook);nbformat.write(notebook,R/'starter/anime_faces_homework.ipynb')
    front='''---
title: "Anime Faces · Reading Notebook"
subtitle: "An independent generative-model investigation"
toc: true
toc-depth: 2
execute:
  enabled: false
format:
  html:
    include-in-header:
      text: |
        <style>#title-block-header.quarto-title-block {display:block} main h1:not(.title){font-size:1.8rem} main h2{font-size:1.4rem} details{margin:1rem 0} summary{cursor:pointer;font-weight:600}</style>
---

[Assignment](index.qmd) · <a href="starter/anime_faces_homework.ipynb?download=1" download="anime_faces_homework.ipynb">Download notebook</a> · [Starter ZIP](assets/anime-homework.zip)

This companion and the notebook are built from one source. The previews and diagnostic controls below are measured infrastructure examples, not a trained-model solution. Website rendering never downloads the dataset or trains a model.

'''
    (R/'reading.qmd').write_text(front+'\n\n'.join(reading)+'\n')
    with zipfile.ZipFile(R/'assets/anime-homework.zip','w',zipfile.ZIP_DEFLATED) as z:
        for src,dst in [('starter/anime_faces_homework.ipynb','anime_faces_homework.ipynb'),('support.py','support.py'),('assets/_report-template.qmd','report_template.qmd'),('README.md','README.md')]:z.write(R/src,dst)
    print('Built',len(nb),'cells, reading companion, and starter ZIP.')
if __name__=='__main__':build()
