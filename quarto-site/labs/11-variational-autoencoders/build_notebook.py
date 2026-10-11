"""Single-source student tasks, solutions, Quarto companion, and offline session ZIP.
Run after build_evidence.py. Site rendering never runs training.
"""
from pathlib import Path
from textwrap import dedent
import base64, json, re, zipfile
ROOT=Path(__file__).resolve().parent
CELLS=[]
def md(text): CELLS.append(dict(kind='markdown',text=dedent(text).strip()))
def code(text,provided=False,solution=None):
    CELLS.append(dict(kind='code',text=dedent(text).strip(),provided=provided,solution=solution))
def picture(file,caption): md(f'![{caption}](assets/{file})')
reference=(ROOT/'reference_model.py').read_text()
solutions={key:reference.split('# BEGIN '+key+'\n')[1].split('# END '+key)[0].strip() for key in 'ABC'}
metrics=json.loads((ROOT/'assets/expected/metrics.json').read_text())
manifest=json.loads((ROOT/'assets/checkpoints/manifest.json').read_text())

md('''
**90-minute homework / guided coding session · Fashion-MNIST · Chapter 11**

**What must change so a decoder can generate images from fresh random codes?**

An autoencoder can reconstruct a boot and still produce a poor image from a fresh random code.
Your goal is to implement the three changes that turn this problem into a VAE experiment:
a Gaussian encoder, a differentiable draw, and reconstruction plus KL.

Only **Tasks A, B, and C** require implementation. Data preparation, the decoder,
training, checks, checkpoints, plots, and controls are provided. You do not need the
previous homework's files. Familiarity with its encoder–decoder idea is enough.

| Clock | Focus | Stop when you have… |
|---|---|---|
| 0–10 min | Observe and prepare | A prediction and loaded data |
| 10–20 min | A · Gaussian encoder | Two unrestricted `(B, 2)` outputs |
| 20–30 min | B · Reparameterization | Passing checks and repeated posterior outputs |
| 30–45 min | C · Objective | Correct reductions and gradient checks |
| 45–60 min | Short training run | Separate reconstruction/KL measurements |
| 60–80 min | Controlled experiment | Fixed-image and fixed-prior comparisons |
| 80–90 min | Interpret and submit | One evidence figure and three short answers |

If a task blocks you for more than its allotted time, open its worked solution,
record that you used it, and continue. Interpretation matters more than hiding assistance.
''')
picture('expected/opening.png','Measured reference: original images, deterministic AE reconstructions, and decoding ten standard-normal codes. Columns in the bottom row are independent draws, not reconstructions of the images above.')
md('''
## 0 · Predict before coding · 0–10 min

**Predict:** why can the middle row look more plausible than the bottom row?
Name what the autoencoder was trained to do and what the random-code experiment asks it to do.

**Setup.** Open in Colab, or use a local Python environment with `torch`, `torchvision`,
`numpy`, `matplotlib`, and optionally `ipywidgets`. CPU is sufficient. Run the next
provided cell; it defines the support functions but does not train anything.
The notebook downloads Fashion-MNIST and about 3.5 MB of course checkpoints on first use.
For class, download the session ZIP and run the preparation cell before the timed session.
Extract the ZIP and open `starter/vae_lab.ipynb`; the adjacent assets work without a course-server connection.
Fashion-MNIST must still be downloaded once (or available through `COURSE_DATA_DIR`).

**Time protection:** the live run is limited to two small epochs or about 60 seconds
(checked after each batch). The comparison uses supplied 20-epoch models; you never
have to train the KL sweep. If setup fails, use the visual reading page with your
instructor and complete the code later; do not spend the session debugging downloads.
''')
code((ROOT/'support.py').read_text(),provided=True)
code('''
train_x, val_x, val_y, selected, val_ids = prepare_data()
print('Training:', tuple(train_x.shape), '| Validation:', tuple(val_x.shape))
print('Device:', DEVICE)
''',provided=True)
md('''
**Expected:** `(12000, 1, 28, 28)` training and `(2000, 1, 28, 28)` validation.
A fixed permutation (seed 41) selects disjoint subsets of the official training split.
The official test split is unused: this session investigates mechanisms, not final performance.
Labels are only used to choose and color inspection examples. They never enter the loss.

## 1 · A distribution instead of one code · 10–20 min

The provided `VAE` owns an encoder and decoder. Your encoder owns a shared feature
extractor and **two different linear heads**. The decoder structure is already familiar.
''')
picture('ownership.svg','Course-made ownership and tensor-flow diagram: the two encoder heads feed a sampled code, while generation sends a fresh prior code directly to the same decoder.')
md(r'''
### Task A · Complete the Gaussian encoder

**Given:** `features(x)` has shape `(B, 64)` and the latent dimension is 2.
**Implement:** two independent `nn.Linear(64, 2)` heads and return `(mu, logvar)`.
Both outputs are unrestricted real values. Do not add Sigmoid, ReLU, or sampling here.
`logvar` represents $\log\sigma^2$, not $\sigma$ or $\log\sigma$.

> **PAUSE · 7 minutes.** Trace the diagram into module attributes before editing the code.
''')
code('''
class GaussianEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        # TODO A: create self.mu and self.logvar as separate heads.
        raise NotImplementedError('Task A: create the two heads')

    def forward(self, x):
        h = self.features(x)
        # TODO A: return both head outputs, with no output activation.
        raise NotImplementedError('Task A: return mu, logvar')
''',solution=solutions['A'])
code('''
encoder = GaussianEncoder()
mu, logvar = encoder(torch.zeros(4, 1, 28, 28))
assert mu.shape == logvar.shape == (4, 2)
assert encoder.mu is not encoder.logvar
assert encoder.mu.weight.data_ptr() != encoder.logvar.weight.data_ptr()
# Heads must be able to emit negative means AND positive log variances.
with torch.no_grad():
    encoder.mu.weight.zero_(); encoder.mu.bias.fill_(-2.)
    encoder.logvar.weight.zero_(); encoder.logvar.bias.fill_(1.)
a, b = encoder(torch.zeros(4, 1, 28, 28))
torch.testing.assert_close(a, torch.full_like(a, -2.))
torch.testing.assert_close(b, torch.ones_like(b))
print('PASS A: shapes, separate parameters, unrestricted outputs')
''',provided=True)
md(r'''
## 2 · Draw a code without cutting the gradient · 20–30 min

### Task B · Reparameterize

**Given:** `mu`, `logvar`, and optional fixed standard-normal noise `eps`, all shaped `(B, 2)`.
**Implement:** $z=\mu+\exp(\tfrac12\,\mathrm{logvar})\odot\epsilon$.
If `eps` is absent, create it with `torch.randn_like(mu)`; otherwise use the supplied values.
Preserve dtype, device, and the gradient path. Do not detach or convert to NumPy.

> **PAUSE · 5 minutes.** First work out why a variance of 4 must multiply noise by 2, not 4.
''')
code('''
def reparameterize(mu, logvar, eps=None):
    # TODO B: draw noise only if needed, then shift and scale it.
    raise NotImplementedError('Task B')
''',solution=solutions['B'])
code('''
mu = torch.tensor([[1., -1.]], requires_grad=True)
lv = torch.tensor([[4., .25]]).log().requires_grad_()
eps = torch.tensor([[.5, -2.]])
z = reparameterize(mu, lv, eps)
torch.testing.assert_close(z, torch.tensor([[2., -2.]]))
g_mu, g_lv = torch.autograd.grad(z.sum(), (mu, lv))
torch.testing.assert_close(g_mu, torch.ones_like(mu))
torch.testing.assert_close(g_lv, torch.tensor([[.5, -.5]]))
assert z.dtype == mu.dtype and z.device == mu.device
samples = reparameterize(torch.zeros(20000, 2), torch.zeros(20000, 2))
assert samples.mean().abs() < .04 and abs(samples.std().item()-1) < .04
print('PASS B: fixed-noise arithmetic, gradient paths, standard-normal sampling')
models, manifest = load_reference_models()
posterior_view(models['beta1'], val_x, selected, category=9, draw_seed=8)
''',provided=True)
picture('expected/posterior.png','Reference β=1 model: the same boot, decoding its posterior mean, then eight posterior draws. Each image is a decoder output, not a sampled binary pixel image.')
picture('expected/posterior-cloud.png','The eight numbered latent draws used above. The ellipse has radii equal to the two posterior standard deviations; it is not a 95% probability contour.')
md('''
**Interpret:** the weights and input stayed fixed. Only the noise changed. Decoding the
posterior mean is deterministic here, but it is not generally the expected decoded image.
`eval()` does not disable a random draw written explicitly in the forward method.
The modest visible variation in this example is evidence to inspect, not a promise
that every input has equally varied outputs.
''')
md(r'''
## 3 · Make the training objective precise · 30–45 min

### Task C · Reconstruction and KL

**Given:** `x_hat, x` shaped `(B, 1, 28, 28)`; `mu, logvar` shaped `(B, 2)`.
**Return three scalars:** mean total loss, mean reconstruction cost, and mean **unweighted** KL.

$$
\begin{aligned} R_i&=\sum_{p=1}^{784}\operatorname{BCE}(\hat x_{ip},x_{ip}),\\
K_i&=\frac12\sum_{j=1}^2(\mu_{ij}^2+e^{\ell_{ij}}-1-\ell_{ij}),\\
L&=\frac1B\sum_i(R_i+\beta K_i).\end{aligned}
$$

Sum over each image's pixels and latent coordinates **before** averaging over images.
A pixel-mean reconstruction term would be 784 times smaller and would change the effective
KL weight. Fractional Fashion-MNIST intensities are soft BCE targets; this is the course's
reconstruction convention, not an exact continuous-pixel likelihood model.
At $\beta=1$ this is the usual VAE objective under that convention; other weights are
controlled ablations and should not all be called the standard negative ELBO.

> **PAUSE · 9 minutes.** Annotate the shape after each reduction. Keep KL unweighted in the reported metric.
''')
code('''
def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    # TODO C: per-image reconstruction and KL, then batch means.
    raise NotImplementedError('Task C')
''',solution=solutions['C'])
code('''
x = torch.zeros(2, 1, 28, 28)
x_hat = torch.full_like(x, .5)
mu = torch.tensor([[1., 0.], [0., 2.]])
lv = torch.zeros_like(mu)
loss, rec, kl = vae_loss(x_hat, x, mu, lv, beta=2.)
torch.testing.assert_close(rec, torch.tensor(784*np.log(2), dtype=rec.dtype))
torch.testing.assert_close(kl, torch.tensor(1.25))
torch.testing.assert_close(loss, rec + 2*kl)
assert vae_loss(x_hat,x,torch.zeros_like(mu),lv)[2].item() == 0.
for a,b in zip(vae_loss(x_hat,x,mu,lv,2.),
               vae_loss(*[torch.cat([v,v]) for v in [x_hat,x,mu,lv]],beta=2.)):
    torch.testing.assert_close(a,b)
# Non-unit variance catches omitted exp(logvar) and incorrect signs.
variance_kl = vae_loss(x_hat,x,torch.zeros_like(mu),torch.full_like(mu,np.log(4)))[2]
torch.testing.assert_close(variance_kl, torch.tensor(3-np.log(4),dtype=variance_kl.dtype))
model = fresh_model()
batch = train_x[:16].to(DEVICE)
x_hat, mu, lv = model(batch)
_, reconstruction, _ = vae_loss(x_hat,batch,mu,lv)
grads = torch.autograd.grad(reconstruction,
    [model.encoder.mu.weight, model.encoder.logvar.weight])
assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
print('PASS C: reductions, KL cases, batch duplication, reconstruction gradients to both heads')
''',provided=True)
md('''
**Numerical check:** reconstruction is about **543.43**, KL is **1.25**, and the
β=2 total is about **545.93**. Duplicating the batch leaves all three means unchanged.
A nonzero KL gradient alone would not prove that reconstruction can train both heads;
the final check therefore differentiates reconstruction alone.

## 4 · Run the mechanism you implemented · 45–60 min

Run the provided training cell. It creates a fresh model using **your three implementations**.
The short run uses 4,096 training and 512 validation images. It verifies learning and
produces an early gallery; it is not expected to match the longer reference models.
Read `fit` in the folded support cell if you want to trace zeroing gradients, backward,
and the optimizer step. There is no training-loop implementation task.
''')
code('''
live_model = fresh_model()
live_run = fit(live_model, train_x[:4096], val_x[:512], beta=1., epochs=2, seconds=60.)
with torch.inference_mode():
    examples = val_x[selected].to(DEVICE)
    mu, _ = live_model.encoder(examples)
    image_rows([examples, live_model.decoder(mu)],
               ['Input', 'Short live run'], 'my-short-run.png', NAMES)
print('Measured seconds:', round(live_run['seconds'], 2), '| updates:', live_run['steps'])
''',provided=True)
picture('expected/short-run.png','Measured two-epoch reference for the short student run. Early blur is expected; this gallery is separate from the 20-epoch comparison below.')
bench=metrics['live_benchmark']
md(f"""
**Recorded timing on the reference machine:** {bench['seconds']:.1f} seconds for
{bench['steps']} optimizer updates on {manifest['device']} with PyTorch {manifest['torch']}.
Hardware and download times vary. The timer excludes data preparation; its stop is checked
between batches and a final validation pass adds a little time. If computation is slow,
continue with the supplied checkpoints. No marks depend on runtime or image sharpness.

**Discuss:** which term is larger? Does a falling total alone tell you whether KL is being ignored?
Do not compare the short-run score with the 20-epoch score as if their training budgets matched.
""")
md('''
## 5 · A controlled KL experiment · 60–80 min

**Before running:** predict how β=0, 1, and 20 might affect reconstruction, prior samples,
and dependence on the input. Then run the provided cells and revise your prediction.

All three reference models use the same architecture, initial parameter values, training
split, shuffled-batch stream, 20-epoch budget, and optimizer settings. β=0 still samples
from an encoder distribution: it is an **unregularized stochastic autoencoder**, distinct
from the deterministic AE in the opening. These are single-seed experiments.

The reconstruction galleries decode **posterior means** to remove sampling noise.
The reported validation reconstruction cost instead uses one fixed posterior draw per image.
The prior gallery uses identical latent coordinates across models, although a given coordinate
need not acquire the same semantic meaning after different training runs.
''')
code('''
comparison(models, val_x, val_y, selected)
training_curves(manifest)
probe_scores = latent_use(models, val_x)
for beta in BETAS:
    key = f'beta{beta:g}'
    print(key, evaluate(models[key], val_x))
print('Matched/shuffled mean-code probe:', probe_scores)
''',provided=True)
picture('expected/reconstructions.png','Measured comparison: the same ten class-selected validation images and posterior-mean reconstructions for three KL weights. Selection is illustrative, not a representative quality score.')
picture('expected/prior-comparison.png','Measured comparison: the same ten standard-normal codes decoded by each model. Inputs and encoder are not used. Compare diversity, silhouettes, and repeated or ambiguous outputs.')
picture('expected/latent-maps.png','Actual two-dimensional latent coordinates for the same 600 validation images, with shared axes. Top: posterior means. Bottom: one posterior draw per image. Colors indicate labels after training; the dashed circle contains 95% of standard-normal prior mass.')
picture('expected/latent-zoom.png','Closer view of the same β=1 and β=20 points on new shared axes. The unregularized model is omitted from this explicitly labeled zoom so posterior spread remains visible.')
md('''
**Read the maps carefully:** a cluster of means near zero does not imply that posterior
draws have zero spread. Nor does a visually prior-like cloud establish that codes carry
useful image information. These panels separate the two questions; they are not PCA views.
''')
picture('expected/training-curves.png','Recorded validation curves: reconstruction and unweighted KL shown separately over the same 20 epochs. Weighted totals across different β values are different objectives, not comparable quality scores.')
rows=['| β | Validation reconstruction | Validation KL | Matched μ BCE | Shuffled μ BCE |',
      '|---:|---:|---:|---:|---:|']
for beta,probe in zip([0,1,20],metrics['latent_use']):
    record=metrics['reference'][f'beta{beta}']
    rows.append(f"| {beta} | {record['rec']:.2f} | {record['kl']:.2f} | {probe['matched']:.2f} | {probe['shuffled']:.2f} |")
md('\n'.join(rows)+'\n\nAll costs are pixel-summed/latent-summed, averaged over 2,000 validation images. These are reference observations, not targets to match.')
picture('expected/latent-use.png','A separate deterministic probe: decode each image’s own posterior mean, then assign shuffled means to the same targets. A small penalty for shuffling is evidence to investigate weak image-specific latent use; it is not by itself a proof of complete posterior collapse.')
md('''
<details>
<summary>Interpretation checkpoint · open after comparing the panels</summary>

In this recorded run, β=0 reconstructs well but its codes are far from the prior region,
and fresh prior outputs are poor. β=1 makes the prior draws more useful with a modest
reconstruction cost increase. β=20 reduces KL and produces blurrier reconstructions,
but shuffling its mean codes still substantially increases reconstruction cost:
**these results do not show complete posterior collapse**. A stronger penalty does not
automatically imply that the model ignores its input. Cite the actual probe and images.

</details>
''')
md('''
### Try changing only the random draw

Run the controls, choose a model and input, then click **Resample posterior**.
Use **Draw from prior** to remove the input entirely. Weights stay fixed.
Widgets are optional: the static calls below provide the same experiment everywhere.
''')
code('''
lab_controls = controls(models, val_x, selected)
# Static equivalents; change category (0–9) or draw_seed and rerun.
posterior_view(models['beta1'], val_x, selected, category=9, draw_seed=9,
               filename='my-posterior.png')
prior_view(models['beta1'], draw_seed=92, filename='my-prior.png')
''',provided=True)
picture('expected/prior-draw.png','Reference prior experiment: the displayed coordinates are the actual codes used to generate these outputs. No input image was encoded.')
md('''
## 6 · Submit evidence, not a training marathon · 80–90 min

Submit this notebook with A–C completed, all checks passing, and visible outputs from the
short run (or a recorded reason for using the checkpoint fallback). Use **Restart & Run All**
after editing; all later cells use provided infrastructure. Keep the data and checkpoints
out of your submission.

Attach **one figure with your fixed-image reconstruction and prior-sample comparisons**
(the cell below creates it), then write three short answers in the final markdown cell.
There is no additional report, sweep, test-set search, PCA task, or interpolation task.

1. **Sampling:** identify what changes and what remains fixed in a posterior draw versus
   a prior draw. Explain the role of `0.5 * logvar` using the variance-4 numerical check.
2. **Trade-off:** cite two specific panels/images and the separate reconstruction/KL values
   to explain what changed across β. Do not use a weighted-total ranking.
3. **Latent use and limits:** use the matched/shuffled probe together with a visible output
   to assess whether image-specific codes matter. State one limitation of this experiment
   (for example, one seed, selected images, small latent space, or a short training budget).

**Assessment:** implementations and behavioral checks 50%; controlled visual evidence 20%;
mechanistic explanations and limitations 30%. Reference images help you interpret your run;
do not present them as images you trained. Disclose any worked solution you consulted.
''')
code('''
# Combine YOUR generated comparison plots into one submission figure.
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
for ax, filename, title in zip(axes,
    ['reconstructions.png', 'prior-comparison.png'],
    ['Fixed inputs → posterior-mean reconstructions', 'Fixed prior codes → outputs']):
    ax.imshow(plt.imread(OUT/filename)); ax.set_title(title); ax.axis('off')
fig.tight_layout(); save_figure(fig, 'submission-evidence.png')
''',provided=True)
md('''
### Your three answers

**1 · Sampling:** write here.

**2 · Trade-off:** write here.

**3 · Latent use and limits:** write here.

**Assistance / fallback used:** record here, or write “none”.

### Reading and provenance

- [Chapter 11 notes](https://anassbelcaid.github.io/deeplearning-course/notes/11-variational-autoencoders/): derivation, implementation contracts, and evaluation caveats.
- [Kingma & Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): original VAE mechanism.
- [Fashion-MNIST](https://arxiv.org/abs/1708.07747): dataset by Xiao, Rasul, and Vollgraf. All garment images here come from that dataset; model outputs and figures were generated for this session.
- [PyTorch checkpoint loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html): the provided loader uses tensor-only state dictionaries with `weights_only=True`.

Reference runs use seed 117, split/loader seed 41, Adam at 0.001, batch size 256,
and a 784→128→64 encoder with two 2-D heads and a 2→64→128→784 decoder.
The opening deterministic AE has the same active encoder/decoder dimensions and training
budget; its unused variance head is retained only for checkpoint compatibility.
Rebuilding the site reads stored assets and never downloads data or trains models.
''')

def build():
    import nbformat
    cells=[]; reading=[]
    for i,c in enumerate(CELLS):
        text=c['text']
        if c['kind']=='markdown':
            # Notebook attachments preserve all expected visuals offline.
            attachments={}
            def embed(match):
                alt,file=match.groups(); path=ROOT/'assets'/file
                mime='image/svg+xml' if path.suffix=='.svg' else 'image/png'
                key=path.name
                attachments[key]={mime:base64.b64encode(path.read_bytes()).decode()}
                return f'![{alt}](attachment:{key})'
            nbtext=re.sub(r'!\[([^\]]*)\]\(assets/([^\)]+)\)',embed,text)
            cell=nbformat.v4.new_markdown_cell(nbtext)
            if attachments: cell['attachments']=attachments
            cells.append(cell); reading.append(text)
        else:
            cell=nbformat.v4.new_code_cell(text)
            if c['provided']: cell['metadata']['jupyter']={'source_hidden':True}
            cells.append(cell)
            if c['provided']:
                reading.append('<details>\n<summary>Provided code · run unchanged</summary>\n\n```python\n'+text+'\n```\n\n</details>')
            else: reading.append('```python\n'+text+'\n```')
            if c.get('solution'):
                solution='<details>\n<summary>Worked solution · open after your attempt</summary>\n\n```python\n'+c['solution']+'\n```\n\n</details>'
                cells.append(nbformat.v4.new_markdown_cell(solution)); reading.append(solution)
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}})
    # Stable IDs keep rebuilds reviewable.
    for i,c in enumerate(notebook.cells): c['id']=f'vae-lab-{i:03d}'
    nbformat.validate(notebook)
    nbformat.write(notebook,ROOT/'starter/vae_lab.ipynb')
    header='''---
title: "VAE Lab · Reading Notebook"
subtitle: "From reconstruction to generation · 90 minutes"
toc: true
toc-depth: 2
execute:
  enabled: false
format:
  html:
    include-in-header:
      text: |
        <style>#title-block-header.quarto-title-block { display: block; } details { margin: 1rem 0; } summary { cursor: pointer; font-weight: 600; } </style>
---

[Assignment brief](index.qmd) · <a href="starter/vae_lab.ipynb?download=1" download="vae_lab.ipynb">Download notebook</a> · [Session ZIP](assets/vae-session.zip)

This reading view and the notebook share one authoring source. Images below are measured reference outputs; code is displayed, not executed during website rendering. The notebook embeds the same visual guidance.

'''
    (ROOT/'reading.qmd').write_text(header+'\n\n'.join(reading)+'\n')
    with zipfile.ZipFile(ROOT/'assets/vae-session.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT/'starter/vae_lab.ipynb','starter/vae_lab.ipynb')
        for p in sorted((ROOT/'assets/checkpoints').glob('*')): z.write(p,'assets/checkpoints/'+p.name)
    print('Built',len(cells),'cells, reading companion, and session ZIP.')

if __name__=='__main__': build()
