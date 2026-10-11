"""Provided lab infrastructure, embedded verbatim in the student notebook.
No student TODOs: preparation, training, checkpoints, plots, and controls.
"""
import os, json, time, hashlib, urllib.request
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST

SEED = 117
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(4, os.cpu_count() or 1))
OUT = Path(os.environ.get('COURSE_OUTPUT_DIR', 'vae-results'))
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = os.environ.get('COURSE_DATA_DIR', 'data')
ASSET_URL = 'https://anassbelcaid.github.io/deeplearning-course/labs/11-variational-autoencoders/assets/'
NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
BETAS = [0., 1., 20.]


def prepare_data():
    ds = FashionMNIST(DATA_DIR, train=True, download=True)
    permutation = torch.randperm(len(ds), generator=torch.Generator().manual_seed(41))
    train_ids, val_ids = permutation[:12000], permutation[12000:14000]
    # Keep only this session's 14,000 images in float32 memory.
    train_x = ds.data[train_ids].unsqueeze(1).float().div(255)
    val_x = ds.data[val_ids].unsqueeze(1).float().div(255)
    val_y = ds.targets[val_ids]
    selected = torch.stack([(val_y == k).nonzero()[0, 0] for k in range(10)])
    return train_x, val_x, val_y, selected, val_ids


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GaussianEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.decoder(reparameterize(mu, logvar)), mu, logvar


def fresh_model():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return VAE().to(DEVICE)


def fit(model, train_x, val_x, beta=1., epochs=2, seconds=60., deterministic=False):
    """Time guard checked after each batch; validation uses a fixed noise stream."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader_rng = torch.Generator().manual_seed(41)
    start = time.perf_counter()
    history, steps = [], 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=loader_rng)
        totals = np.zeros(3); seen = 0
        for ids in order.split(256):
            x = train_x[ids].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            mu, lv = model.encoder(x)
            z = mu if deterministic else reparameterize(mu, lv)
            output = model.decoder(z)
            loss, rec, kl = vae_loss(output, x, mu, lv, beta)
            loss.backward(); optimizer.step()
            totals += np.array([loss.item(), rec.item(), kl.item()]) * len(x)
            seen += len(x); steps += 1
            if time.perf_counter() - start >= seconds:
                break
        stats = evaluate(model, val_x, deterministic=deterministic)
        history.append(dict(epoch=epoch+1, train_rec=float(totals[1]/seen),
                            train_kl=float(totals[2]/seen), **stats))
        print(f"epoch {epoch+1}: val reconstruction={stats['rec']:.2f}, KL={stats['kl']:.2f}", flush=True)
        if time.perf_counter() - start >= seconds:
            break
    return dict(history=history, seconds=time.perf_counter()-start,
                steps=steps, completed_epochs=len(history))


@torch.inference_mode()
def evaluate(model, x, deterministic=False):
    model.eval(); totals = np.zeros(2)
    rng = torch.Generator().manual_seed(811)
    for cpu_x in x.split(256):
        batch = cpu_x.to(DEVICE)
        mu, lv = model.encoder(batch)
        eps = torch.randn(mu.shape, generator=rng).to(DEVICE)
        z = mu if deterministic else reparameterize(mu, lv, eps)
        _, rec, kl = vae_loss(model.decoder(z), batch, mu, lv)
        totals += np.array([rec.item(), kl.item()])*len(batch)
    return dict(rec=float(totals[0]/len(x)), kl=float(totals[1]/len(x)))


def asset_file(relative):
    """Use local course files when available, otherwise the public course site."""
    explicit = os.environ.get('VAE_LAB_ASSETS')
    roots = ([Path(explicit)] if explicit else []) + [Path('assets'), Path('../assets'),
        Path('quarto-site/labs/11-variational-autoencoders/assets')]
    for root in roots:
        candidate = root/relative
        if candidate.is_file(): return candidate
    cache = OUT/'downloaded'/relative
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(ASSET_URL+relative, timeout=30) as response:
                cache.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError('Course asset unavailable. Use the supplied session ZIP, '
                               'extract it, and open starter/vae_lab.ipynb.') from exc
    return cache


def load_reference_models():
    manifest = json.loads(asset_file('checkpoints/manifest.json').read_text())
    models = {}
    for key, record in manifest['models'].items():
        path = asset_file('checkpoints/'+record['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], 'Checkpoint checksum mismatch'
        model = VAE().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval(); models[key] = model
    return models, manifest


def save_figure(fig, filename):
    fig.savefig(OUT/filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig)


def image_rows(rows, labels, filename, titles=None):
    count = len(rows[0]); fig, axes = plt.subplots(len(rows), count,
        figsize=(max(9, count*1.12), len(rows)*1.5), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r,c]; ax.imshow(img.detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if c == 0: ax.set_ylabel(labels[r], fontsize=10)
            if r == 0 and titles: ax.set_title(titles[c], fontsize=9)
    fig.tight_layout(); save_figure(fig, filename)


@torch.inference_mode()
def opening_gallery(ae, val_x, selected):
    x = val_x[selected].to(DEVICE)
    mu, _ = ae.encoder(x)
    prior = torch.randn((10,2), generator=torch.Generator().manual_seed(91)).to(DEVICE)
    image_rows([x, ae.decoder(mu), ae.decoder(prior)],
        ['Input', 'AE reconstruction', 'AE random codes'], 'opening.png', NAMES)


@torch.inference_mode()
def posterior_view(model, val_x, selected, category=9, draw_seed=8, filename='posterior.png'):
    x = val_x[selected[category]:selected[category]+1].to(DEVICE)
    mu, lv = model.encoder(x)
    eps = torch.randn((8,2), generator=torch.Generator().manual_seed(draw_seed)).to(DEVICE)
    z = reparameterize(mu.expand(8,-1), lv.expand(8,-1), eps)
    images = torch.cat([x, model.decoder(mu), model.decoder(z)])
    image_rows([images], ['Same input'], filename,
        ['Input', 'Decode μ']+[f'Draw {i+1}' for i in range(8)])
    fig, ax = plt.subplots(figsize=(7,4.5))
    mean = mu.cpu().numpy()[0]; sigma = (lv*.5).exp().cpu().numpy()[0]
    ax.add_patch(Ellipse(mean, 2*sigma[0], 2*sigma[1], facecolor='#5599bb', alpha=.15))
    points = z.cpu().numpy()
    ax.scatter(*points.T, color='#bd573e', label='8 posterior draws')
    ax.scatter(*mean, marker='x', s=80, color='black', label='Posterior mean')
    for i, point in enumerate(points): ax.annotate(str(i+1), point, xytext=(4,4), textcoords='offset points')
    ax.set(xlim=(mean[0]-3*sigma[0],mean[0]+3*sigma[0]),
           ylim=(mean[1]-3*sigma[1],mean[1]+3*sigma[1]), xlabel='z₁',ylabel='z₂',
           title=f'{NAMES[category]}: μ={mean.round(2)}, σ={sigma.round(2)}')
    ax.legend(loc='best'); fig.tight_layout()
    save_figure(fig, filename.replace('.png','-cloud.png'))


@torch.inference_mode()
def prior_view(model, draw_seed=91, filename='prior-draw.png'):
    z = torch.randn((10,2), generator=torch.Generator().manual_seed(draw_seed)).to(DEVICE)
    image_rows([model.decoder(z)], ['Prior → decoder'], filename,
               [f'({a:.1f}, {b:.1f})' for a,b in z.cpu().tolist()])


@torch.inference_mode()
def comparison(models, val_x, val_y, selected):
    x = val_x[selected].to(DEVICE)
    prior = torch.randn((10,2), generator=torch.Generator().manual_seed(91)).to(DEVICE)
    recons, samples, labels, means, draws = [x], [], ['Input'], [], []
    for beta in BETAS:
        model = models[f'beta{beta:g}']; mu, lv = model.encoder(x)
        recons.append(model.decoder(mu)); labels.append(f'β = {beta:g}')
        samples.append(model.decoder(prior))
        all_mu, all_lv = model.encoder(val_x[:600].to(DEVICE))
        noise = torch.randn(all_mu.shape, generator=torch.Generator().manual_seed(81)).to(DEVICE)
        means.append(all_mu.cpu().numpy())
        draws.append(reparameterize(all_mu, all_lv, noise).cpu().numpy())
    image_rows(recons, labels, 'reconstructions.png', NAMES)
    image_rows(samples, labels[1:], 'prior-comparison.png', [f'z {i+1}' for i in range(10)])
    # Shared axes include every displayed point and the reference prior region.
    bound = max(3.5, float(np.abs(np.concatenate(means+draws)).max())*1.05)
    fig, axes = plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    for j,beta in enumerate(BETAS):
        for r, points in enumerate([means[j], draws[j]]):
            ax = axes[r,j]
            ax.scatter(*points.T,c=val_y[:600],cmap='tab10',vmin=-.5,vmax=9.5,s=7,alpha=.5)
            ax.add_patch(plt.Circle((0,0),2.45,fill=False,color='black',ls='--'))
            ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal',xlabel='z₁',ylabel='z₂',
                   title=f'β={beta:g} · '+('posterior means' if r==0 else 'posterior draws'))
    fig.suptitle('Same 600 validation images · shared axes · dashed circle: 95% prior mass',fontsize=13)
    fig.tight_layout(); save_figure(fig,'latent-maps.png')
    # The unregularized codes set a much larger scale. A labeled second view
    # preserves the overview while making the regularized distributions legible.
    bound = max(3.5, float(np.abs(np.concatenate(means[1:]+draws[1:])).max())*1.05)
    fig, axes = plt.subplots(2,2,figsize=(9,8),sharex=True,sharey=True)
    for j,beta in enumerate(BETAS[1:]):
        for r,points in enumerate([means[j+1],draws[j+1]]):
            ax=axes[r,j]
            ax.scatter(*points.T,c=val_y[:600],cmap='tab10',vmin=-.5,vmax=9.5,s=9,alpha=.5)
            ax.add_patch(plt.Circle((0,0),2.45,fill=False,color='black',ls='--'))
            ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal',xlabel='z₁',ylabel='z₂',
                   title=f'β={beta:g} · '+('posterior means' if r==0 else 'posterior draws'))
    handles=[plt.Line2D([],[],marker='o',ls='',color=plt.cm.tab10(i),label=name)
             for i,name in enumerate(NAMES)]
    fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=9)
    fig.suptitle('Regularized models · closer view · same points, new shared axes',fontsize=12)
    fig.tight_layout(rect=(0,.065,1,.97)); save_figure(fig,'latent-zoom.png')


@torch.inference_mode()
def latent_use(models, val_x):
    """Deterministic decode-mean probe; separate from stochastic ELBO evaluation."""
    x=val_x.to(DEVICE)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(31)).to(DEVICE)
    scores=[]
    for beta in BETAS:
        model=models[f'beta{beta:g}']; mu,_=model.encoder(x)
        matched=F.binary_cross_entropy(model.decoder(mu),x,reduction='none').flatten(1).sum(1).mean().item()
        shuffled=F.binary_cross_entropy(model.decoder(mu[order]),x,reduction='none').flatten(1).sum(1).mean().item()
        scores.append(dict(beta=beta,matched=matched,shuffled=shuffled,gap=shuffled-matched))
    fig,ax=plt.subplots(figsize=(8,4)); loc=np.arange(3)
    ax.bar(loc-.17,[s['matched'] for s in scores],.34,label='Matched μ',color='#4c91ad')
    ax.bar(loc+.17,[s['shuffled'] for s in scores],.34,label='Shuffled μ',color='#ce7154')
    ax.set(xticks=loc,xticklabels=[f'β={b:g}' for b in BETAS],ylabel='Mean pixel-summed BCE',
           title='Does assigning the right image-specific code matter?')
    ax.legend(); fig.tight_layout(); save_figure(fig,'latent-use.png')
    return scores


def training_curves(manifest):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for beta,color in zip(BETAS,['#4c91ad','#ce7154','#84669d']):
        history=manifest['models'][f'beta{beta:g}']['history']
        for ax,key in zip(axes,['rec','kl']):
            ax.plot([h['epoch'] for h in history],[h[key] for h in history],label=f'β={beta:g}',color=color)
    for ax,title in zip(axes,['Validation reconstruction cost','Validation KL per image']):
        ax.set(xlabel='Epoch',ylabel='Nats / image',title=title); ax.legend()
    fig.tight_layout(); save_figure(fig,'training-curves.png')


def controls(models, val_x, selected):
    """Optional controls; equivalent static function calls always work."""
    try:
        import ipywidgets as w
        from IPython.display import display, clear_output
        model_select=w.Dropdown(options=[('β = '+str(b),f'beta{b:g}') for b in BETAS],value='beta1',description='Model')
        garment=w.Dropdown(options=list(zip(NAMES,range(10))),value=9,description='Input')
        posterior=w.Button(description='Resample posterior'); prior=w.Button(description='Draw from prior')
        output=w.Output(); state={'seed':8}
        def update(kind):
            state['seed']+=1
            with output:
                clear_output(wait=True)
                print('Weights fixed. New random draw; posterior also uses the selected input.')
                model=models[model_select.value]
                if kind=='posterior': posterior_view(model,val_x,selected,garment.value,state['seed'],'interactive-posterior.png')
                else: prior_view(model,state['seed'],'interactive-prior.png')
        posterior.on_click(lambda _: update('posterior'))
        prior.on_click(lambda _: update('prior'))
        display(w.VBox([w.HBox([model_select,garment]),w.HBox([posterior,prior]),output]))
        return dict(model=model_select,garment=garment,posterior=posterior,prior=prior,output=output)
    except ImportError:
        print('Widgets unavailable: use posterior_view(...) and prior_view(...) below.')
