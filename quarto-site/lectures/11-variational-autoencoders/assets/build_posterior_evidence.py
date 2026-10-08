"""Section 2: a reproducible posterior-sampling experiment, independent of rendering.

Reuses the section-1 architecture and training protocol, but caches its own run.
No selection by output quality. Run with --data pointing to FashionMNIST's parent.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from build_generation_evidence import VAE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence' / 'posterior'
COLORS = ['#2878c8', '#ef5b45', '#38865a']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/tmp/fashion-mnist')
    parser.add_argument('--checkpoint', default='/tmp/vae-section2.pt')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(113)
    ds = FashionMNIST(args.data, train=True, download=False, transform=ToTensor())
    perm = torch.randperm(60000, generator=torch.Generator().manual_seed(41))
    model = VAE()
    checkpoint = Path(args.checkpoint)
    history = []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        history = state['history']
    else:
        loader = DataLoader(Subset(ds, perm[:50000]), batch_size=256, shuffle=True,
                            generator=torch.Generator().manual_seed(41))
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        for epoch in range(10):
            total = 0.
            for x, _ in loader:
                opt.zero_grad()
                recon, mu, lv = model(x)
                rec = nn.functional.binary_cross_entropy(recon, x, reduction='sum') / len(x)
                kl = -.5 * (1 + lv - mu.square() - lv.exp()).sum(1).mean()
                loss = rec + kl
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            history.append(total / 50000)
            print(f'Epoch {epoch + 1}: {history[-1]:.4f}', flush=True)
        torch.save({'model': model.state_dict(), 'history': history}, checkpoint)
    model.eval()
    # First validation example of each specified class; chosen before inspecting outputs.
    classes = [(0, 'T-shirt/top'), (1, 'Trouser'), (9, 'Ankle boot')]
    selected = [next(int(i) for i in perm[50000:] if int(ds.targets[i]) == c) for c, _ in classes]
    x = torch.stack([ds[i][0] for i in selected])
    generator = torch.Generator().manual_seed(307)
    records = []
    with torch.inference_mode():
        h = model.encoder(x)
        mu, lv = model.mu(h), model.logvar(h)
        sigma = (.5 * lv).exp()
        noise = torch.randn(3, 8, 2, generator=generator)
        z = mu[:, None] + sigma[:, None] * noise
        decoded = model.decoder(z.reshape(-1, 2)).reshape(3, 8, 28, 28)
        means = model.decoder(mu).reshape(3, 28, 28)
        prior = torch.randn(8, 2, generator=generator)
        prior_outputs = model.decoder(prior).reshape(8, 28, 28)
    def save(name, image):
        plt.imsave(OUT / name, image.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    for i, (_, label) in enumerate(classes):
        save(f'input-{i}.png', x[i])
        save(f'mean-{i}.png', means[i])
        for j in range(8):
            save(f'sample-{i}-{j}.png', decoded[i, j])
        records.append({'label': label, 'dataset_index': selected[i], 'mu': mu[i].tolist(),
                        'logvar': lv[i].tolist(), 'sigma': sigma[i].tolist(), 'codes': z[i].tolist()})
    for j in range(8):
        save(f'prior-{j}.png', prior_outputs[j])
    plt.rcParams.update({'font.size': 17, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13, 4.8))
    layout = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=.22)
    ax = fig.add_subplot(layout[0])
    for radius in [1, 2]:
        ax.add_patch(Ellipse((0, 0), 2*radius, 2*radius, facecolor='#7655ad',
                            edgecolor='#7655ad', alpha=.06 if radius == 2 else .13))
    ax.scatter([0], [0], marker='+', s=150, color='#7655ad')
    for i, record in enumerate(records):
        m, s = np.array(record['mu']), np.array(record['sigma'])
        ax.add_patch(Ellipse(m, *(2*s), facecolor=COLORS[i], edgecolor=COLORS[i], alpha=.28))
        ax.scatter(*m, color=COLORS[i], s=20, label=record['label'])
        ax.annotate(str(i+1), m, xytext=(9, 7), textcoords='offset points', color=COLORS[i], fontsize=19)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='Latent coordinate z₁', ylabel='Latent coordinate z₂')
    ax.set_aspect('equal')
    previews = layout[1].subgridspec(3, 2, width_ratios=[1, 3], hspace=.18, wspace=.13)
    for i, r in enumerate(records):
        thumb = fig.add_subplot(previews[i, 0])
        thumb.imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        thumb.axis('off')
        label = fig.add_subplot(previews[i, 1]); label.axis('off')
        label.text(0, .80, f"{i+1} · {r['label']}", fontsize=19, color=COLORS[i], weight='bold')
        label.text(0, .42, 'μ = (' + ', '.join(f'{v:.2f}' for v in r['mu']) + ')', fontsize=18)
        label.text(0, .05, 'σ = (' + ', '.join(f'{v:.2f}' for v in r['sigma']) + ')', fontsize=18)
    fig.subplots_adjust(left=.065, right=.98, bottom=.14, top=.96); fig.savefig(OUT/'prior-posterior.svg', facecolor='#f5f1e8'); plt.close(fig)
    # Teaching example: deliberately chosen parameters, not inferred model values.
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.3))
    grid = np.linspace(-3, 3, 1500)
    for j, (m, s) in enumerate([(1., .25), (-.5, 1.)]):
        density = np.exp(-.5*((grid-m)/s)**2)/(s*np.sqrt(2*np.pi))
        axes[j].plot(grid, density, color=COLORS[j], lw=3)
        axes[j].fill_between(grid, density, where=(grid>=m-s)&(grid<=m+s), color=COLORS[j], alpha=.2)
        axes[j].axvline(m, color=COLORS[j], ls='--', lw=1.5)
        axes[j].set(xlim=(-3,3), ylim=(0,1.8), xlabel=f'z{j+1}', ylabel='Density')
        axes[j].set_xticks([-2,-1,0,1,2])
    fig.tight_layout(); fig.savefig(OUT/'gaussian-exercise.svg', facecolor='#f5f1e8'); plt.close(fig)
    manifest = {'dataset': 'Fashion-MNIST', 'split_seed': 41, 'train_size': 50000,
                'validation_size': 10000, 'training_seed': 113, 'draw_seed': 307,
                'epochs': 10, 'torch': torch.__version__, 'history': history,
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'selection': 'first validation item of classes 0, 1, 9; first 8 draws each',
                'display': 'decoder mean pixels, no pixel draws; separate section-2 training run',
                'records': records, 'prior_codes': prior.tolist()}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(records, indent=2), flush=True)

if __name__ == '__main__':
    main()
