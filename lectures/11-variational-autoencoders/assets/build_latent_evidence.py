"""Section 6 evidence from the unchanged section-2 checkpoint; no training.
Run with --data FashionMNIST-parent. Outputs and selections recorded in manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from PIL import Image
from vae_training import VAE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence' / 'latent'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', default='/tmp/vae-section2.pt')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = VAE().eval()
    checkpoint = Path(args.checkpoint)
    old = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    renamed = {('encoder.features.' + k[8:] if k.startswith('encoder.') else
                'encoder.' + k if k.startswith(('mu.', 'logvar.')) else k): v for k, v in old.items()}
    model.load_state_dict(renamed)
    ds = FashionMNIST(args.data, train=True, download=False, transform=ToTensor())
    val = torch.randperm(60000, generator=torch.Generator().manual_seed(41))[50000:]
    selected = [next(int(i) for i in val if int(ds.targets[i]) == c) for c in [0, 1, 9]]
    inputs = torch.stack([ds[i][0] for i in selected])
    def save(name, x):
        Image.fromarray((x.squeeze().clamp(0, 1).numpy()*255).round().astype('uint8')).save(OUT/name)
    with torch.inference_mode():
        endpoints, _ = model.encoder(inputs)
        paths = []
        for key, a, b, label in [('shirt-trouser', 0, 1, 'T-shirt → trouser'),
                                  ('shirt-boot', 0, 2, 'T-shirt → ankle boot')]:
            t = torch.linspace(0, 1, 21)
            z = (1-t[:, None])*endpoints[a] + t[:, None]*endpoints[b]
            outputs = model.decoder(z)
            for j, output in enumerate(outputs): save(f'{key}-{j}.png', output)
            paths.append(dict(key=key, label=label, a=a, b=b, codes=z.tolist()))
        for j, x in enumerate(inputs): save(f'input-{j}.png', x)
        prior = torch.randn(16, 2, generator=torch.Generator().manual_seed(619))
        for j, output in enumerate(model.decoder(prior)): save(f'prior-{j}.png', output)
        x = torch.stack([ds[int(i)][0] for i in val[:1024]])
        mu, lv = model.encoder(x)
        kl = .5*(mu.square()+lv.exp()-1-lv).sum(1)
        noise = torch.randn(8, len(x), 2, generator=torch.Generator().manual_seed(620))
        z = mu[None] + (.5*lv).exp()[None]*noise
        # Deterministic cyclic reassignment breaks the image/code pairing, retaining every code.
        right = model.decoder(z.reshape(-1,2)).reshape(8,len(x),1,28,28)
        wrong = right.roll(1, dims=1)
        target = x[None].expand_as(right)
        costs = [F.binary_cross_entropy(y, target, reduction='none').flatten(2).sum(2).mean().item() for y in [right, wrong]]
        for j in range(4):
            save(f'probe-input-{j}.png', x[j]); save(f'probe-right-{j}.png', right[0,j]); save(f'probe-wrong-{j}.png', wrong[0,j])
        # Coordinates for the map; labels used only for display, never for training.
        points = [dict(mu=m.tolist(), label=int(ds.targets[int(i)])) for m,i in zip(mu,val[:1024])]
    manifest = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        torch=torch.__version__, split_seed=41, selected_indices=selected,
        paths=paths, endpoints=endpoints.tolist(), points=points, prior_codes=prior.tolist(),
        prior_seed=619, probe_noise_seed=620, probe_indices=val[:1024].tolist(), draws_per_image=8,
        reassignment='cyclic roll by +1 along image axis, same decoded samples',
        reconstruction_cost=dict(paired=costs[0], reassigned=costs[1]), mean_kl=kl.mean().item(),
        display='Decoder means; grayscale-target BCE surrogate, pixel sum then draw/image mean; no training or browser inference')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    assert costs[1]>costs[0], 'Inspect unexpected reassignment result before authoring interpretation'
    print(json.dumps({k:manifest[k] for k in ['selected_indices','reconstruction_cost','mean_kl','checkpoint_sha256']},indent=2))

if __name__ == '__main__': main()
