"""Reproduce Section 1 evidence; model checkpoints stay outside the website.

Run with torch, torchvision, numpy, matplotlib installed:
python build_generation_evidence.py --data /tmp/fashion-mnist
Reuses Project 3's d=2 checkpoint, or recreates it with its documented protocol.
VAE preview is not a controlled performance comparison with that MSE autoencoder.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / '10-autoencoders-latent-representations/assets'))
from build_latent_sweep import Autoencoder

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())
        self.mu = nn.Linear(128,2)
        self.logvar = nn.Linear(128,2)
        self.decoder = Autoencoder(2).decoder
    def forward(self, x):
        h = self.encoder(x)
        mu, lv = self.mu(h), self.logvar(h)
        return self.decoder(mu + (lv * .5).exp() * torch.randn_like(mu)), mu, lv

def gallery(rows, labels, name):
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(13, 2.1*len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c, x in enumerate(row):
            axes[r,c].imshow(x.squeeze(), cmap='gray', vmin=0,vmax=1, interpolation='nearest')
            axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
            for sp in axes[r,c].spines.values(): sp.set_visible(False)
        axes[r,0].set_ylabel(labels[r], fontsize=14, labelpad=14)
    fig.tight_layout(pad=.8)
    fig.savefig(ROOT/'evidence'/name, dpi=160, facecolor='#f4f0e7')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='/tmp/fashion-mnist')
    parser.add_argument('--checkpoints',default='/tmp/latent-laboratory-checkpoints')
    args=parser.parse_args()
    out=ROOT/'evidence'; out.mkdir(exist_ok=True)
    ck=Path(args.checkpoints); ck.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(41)
    ds=FashionMNIST(args.data,train=True,download=True,transform=ToTensor())
    perm=torch.randperm(60000,generator=torch.Generator().manual_seed(41))
    train=Subset(ds,perm[:50000]); val=Subset(ds,perm[50000:])
    def loader():
        return DataLoader(train,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(41))
    ae=Autoencoder(2)
    if (ck/'ae-d2.pt').exists():
        state=torch.load(ck/'ae-d2.pt',map_location='cpu',weights_only=True)
        ae.load_state_dict(state)
    else:
        torch.manual_seed(41); ae=Autoencoder(2); opt=torch.optim.Adam(ae.parameters(),lr=.001)
        batches=loader()
        for epoch in range(6):
            for x,_ in batches:
                opt.zero_grad(); loss=(ae(x)-x).square().mean(); loss.backward(); opt.step()
        torch.save(ae.state_dict(),ck/'ae-d2.pt')
    torch.manual_seed(113); vae=VAE(); history=[]
    path=ck/'vae-section1-d2.pt'
    if path.exists():
        state=torch.load(path,weights_only=True); vae.load_state_dict(state['model']); history=state['history']
    else:
        opt=torch.optim.Adam(vae.parameters(),lr=.001); batches=loader()
        for epoch in range(10):
            total=0
            for x,_ in batches:
                opt.zero_grad(); decoded,mu,lv=vae(x)
                # BCE with grayscale targets is the common soft-target surrogate;
                # it is not an exact continuous-image likelihood.
                rec=nn.functional.binary_cross_entropy(decoded,x,reduction='sum')/len(x)
                kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean()
                loss=rec+kl; loss.backward(); opt.step(); total+=loss.item()*len(x)
            history.append(total/len(train)); print('VAE epoch',epoch+1,history[-1],flush=True)
        torch.save({'model':vae.state_dict(),'history':history},path)
    ae.eval(); vae.eval()
    x=torch.stack([val[i][0] for i in range(2000)])
    with torch.inference_mode():
        z=ae.encoder(x); recon=ae.decoder(z)
        torch.manual_seed(211)
        low,high=z.quantile(.01,dim=0),z.quantile(.99,dim=0)
        probes=low+torch.rand(16,2)*(high-low); arbitrary=ae.decoder(probes)
        prior=torch.randn(16,2); generated=vae.decoder(prior)
    gallery([x[:8],recon[:8]],['Input','AE output'],'reconstruction.png')
    torch.manual_seed(97)
    gallery([x[:8],torch.rand(8,1,28,28)],['Real data','Random pixels'],'pixels.png')
    gallery([arbitrary[:8],arbitrary[8:]],['Draws 1–8','Draws 9–16'],'ae-samples.png')
    gallery([generated[:8],generated[8:]],['Draws 1–8','Draws 9–16'],'vae-samples.png')
    gallery([generated[:8]],['Decoder mean'],'sample-strip.png')
    for i, img in enumerate(generated):
        plt.imsave(out/f'draw-{i:02}.png',img.squeeze().numpy(),cmap='gray',vmin=0,vmax=1)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(z[:,0],z[:,1],s=8,alpha=.22,color='#2878c8',label='Encoded validation images')
    ax.scatter(probes[:,0],probes[:,1],s=70,marker='x',color='#ef5b45',label='Uniform draws in quantile box')
    for i,p in enumerate(probes[:4]): ax.annotate(str(i+1),p,xytext=(6,6),textcoords='offset points',fontsize=12)
    ax.set(xlabel='Actual latent coordinate z₁',ylabel='Actual latent coordinate z₂')
    ax.legend(fontsize=11); fig.tight_layout(); fig.savefig(out/'ae-map.png',dpi=160,facecolor='#f4f0e7');plt.close(fig)
    # Exact pixel-space nearest neighbors among all 50k training images.
    all_train=ds.data[perm[:50000]].float().reshape(50000,-1)/255
    distances=torch.cdist(generated[:8].flatten(1),all_train)
    nearest=distances.argmin(1)
    gallery([generated[:8],all_train[nearest].reshape(-1,1,28,28)],['VAE mean','Nearest train'],'neighbors.png')
    np.savez(out/'coordinates.npz',encoded=z.numpy(),uniform=probes.numpy(),prior=prior.numpy())
    (out/'manifest.json').write_text(json.dumps({'dataset':'Fashion-MNIST','train_size':50000,'validation_size':10000,'map_size':2000,'split_seed':41,'ae_epochs':6,'vae_epochs':10,'vae_seed':113,'sample_seed':211,'d':2,'vae_objective':'sum-pixel soft-target BCE + KL per image; beta=1','vae_history':history,'prior_codes':prior.tolist(),'uniform_codes':probes.tolist(),'nearest_train_dataset_indices':perm[:50000][nearest].tolist(),'nearest_train_mse':(distances.min(1).values.square()/784).tolist(),'selection':'first 16 fixed-seed draws, no cherry-picking','display':'decoder means, not additional Bernoulli pixel samples','torch':torch.__version__},indent=2)+'\n')
    print('Evidence saved to',out,flush=True)

if __name__=='__main__': main()
