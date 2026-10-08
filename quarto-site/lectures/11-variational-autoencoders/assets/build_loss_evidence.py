"""Controlled section-3 loss ablation. Rendering never trains models.

Uses section 2's beta=1 checkpoint and trains beta=0 with its exact protocol.
First validation items and fixed draws are retained, without quality selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_generation_evidence import VAE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence' / 'loss'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/tmp/fashion-mnist')
    p.add_argument('--vae-checkpoint',default='/tmp/vae-section2.pt')
    p.add_argument('--unregularized-checkpoint',default='/tmp/vae-section3-beta0.pt')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    ds=FashionMNIST(args.data,train=True,download=False,transform=ToTensor())
    perm=torch.randperm(60000,generator=torch.Generator().manual_seed(41))
    torch.manual_seed(113)
    unregularized=VAE()
    path=Path(args.unregularized_checkpoint)
    history=[]
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True)
        unregularized.load_state_dict(state['model']);history=state['history']
    else:
        batches=DataLoader(Subset(ds,perm[:50000]),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(41))
        opt=torch.optim.Adam(unregularized.parameters(),lr=.001)
        for epoch in range(10):
            total=0.
            for x,_ in batches:
                opt.zero_grad();decoded,mu,lv=unregularized(x)
                rec=nn.functional.binary_cross_entropy(decoded,x,reduction='sum')/len(x)
                # Preserve the same forward computation and random draws as beta=1.
                kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean()
                loss=rec+0*kl
                loss.backward();opt.step();total+=loss.item()*len(x)
            history.append(total/50000)
            print(f'Beta 0 epoch {epoch+1}: {history[-1]:.4f}',flush=True)
        torch.save({'model':unregularized.state_dict(),'history':history},path)
    regularized=VAE();regularized.load_state_dict(torch.load(args.vae_checkpoint,map_location='cpu',weights_only=True)['model'])
    val_indices=perm[50000:]
    x=torch.stack([ds[int(i)][0] for i in val_indices[:1024]])
    eps=torch.randn(8,1024,2,generator=torch.Generator().manual_seed(509))
    prior=torch.randn(8,2,generator=torch.Generator().manual_seed(601))
    manifest={'split_seed':41,'initialization_seed':113,'posterior_draw_seed':509,'prior_draw_seed':601,
              'train_size':50000,'epochs':10,'batch_size':256,'optimizer':'Adam lr=0.001',
              'evaluation':'first 1024 validation images, 8 posterior draws each; pixel sums, image means',
              'selection':'first 4 validation images and first 8 prior draws; no selection by quality',
              'validation_indices':val_indices[:1024].tolist(),'prior_codes':prior.tolist(),
              'torch':torch.__version__,'models':{}}
    for name,model,ck in [('recon',unregularized,path),('vae',regularized,Path(args.vae_checkpoint))]:
        model.eval()
        with torch.inference_mode():
            h=model.encoder(x);mu,lv=model.mu(h),model.logvar(h)
            z=mu[None]+(.5*lv).exp()[None]*eps
            dec=model.decoder(z.reshape(-1,2)).reshape(8,1024,1,28,28)
            rec=nn.functional.binary_cross_entropy(dec,x[None].expand_as(dec),reduction='none').sum((2,3,4))
            kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1)
            generated=model.decoder(prior)
        manifest['models'][name]={'reconstruction':rec.mean().item(),'kl':kl.mean().item(),
            'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
            'first_image_draw_costs':rec[:,0].tolist(),'first_image_kl':kl[0].item(),
            'first_image_mu':mu[0].tolist(),'first_image_sigma':(.5*lv[0]).exp().tolist()}
        # Three rows let the instructor compare reconstruction and prior generation.
        fig,axes=plt.subplots(3,4,figsize=(12,6.5))
        for row,items in enumerate([x[:4],dec[0,:4],generated[:4]]):
            for col,img in enumerate(items):
                axes[row,col].imshow(img.squeeze(),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                for sp in axes[row,col].spines.values():sp.set_visible(False)
            axes[row,0].set_ylabel(['Input x','Posterior draw','Prior draw'][row],fontsize=18,labelpad=18)
        fig.subplots_adjust(left=.13,right=.99,bottom=.015,top=.99,hspace=.13,wspace=.08)
        fig.savefig(OUT/f'{name}-gallery.png',dpi=140,facecolor='#f5f1e8');plt.close(fig)
        for i in range(8):
            plt.imsave(OUT/f'{name}-prior-{i}.png',generated[i].squeeze().numpy(),cmap='gray',vmin=0,vmax=1)
            if name=='vae':plt.imsave(OUT/f'reconstruction-{i}.png',dec[i,0].squeeze().numpy(),cmap='gray',vmin=0,vmax=1)
    plt.imsave(OUT/'target.png',x[0].squeeze().numpy(),cmap='gray',vmin=0,vmax=1)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['models'],indent=2),flush=True)

if __name__=='__main__': main()
