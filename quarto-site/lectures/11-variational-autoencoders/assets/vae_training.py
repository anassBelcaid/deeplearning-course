"""Runnable Chapter 11 VAE: model, loss reductions, and optimizer step.

Examples (torch and torchvision required; dataset must already be available):
  python vae_training.py --data /path/to/data --epochs 10 --output /tmp/course-vae.pt
  python vae_training.py --data /path/to/data --check

The lecture site never executes training during rendering.
"""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

class GaussianEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())
        self.mu = nn.Linear(128, 2)
        self.logvar = nn.Linear(128, 2)

    def forward(self, x):
        h = self.features(x)
        return self.mu(h), self.logvar(h)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GaussianEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        sigma = (0.5 * logvar).exp()
        z = mu + sigma * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    pixel_cost = F.binary_cross_entropy(
        x_hat, x, reduction='none')
    rec = pixel_cost.flatten(1).sum(1)
    kl = 0.5 * (
        mu.square() + logvar.exp() - 1 - logvar
    ).sum(1)
    loss = (rec + kl).mean()
    return loss, rec.mean(), kl.mean()

def train_step(model, optimizer, x):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x_hat, mu, logvar = model(x)
    loss, rec, kl = vae_loss(x_hat, x, mu, logvar)
    loss.backward()
    optimizer.step()
    return loss.item(), rec.item(), kl.item()

def check(model, x):
    """Meaningful checks: reduction semantics, gradient paths, and parameter updates."""
    model.train()
    torch.manual_seed(811)
    x_hat, mu, lv = model(x)
    loss,rec,kl=vae_loss(x_hat,x,mu,lv)
    assert x_hat.shape==x.shape and mu.shape==lv.shape==(len(x),2)
    assert loss.ndim==0 and torch.isfinite(loss)
    duplicated=vae_loss(*[torch.cat([t,t]) for t in [x_hat,x,mu,lv]])[0]
    torch.testing.assert_close(loss,duplicated)
    expected=F.binary_cross_entropy(x_hat,x,reduction='sum')/len(x)
    torch.testing.assert_close(rec,expected)
    # Reconstruction alone must reach both heads, not merely KL.
    grads=torch.autograd.grad(rec,[model.encoder.mu.weight,model.encoder.logvar.weight],retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    changed={k:bool((v.detach()!=before[k]).any()) for k,v in model.named_parameters()}
    assert all(changed.values())
    # eval/no_grad do not suppress the explicit random draw; decoding mu is deterministic here.
    model.eval()
    with torch.no_grad():
        first=model(x)[0];second=model(x)[0]
        assert not torch.equal(first,second)
        m,_=model.encoder(x)
        torch.testing.assert_close(model.decoder(m),model.decoder(m),rtol=0,atol=0)
    return {'batch_size':len(x),'input_shape':list(x.shape),'mean_shape':list(mu.shape),
            'output_shape':list(x_hat.shape),'pre_update_loss':loss.item(),
            'pre_update_rec':rec.item(),'pre_update_kl':kl.item(),
            'rec_head_gradient_norms':[g.norm().item() for g in grads],
            'all_parameters_updated':all(changed.values()),'duplicated_batch_same_loss':True,
            'eval_forward_remains_stochastic':True,'torch':torch.__version__,
            'seed':113,'noise_seed':811,'selection':'first 32 training-split images; one update, no quality claim'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='/tmp/fashion-mnist')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--output',default='/tmp/course-vae.pt')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 and not args.check:parser.error('--epochs must be positive')
    torch.set_num_threads(4);torch.manual_seed(113)
    ds=FashionMNIST(args.data,train=True,download=False,transform=ToTensor())
    perm=torch.randperm(len(ds),generator=torch.Generator().manual_seed(41))
    model=VAE()
    if args.check:
        x=torch.stack([ds[int(i)][0] for i in perm[:32]])
        result=check(model,x)
        out=Path(__file__).resolve().parent/'evidence'/'training'
        out.mkdir(parents=True,exist_ok=True)
        (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    loader=DataLoader(Subset(ds,perm[:50000]),batch_size=256,shuffle=True,
                      generator=torch.Generator().manual_seed(41))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]
    for epoch in range(args.epochs):
        totals=torch.zeros(3)
        for x,_ in loader:
            metrics=train_step(model,optimizer,x)
            totals+=torch.tensor(metrics)*len(x)
        means=(totals/50000).tolist();history.append(means)
        print(f'Epoch {epoch+1}: loss={means[0]:.3f} rec={means[1]:.3f} kl={means[2]:.3f}',flush=True)
    torch.save({'model':model.state_dict(),'history':history,
                'config':{'d':2,'split_seed':41,'seed':113,'epochs':args.epochs,
                          'objective':'pixel-summed soft-target BCE + latent-summed KL; batch mean',
                          'torch':torch.__version__}},args.output)

if __name__=='__main__': main()
