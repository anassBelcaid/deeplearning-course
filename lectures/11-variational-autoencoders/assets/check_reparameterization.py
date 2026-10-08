"""Reproduce section 4's autograd evidence and validate its numerical exercise."""
import json
from pathlib import Path
import torch

OUT=Path(__file__).resolve().parent/'evidence'/'reparameterization'
OUT.mkdir(parents=True,exist_ok=True)

def probe(mode):
    torch.manual_seed(7)
    mu=torch.tensor([1.],requires_grad=True)
    logvar=torch.tensor([0.],requires_grad=True)
    a=torch.tensor([2.],requires_grad=True)
    sigma=(.5*logvar).exp()
    if mode=='sample': z=torch.distributions.Normal(mu,sigma).sample()
    elif mode=='rsample': z=torch.distributions.Normal(mu,sigma).rsample()
    else:
        eps=torch.randn_like(mu)
        z=mu+sigma*eps
    loss=.5*(a*z-2).square().mean()
    loss.backward()
    return {'z':z.item(),'loss':loss.item(),'z_requires_grad':z.requires_grad,
            'mu_grad':None if mu.grad is None else mu.grad.item(),
            'logvar_grad':None if logvar.grad is None else logvar.grad.item(),
            'a_grad':a.grad.item()}

records={mode:probe(mode) for mode in ['sample','manual','rsample']}
assert records['sample']['mu_grad'] is None
assert records['sample']['logvar_grad'] is None
for mode in ['manual','rsample']:
    r=records[mode];s=records['sample']
    assert abs(r['z']-s['z'])<1e-7 and abs(r['loss']-s['loss'])<1e-7
    assert r['mu_grad'] is not None and r['logvar_grad'] is not None
    assert abs(r['a_grad']-s['a_grad'])<1e-7
# Independent derivative checks with a fixed epsilon and double precision.
for mu_value,sigma_value,eps in [(1.,.5,-1.),(-1.,.25,1.5),(2.,1.5,-.3)]:
    mu=torch.tensor(mu_value,dtype=torch.float64,requires_grad=True)
    sigma=torch.tensor(sigma_value,dtype=torch.float64,requires_grad=True)
    z=mu+sigma*eps
    loss=.5*(2*z-2).square()
    loss.backward()
    def f(m,s):return .5*(2*(m+s*eps)-2)**2
    h=1e-5
    assert abs(mu.grad.item()-(f(mu_value+h,sigma_value)-f(mu_value-h,sigma_value))/(2*h))<1e-8
    assert abs(sigma.grad.item()-(f(mu_value,sigma_value+h)-f(mu_value,sigma_value-h))/(2*h))<1e-8
# Exact worked example, including log variance parameterization.
mu=torch.tensor(1.,requires_grad=True)
lv=torch.tensor(.25).log().requires_grad_()
z=mu+(.5*lv).exp()*(-1.)
loss=.5*(2*z-2).square();loss.backward()
assert z.item()==.5 and loss.item()==.5
assert mu.grad.item()==-2 and lv.grad.item()==.5
noise=torch.randn(8,generator=torch.Generator().manual_seed(73)).tolist()
manifest={'torch':torch.__version__,'device':'cpu','probe_seed':7,'noise_seed':73,
          'noise':noise,'probes':records,'exercise':{'mu':1,'sigma':.5,'eps':-1,'z':.5,'loss':.5,'dmu':-2,'dsigma':2,'dlogvar':.5}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
print('PASS: detached vs pathwise gradients; same seeded forward values; finite differences; worked exercise.')
