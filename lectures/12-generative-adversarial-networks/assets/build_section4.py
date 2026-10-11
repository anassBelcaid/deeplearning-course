"""One verified alternating update on the section 2 MLPs; no quality claim.
Run with course PyTorch environment and COURSE_DATA_DIR pointing to FashionMNIST.
"""
from pathlib import Path
import os,json,html
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/section4';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(1204)
ns={'nn':nn,'torch':torch}
source=(ROOT/'gan_modules.py').read_text()
for key in ['GENERATOR','DISCRIMINATOR']:
    exec(source.split('# BEGIN '+key+'\n')[1].split('# END '+key)[0],ns)
G,D=ns['G'],ns['D'];latent_dim=16
bce=nn.BCEWithLogitsLoss()
opt_G=torch.optim.Adam(G.parameters(),lr=.001)
opt_D=torch.optim.Adam(D.parameters(),lr=.001)
ds=FashionMNIST(os.environ.get('COURSE_DATA_DIR','data'),train=True,download=False)
ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(41))[:4]
real=ds.data[ids].float().unsqueeze(1)/255
z=torch.randn(4,latent_dim)
def snapshot(model):return [p.detach().clone() for p in model.parameters()]
def changed(before,model):return sum(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
def grad_count(model):return sum(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters())
g0,d0=snapshot(G),snapshot(D)
# BEGIN D_STEP
D.requires_grad_(True)
opt_D.zero_grad(set_to_none=True)
fake = G(z)
real_logits = D(real)
fake_logits = D(fake.detach())
loss_D = (
    bce(real_logits, torch.ones_like(real_logits))
    + bce(fake_logits, torch.zeros_like(fake_logits))
)
loss_D.backward()
opt_D.step()
# END D_STEP
D_turn=dict(G_changed=changed(g0,G),D_changed=changed(d0,D),G_grad_tensors=grad_count(G),D_grad_tensors=grad_count(D),loss=loss_D.item())
assert D_turn['G_changed']==0 and D_turn['G_grad_tensors']==0 and D_turn['D_changed']==6
# Clear previous D gradients to audit this pass independently; freezing alone does not erase old .grad.
opt_D.zero_grad(set_to_none=True)
g1,d1=snapshot(G),snapshot(D)
# BEGIN G_STEP
D.requires_grad_(False)
opt_G.zero_grad(set_to_none=True)
fake = G(z)
fake_logits = D(fake)
loss_G = bce(
    fake_logits, torch.ones_like(fake_logits)
)
loss_G.backward()
opt_G.step()
D.requires_grad_(True)
# END G_STEP
G_turn=dict(G_changed=changed(g1,G),D_changed=changed(d1,D),G_grad_tensors=grad_count(G),D_grad_tensors=grad_count(D),loss=loss_G.item())
assert G_turn['G_changed']==6 and G_turn['D_changed']==0 and G_turn['G_grad_tensors']==6 and G_turn['D_grad_tensors']==0
# Diagnose the tempting but incorrect no_grad replacement.
D.requires_grad_(False);opt_G.zero_grad(set_to_none=True)
fake=G(z)
with torch.no_grad():bad_logits=D(fake)
bad_loss=bce(bad_logits,torch.ones_like(bad_logits))
assert not bad_loss.requires_grad
try:bad_loss.backward()
except RuntimeError as err:error=str(err)
else:raise AssertionError('Disconnected loss must not support backward')
assert all(p.grad is None for p in G.parameters())
D.requires_grad_(True)
record=dict(seed=1204,real_dataset_ids=ids.tolist(),batch_size=4,optimizer='Adam, lr=0.001; independent optimizer state',D_turn=D_turn,G_turn=G_turn,bug=dict(loss_requires_grad=bad_loss.requires_grad,G_grad_tensors=grad_count(G),error=error),parameter_tensors_per_model=6,purpose='One alternating update verifies ownership and gradient flow, not image quality',torch=str(torch.__version__))
(OUT/'update-checks.json').write_text(json.dumps(record,indent=2)+'\n')
# Editors display the exact executed source blocks; reveal backward/step together.
source=Path(__file__).read_text()
for key,label in [('D_STEP','DISCRIMINATOR TURN'),('G_STEP','GENERATOR TURN')]:
    code=source.split('# BEGIN '+key+'\n')[1].split('# END '+key)[0].strip()
    line='loss_D.backward()' if key=='D_STEP' else 'loss_G.backward()'
    pre,post=code.split(line)
    markup=html.escape(pre)+'<span class="fragment" data-fragment-index="0">'+html.escape(line+post)+'</span>'
    (ROOT/(key.lower()+'-code.html')).write_text('<div class="light-code-editor"><div class="editor-bar"><span>PYTORCH · '+label+'</span><strong>one optimizer step</strong></div><pre><code>'+markup+'</code></pre></div>')
print(json.dumps(record,indent=2))
