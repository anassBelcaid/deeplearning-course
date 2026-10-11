"""Evaluation evidence from the saved section 5 GAN. No training, no sample ranking."""
from pathlib import Path
import os,json,hashlib
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from PIL import Image
R=Path(__file__).resolve().parent;O=R/'evidence/section6';O.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
ns={'nn':nn};src=(R/'conv_gan.py').read_text()
for key in ['G','D']:exec(src.split('# BEGIN '+key+'\n')[1].split('# END '+key)[0],ns)
G,D=ns['G'],ns['D'];ck=torch.load(R/'evidence/section5/final-checkpoint.pt',map_location='cpu',weights_only=True)
G.load_state_dict(ck['G']);D.load_state_dict(ck['D']);G.eval();D.eval()
ds=FashionMNIST(os.environ.get('COURSE_DATA_DIR','data'),train=True,download=False)
perm=torch.randperm(len(ds),generator=torch.Generator().manual_seed(41))
train=ds.data[perm[:12000]].float().unsqueeze(1)/255
real=ds.data[perm[12000:12036]].float().unsqueeze(1)/255
z=torch.randn(36,16,1,1,generator=torch.Generator().manual_seed(1208))
with torch.no_grad():fake=(G(z)+1)/2
assert fake.shape==(36,1,28,28) and torch.isfinite(fake).all()
def save(im,path):Image.fromarray((im.squeeze().clamp(0,1)*255).round().byte().numpy()).save(path)
def grid(batch,path):
 canvas=Image.new('L',(4*112,3*112),0)
 for j,im in enumerate(batch):canvas.paste(Image.fromarray((im.squeeze().clamp(0,1)*255).round().byte().numpy()).resize((104,104),Image.Resampling.NEAREST),((j%4)*112+4,(j//4)*112+4))
 canvas.save(path)
for batch in range(3):
 grid(real[batch*12:(batch+1)*12],O/f'real-{batch}.png');grid(fake[batch*12:(batch+1)*12],O/f'generated-{batch}.png')
 grid(fake[batch*12].repeat(12,1,1,1),O/f'repeated-{batch}.png')
# The first three fresh outputs, with exact nearest neighbors in the 12,000-image training subset.
pairs=[]
for i in range(3):
 mse=(train-fake[i:i+1]).square().flatten(1).mean(1);j=int(mse.argmin())
 save(fake[i],O/f'query-{i}.png');save(train[j],O/f'nearest-{i}.png')
 pairs.append(dict(query_index=i,training_subset_index=j,dataset_id=int(perm[j]),mse=float(mse[j]),exact_float_match=bool(torch.equal(fake[i],train[j]))))
record=dict(generator='section5 final checkpoint, epoch60, eval mode',fresh_seed=1208,fresh_count=36,real_ids=perm[12000:12036].tolist(),selection='Three consecutive batches of 12; first three queries for nearest neighbors; no ranking or removal',repeated_control='First generated output of each batch repeated 12 times by the viewer data builder; not an observed model collapse',nearest_search='Exhaustive pixel MSE in [0,1] against all 12000 training images; original floating point generated outputs',nearest_pairs=pairs,limitations='Small galleries do not measure population coverage. Pixel nearest neighbors are a sanity check, not a proof against memorization.',checkpoint_sha256=hashlib.sha256((R/'evidence/section5/final-checkpoint.pt').read_bytes()).hexdigest())
(O/'evaluation.json').write_text(json.dumps(record,indent=2)+'\n')
# Reproducible SVG plot from all 60 recorded epoch means, on one shared y-axis.
h=json.loads((R/'evidence/section5/run.json').read_text())['history'];max_y=4
assert max(max(r['loss_D'],r['loss_G']) for r in h)<max_y
b='<svg viewBox="0 0 900 440" role="img" aria-label="Recorded discriminator and generator training losses over 60 epochs, using a shared zero to four vertical axis">'
for v in range(5):
 y=355-v*72;b+=f'<path d="M70 {y}H850" stroke="#d8dfe2"/><text x="52" y="{y+7}" text-anchor="end">{v}</text>'
for e in [1,20,40,60]:b+=f'<text x="{70+(e-1)*780/59}" y="395" text-anchor="middle">{e}</text>'
b+='<text x="460" y="435" text-anchor="middle">Epoch · mean of minibatch losses</text>'
for key,color in [('loss_D','#b26950'),('loss_G','#2878c8')]:
 points=' '.join(f'{70+(r["epoch"]-1)*780/59:.2f},{355-r[key]*72:.2f}' for r in h)
 b+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
b+='<line id="gan-loss-epoch-marker" x1="321.19" x2="321.19" y1="55" y2="360" stroke="#526575" stroke-dasharray="6 6"/>'
b+='</svg>'; (R/'training-loss-plot.html').write_text('<div class="gan-training-loss-plot">'+b+'</div>')
(O/'losses.json').write_text(json.dumps(h,indent=2)+'\n')
print(json.dumps(record,indent=2))
# Keep the measured-neighbor slide synchronized with its recorded search.
gallery='<div class="gan-neighbor-gallery">'
for i,pair in enumerate(pairs):
    gallery+=f'<figure><figcaption>Fresh draw {i+1}</figcaption><img src="assets/evidence/section6/query-{i}.png" alt="Fresh generated sample {i+1}"><div class="fragment" data-fragment-index="0"><span>Nearest training image</span><img src="assets/evidence/section6/nearest-{i}.png" alt="Pixel-MSE nearest training image for generated sample {i+1}"><small>MSE = {pair["mse"]:.4f}</small></div></figure>'
gallery+='</div>'; (R/'nearest-gallery.html').write_text(gallery)
