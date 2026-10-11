"""Section 1 evidence: a frozen course VAE, unpaired targets, and a learned source classifier.
Run explicitly; Quarto never executes this script. Requires torch, torchvision, numpy, Pillow.
COURSE_DATA_DIR=/path/to/data python build_section1.py
"""
import os, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
ROOT=Path(__file__).resolve().parent
SITE=ROOT.parents[2]
LAB=SITE/'labs/11-variational-autoencoders'
OUT=ROOT/'evidence/section1';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
# Reconstruct the exact saved lab decoder without importing training or plotting infrastructure.
class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),
                         nn.Linear(128,784),nn.Sigmoid(),nn.Unflatten(1,(1,28,28)))
checkpoint=LAB/'assets/checkpoints/beta1.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
generator=Decoder();generator.load_state_dict({k.removeprefix('decoder.'):v for k,v in state.items() if k.startswith('decoder.')})
generator.eval().requires_grad_(False)
ds=FashionMNIST(os.environ.get('COURSE_DATA_DIR','data'),train=True,download=False)
perm=torch.randperm(len(ds),generator=torch.Generator().manual_seed(41))
real_train=ds.data[perm[:6000]].float().unsqueeze(1)/255
val_ids=perm[12000:13000];real_val=ds.data[val_ids].float().unsqueeze(1)/255
with torch.inference_mode():
    z_train=torch.randn(6000,2,generator=torch.Generator().manual_seed(301))
    z_val=torch.randn(1000,2,generator=torch.Generator().manual_seed(302))
    fake_train=generator(z_train);fake_val=generator(z_val)
x_train=torch.cat([real_train,fake_train]);y_train=torch.cat([torch.ones(6000),torch.zeros(6000)])
x_val=torch.cat([real_val,fake_val]);y_val=torch.cat([torch.ones(1000),torch.zeros(1000)])
# The first three examples of each source, interleaved in a predeclared order.
gallery_ids=[0,1000,1001,1,1002,2]
class Judge(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Flatten(),nn.Linear(784,128),nn.LeakyReLU(.2),
                         nn.Linear(128,64),nn.LeakyReLU(.2),nn.Linear(64,1))
torch.manual_seed(303);judge=Judge();optimizer=torch.optim.Adam(judge.parameters(),lr=.001)
history=[]
def record(epoch):
    judge.eval()
    with torch.inference_mode():
        scores=judge(x_val).flatten().sigmoid()
        correct=(scores>=.5)==y_val.bool()
    row=dict(epoch=epoch,accuracy=correct.float().mean().item(),
             real_accuracy=correct[:1000].float().mean().item(),
             generated_accuracy=correct[1000:].float().mean().item(),
             gallery_scores=scores[gallery_ids].tolist())
    history.append(row);print(row,flush=True)
record(0)
rng=torch.Generator().manual_seed(304)
for epoch in range(1,9):
    judge.train()
    for ids in torch.randperm(len(x_train),generator=rng).split(256):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(judge(x_train[ids]).flatten(),y_train[ids])
        loss.backward();optimizer.step()
    if epoch in [1,4,8]:record(epoch)
def save(tensor,name):
    array=(tensor.detach().squeeze().clamp(0,1).numpy()*255).round().astype(np.uint8)
    Image.fromarray(array).save(OUT/name)
gallery=[]
for pos,idx in enumerate(gallery_ids):
    file=f'gallery-{pos}.png';save(x_val[idx],file)
    gallery.append(dict(letter=chr(65+pos),file=file,source='Dataset' if idx<1000 else 'VAE output',
                        validation_offset=idx if idx<1000 else idx-1000,
                        dataset_id=int(val_ids[idx]) if idx<1000 else None,
                        z=z_val[idx-1000].tolist() if idx>=1000 else None))
# A fixed prior draw, with three unrelated observed images as possible targets.
candidate=fake_val[0];save(candidate,'candidate.png')
targets=[]
for letter,label in zip('ABC',[0,9,8]):
    idx=int((ds.targets[val_ids]==label).nonzero()[0])
    target=real_val[idx];save(target,f'target-{letter}.png')
    diff=(candidate-target).abs().squeeze().numpy()
    # Explicit linear scale: black=0, orange=1. Same scale for every heatmap.
    color=np.array([239,123,67]);rgb=(diff[:,:,None]*color).round().astype(np.uint8)
    Image.fromarray(rgb).save(OUT/f'difference-{letter}.png')
    targets.append(dict(letter=letter,file=f'target-{letter}.png',difference=f'difference-{letter}.png',
        mse=F.mse_loss(candidate,target).item(),dataset_id=int(val_ids[idx]),class_label=label))
manifest=dict(dataset='Fashion-MNIST',split_seed=41,train_real=6000,train_generated=6000,
    validation_real=1000,validation_generated=1000,real_train_positions=[0,5999],
    real_validation_positions=[12000,12999],source_classification_threshold=.5,
    judge_seed=303,shuffle_seed=304,train_noise_seed=301,validation_noise_seed=302,
    generator='Frozen VAE lab beta=1 decoder; no adversarial generator update in this experiment',
    generator_checkpoint=str(checkpoint.relative_to(SITE)),
    generator_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    judge_architecture='784-128-64-1 MLP, LeakyReLU(0.2), BCEWithLogitsLoss',
    optimizer='Adam, lr=0.001, batch=256, 8 epochs',torch=str(torch.__version__),device='cpu',
    gallery_selection='First 3 real and first 3 generated validation examples, fixed order R G G R G R; not selected by classifier scores',
    gallery=gallery,history=history,candidate_z=z_val[0].tolist(),targets=targets,
    limitations='A fixed-source discrimination experiment, not GAN training, calibrated realism estimation, or a comparison of generative families.')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Small self-contained HTML fragments, kept in sync with measured outputs.
rows=[]
for item in gallery:
    rows.append(f'<figure><figcaption>{item["letter"]}</figcaption><img src="assets/evidence/section1/{item["file"]}" alt="Unlabeled garment {item["letter"]} for the source-identification exercise."><div class="fragment gan-origin" data-fragment-index="0">{item["source"]}</div></figure>')
(ROOT/'blind-gallery.html').write_text('<div class="gan-blind-gallery">'+''.join(rows)+'</div>\n')
rows=[]
for item in gallery:
    rows.append(f'<figure><figcaption>{item["letter"]} · {item["source"]}</figcaption><img src="assets/evidence/section1/{item["file"]}" alt="Same garment {item["letter"]} from the earlier exercise; source: {item["source"]}."><div class="gan-score-track"><i id="judge-bar-{item["letter"]}"></i></div><strong id="judge-score-{item["letter"]}">—</strong></figure>')
(ROOT/'judge-gallery.html').write_text('<div class="gan-judge-gallery">'+''.join(rows)+'</div>\n')
print('Built section 1 evidence in',OUT)
