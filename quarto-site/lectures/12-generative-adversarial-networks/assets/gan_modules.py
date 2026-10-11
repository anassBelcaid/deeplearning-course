"""Executable section 2 module contracts. No training is performed.
Run: COURSE_DATA_DIR=/path/to/data python gan_modules.py
Requires torch, torchvision, numpy, and Pillow. Quarto never runs this file.
"""
from pathlib import Path
import os, json, hashlib
import torch
from torch import nn

# BEGIN GENERATOR
latent_dim = 16
G = nn.Sequential(
    nn.Linear(latent_dim, 128), nn.ReLU(),
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 784), nn.Sigmoid(),
    nn.Unflatten(1, (1, 28, 28)),
)
# END GENERATOR

# BEGIN DISCRIMINATOR
D = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 128), nn.LeakyReLU(0.2),
    nn.Linear(128, 1),
)
# END DISCRIMINATOR

def build_evidence():
    from PIL import Image
    import numpy as np
    from torchvision.datasets import FashionMNIST
    root=Path(__file__).resolve().parent
    out=root/'evidence/section2';out.mkdir(parents=True,exist_ok=True)
    # Initialize the exact displayed module definitions with a recorded seed.
    source=Path(__file__).read_text()
    torch.set_num_threads(4);torch.manual_seed(1202)
    ns={'torch':torch,'nn':nn}
    for name in ['GENERATOR','DISCRIMINATOR']:
        exec(source.split('# BEGIN '+name+'\n')[1].split('# END '+name)[0],ns)
    G,D=ns['G'],ns['D']
    ds=FashionMNIST(os.environ.get('COURSE_DATA_DIR','data'),train=True,download=False)
    perm=torch.randperm(len(ds),generator=torch.Generator().manual_seed(41))
    ids=perm[12000:12004]
    real=ds.data[ids].float().unsqueeze(1)/255
    torch.manual_seed(1203)
    # BEGIN FORWARD_G
    z = torch.randn(4, latent_dim)
    fake = G(z)
    print(fake.shape)
    # END FORWARD_G
    # BEGIN FORWARD_D
    real_logits = D(real)
    fake_logits = D(fake)
    print(fake_logits.shape)
    display_scores = fake_logits.sigmoid()
    # END FORWARD_D
    assert fake.shape==(4,1,28,28)
    assert real_logits.shape==fake_logits.shape==(4,1)
    assert fake.min()>=0 and fake.max()<=1
    assert torch.isfinite(fake).all() and torch.isfinite(fake_logits).all()
    assert not set(map(id,G.parameters())) & set(map(id,D.parameters()))
    # Both sources pass through this same D without copying or changing its parameters.
    before={k:v.detach().clone() for k,v in D.state_dict().items()}
    torch.testing.assert_close(D(torch.cat([real,fake])),torch.cat([real_logits,fake_logits]))
    assert all(torch.equal(v,D.state_dict()[k]) for k,v in before.items())
    # G's forward computation is independent of D; no training claim or optimizer step.
    with torch.no_grad():
        first=G(z).clone()
        other=G(torch.randn(4,latent_dim))
        torch.testing.assert_close(first,G(z),rtol=0,atol=0)
        assert not torch.equal(first,other)
    for batch in [1,4,7]:
        assert D(G(torch.randn(batch,latent_dim))).shape==(batch,1)
    for i in range(4):
        for name,batch in [('real',real),('fake',fake)]:
            im=(batch[i].detach().squeeze().numpy()*255).round().astype(np.uint8)
            Image.fromarray(im).save(out/f'{name}-{i}.png')
    record=dict(seed=1202,noise_seed=1203,torch=str(torch.__version__),device='cpu',trained=False,
        latent_dim=16,batch_size=4,real_dataset_ids=ids.tolist(),z=z.tolist(),
        fake_shape=list(fake.shape),real_logits=real_logits.flatten().tolist(),
        fake_logits=fake_logits.flatten().tolist(),display_scores=display_scores.flatten().tolist(),
        fake_range=[fake.min().item(),fake.max().item()],
        checks={'same_discriminator_for_both_sources':True,'parameters_unchanged':True,
                'independent_parameter_sets':True,'generation_needs_only_G':True,
                'batch_sizes_1_4_7':True},
        generator_parameters=sum(p.numel() for p in G.parameters()),
        discriminator_parameters=sum(p.numel() for p in D.parameters()),
        purpose='Untrained executable tensor-contract demonstration; not GAN quality evidence')
    (out/'contracts.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':build_evidence()
