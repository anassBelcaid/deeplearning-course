"""Small DCGAN-inspired Fashion-MNIST run for section 5. CPU, no render-time training.
COURSE_DATA_DIR=/path/to/data python conv_gan.py
"""
from pathlib import Path
import os,json,time,hashlib
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/section5'
# BEGIN G
G = nn.Sequential(
    nn.ConvTranspose2d(16, 64, 7, 1, 0, bias=False),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
    nn.BatchNorm2d(32), nn.ReLU(),
    nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh(),
)
# END G
# BEGIN D
D = nn.Sequential(
    nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2),
    nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
    nn.Conv2d(64, 1, 7, 1, 0),
    nn.Flatten(1),
)
# END D

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(1205)
    def init(m):
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(m.weight,0,.02)
            if m.bias is not None:nn.init.zeros_(m.bias)
        elif isinstance(m,nn.BatchNorm2d):nn.init.normal_(m.weight,1,.02);nn.init.zeros_(m.bias)
    G.apply(init);D.apply(init)
    ds=FashionMNIST(os.environ.get('COURSE_DATA_DIR','data'),train=True,download=False)
    ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(41))[:12000]
    data=ds.data[ids].float().unsqueeze(1)/127.5-1
    fixed=torch.randn(6,16,1,1,generator=torch.Generator().manual_seed(1206))
    shuffle=torch.Generator().manual_seed(1207)
    bce=nn.BCEWithLogitsLoss();og=torch.optim.Adam(G.parameters(),lr=.0002,betas=(.5,.999));od=torch.optim.Adam(D.parameters(),lr=.0002,betas=(.5,.999))
    checks={};G.eval()
    for batch in [1,4,128]:
        with torch.no_grad():
            fake=G(torch.randn(batch,16,1,1));logits=D(fake)
        assert fake.shape==(batch,1,28,28) and logits.shape==(batch,1)
        assert fake.min()>=-1 and fake.max()<=1
    # Exact weighted overlap toy used in the transposed-convolution slide.
    toy=nn.ConvTranspose1d(1,1,3,stride=2,bias=False)
    with torch.no_grad():toy.weight.copy_(torch.tensor([[[1.,2.,1.]]]))
    toy_y=toy(torch.tensor([[[1.,2.]]])).flatten().tolist();assert toy_y==[1.,2.,3.,4.,2.]
    checks['batch_shapes_1_4_128']=True;checks['toy_transpose']=toy_y
    checkpoints=[];start=time.time()
    def capture(epoch):
        G.eval()
        with torch.no_grad():samples=G(fixed)
        for i,sample in enumerate(samples):
            im=((sample.squeeze().clamp(-1,1)+1)*127.5).round().byte().numpy()
            Image.fromarray(im).save(OUT/f'epoch-{epoch}-sample-{i}.png')
        checkpoints.append(epoch)
    capture(0)
    history=[]
    for epoch in range(1,61):
        losses=[]
        for batch_ids in torch.randperm(len(data),generator=shuffle).split(128):
            real=data[batch_ids];z=torch.randn(len(real),16,1,1)
            # No D BatchNorm: no discriminator buffers to freeze. G stats update only on G's turn.
            D.requires_grad_(True);od.zero_grad(set_to_none=True);G.eval()
            with torch.no_grad():fake=G(z)
            r,f=D(real),D(fake)
            ld=bce(r,torch.ones_like(r))+bce(f,torch.zeros_like(f));ld.backward();od.step()
            D.requires_grad_(False);og.zero_grad(set_to_none=True);G.train()
            f=D(G(z));lg=bce(f,torch.ones_like(f));lg.backward();og.step()
            D.requires_grad_(True);losses.append((ld.item(),lg.item()))
        history.append(dict(epoch=epoch,loss_D=sum(x[0] for x in losses)/len(losses),loss_G=sum(x[1] for x in losses)/len(losses)))
        if epoch in [1,5,20,60]:capture(epoch)
        print(f'epoch {epoch}/60, elapsed {time.time()-start:.1f}s',flush=True)
    torch.save({'G':G.state_dict(),'D':D.state_dict(),'fixed_z':fixed},OUT/'final-checkpoint.pt')
    record=dict(torch=str(torch.__version__),device='cpu',seed=1205,fixed_z_seed=1206,shuffle_seed=1207,split_seed=41,train_count=12000,epochs=60,batch_size=128,steps_per_epoch=94,checkpoints=checkpoints,optimizer='Adam, lr=0.0002, betas=(0.5,0.999)',real_range=[-1,1],sample_selection='First six fixed draws; no ranking or removal',batchnorm_policy='G train on G turn; G eval on D turn and snapshots; D has no BatchNorm',checks=checks,history=history,elapsed_seconds=time.time()-start,checkpoint_sha256=hashlib.sha256((OUT/'final-checkpoint.pt').read_bytes()).hexdigest(),limitations='Single small run; no MLP baseline or quality/coverage metric; not a reproduction of the DCGAN paper')
    (OUT/'run.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':run()
