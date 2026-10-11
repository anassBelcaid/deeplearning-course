"""Provided data, visualization, logging and report infrastructure.
No model, loss, optimizer step or trained checkpoint is supplied.
Images returned by the loader are RGB float tensors in [0,1].
"""
from pathlib import Path
import base64, csv, hashlib, html, json, platform, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

DATASET_HANDLE = 'splcher/animefacedataset'
DATASET_URL = 'https://www.kaggle.com/datasets/' + DATASET_HANDLE

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    return {'seed': seed, 'python': platform.python_version(), 'torch': str(torch.__version__),
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}

def rgb_image(path, size):
    """EXIF orientation, RGB conversion, centered square crop and bicubic resize."""
    with Image.open(path) as source:
        return ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'),
                            (size, size), method=Image.Resampling.BICUBIC)

def prepare_data(root, out, size=64, max_images=8000, seed=120, fractions=(.8,.1,.1)):
    """Audit a seeded subset; remove exact processed-image duplicates BEFORE splitting.
    Near duplicates/related characters can remain. Lists store paths, not copied tensors.
    Reuse the saved manifest for every experiment; don't rerun a new split per model.
    """
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if len(fractions)!=3 or any(v<=0 for v in fractions) or not np.isclose(sum(fractions),1):
        raise ValueError('Use three positive train/validation/test fractions summing to one.')
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    if not files: raise FileNotFoundError('No images found. Set DATA_ROOT to the extracted Kaggle image folder.')
    order=np.random.default_rng(seed).permutation(len(files))
    if max_images is not None: order=order[:max_images]
    kept=[];rejected=[];duplicates=[];seen={}
    for j in order:
        p=files[int(j)];rel=str(p.relative_to(root))
        try:
            with Image.open(p) as im: width,height=im.size;mode=im.mode
            image=rgb_image(p,size)
            digest=hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen:
                duplicates.append({'path':rel,'same_as':seen[digest]});continue
            seen[digest]=rel
            kept.append({'path':rel,'width':width,'height':height,'mode':mode,'pixel_sha256':digest})
        except (OSError,ValueError) as e: rejected.append({'path':rel,'reason':str(e)})
    if len(kept)<10: raise ValueError('Need at least 10 unique readable images to form all three splits.')
    n=len(kept);a=int(n*fractions[0]);b=a+int(n*fractions[1])
    parts={'train':kept[:a],'validation':kept[a:b],'test':kept[b:]}
    record={'dataset':DATASET_URL,'seed':seed,'size':size,'files_discovered':len(files),
            'candidates_scanned':len(order),'unique_retained':n,'rejected':rejected,'duplicates':duplicates,
            'preprocessing':'EXIF orientation; RGB; center-square crop; bicubic resize; uint8 / 255',
            'fractions':list(fractions),'splits':parts,
            'limitation':'Exact equality after preprocessing only; near-duplicate/character leakage not eliminated.'}
    (out/'data_manifest.json').write_text(json.dumps(record,indent=2))
    return record

class FaceDataset(Dataset):
    """Owns file paths and size; __getitem__ decodes one image into a new tensor."""
    def __init__(self, root, rows, size): self.root=Path(root);self.rows=rows;self.size=size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        im=rgb_image(self.root/self.rows[i]['path'],self.size)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255

def make_loaders(root, manifest, batch_size=64, seed=121):
    datasets={k:FaceDataset(root,rows,manifest['size']) for k,rows in manifest['splits'].items()}
    generator=torch.Generator().manual_seed(seed)
    # num_workers=0 is portable across Kaggle, Colab and local notebooks.
    loaders={k:DataLoader(ds,batch_size=batch_size,shuffle=k=='train',num_workers=0,
                         generator=generator if k=='train' else None,drop_last=False)
             for k,ds in datasets.items()}
    return datasets,loaders

def collect(dataset, n=256):
    if n<1 or n>len(dataset): raise ValueError(f'n must be in [1,{len(dataset)}]')
    return torch.stack([dataset[i] for i in range(n)])

def checked_images(x):
    x=torch.as_tensor(x).detach().cpu().float()
    if x.ndim!=4 or x.shape[1]!=3 or len(x)==0: raise ValueError('Expected nonempty (N,3,H,W) RGB batch.')
    if not torch.isfinite(x).all() or x.min() < -1e-5 or x.max()>1+1e-5:
        raise ValueError('Convert outputs explicitly to [0,1]; nonfinite/out-of-range values are not silently clipped.')
    return x.clamp(0,1)

def save_grid(images, path, title='', nrow=8):
    x=checked_images(images);n=len(x);cols=min(nrow,n);rows=(n+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(cols*1.4,rows*1.4),squeeze=False)
    for i,ax in enumerate(axes.flat):
        ax.axis('off')
        if i<n: ax.imshow(x[i].permute(1,2,0).numpy());ax.set_title(str(i+1),fontsize=8)
    if title:fig.suptitle(title,fontsize=12)
    fig.tight_layout();Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)
    return Path(path)

class RunLog:
    """Stores scalar observations; students decide losses, updates and when to record."""
    def __init__(self, folder, config):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter();self.rows=[]
        (self.folder/'config.json').write_text(json.dumps(config,indent=2))
    def record(self, step, **scalars):
        values={k:float(v) for k,v in scalars.items()}
        if not all(np.isfinite(v) for v in values.values()): raise ValueError('Nonfinite logged value: stop and diagnose.')
        self.rows.append({'step':int(step),'seconds':time.perf_counter()-self.started,**values})
        (self.folder/'history.json').write_text(json.dumps(self.rows,indent=2))
    def plot(self, path=None):
        if not self.rows:raise ValueError('No logged observations.')
        keys=sorted(set().union(*(r.keys() for r in self.rows))-{'step','seconds'})
        fig,axes=plt.subplots(len(keys),1,figsize=(8,2.5*len(keys)),squeeze=False)
        for ax,key in zip(axes.flat,keys):
            rows=[r for r in self.rows if key in r];ax.plot([r['step'] for r in rows],[r[key] for r in rows]);ax.set(xlabel='Optimizer iteration (declared convention)',ylabel=key);ax.grid(alpha=.2)
        fig.tight_layout();p=Path(path) if path else self.folder/'curves.png';fig.savefig(p,dpi=140);plt.close(fig);return p

def parameter_counts(models):
    return {name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}

def save_checkpoint(path, models, optimizers, config, step):
    """Weights/optimizer state + config. Not an exact mid-epoch resume protocol."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'models':{k:v.state_dict() for k,v in models.items()},
                'optimizers':{k:v.state_dict() for k,v in optimizers.items()},
                'config':config,'step':int(step),'torch_rng':torch.get_rng_state()},path)
    return path

def pairwise_pixel_mse(images):
    """Mean pairwise pixel MSE over distinct pairs, not a perceptual quality metric."""
    x=checked_images(images).flatten(1).double();n=len(x)
    if n<2:raise ValueError('Need at least two images.')
    # Mean unordered-pair squared difference = 2n/(n-1) * population variance.
    return float(2*n/(n-1)*x.var(dim=0,unbiased=False).mean())

def nearest_pixels(queries, reference, chunk_size=128):
    q=checked_images(queries);r=checked_images(reference)
    if q.shape[1:]!=r.shape[1:]:raise ValueError('Match image shapes and preprocessing.')
    q=q.flatten(1);r=r.flatten(1);best=torch.full((len(q),),float('inf'));ids=torch.zeros(len(q),dtype=torch.long)
    # Bounded query/reference chunks; no (queries, references, pixels) allocation.
    for start in range(0,len(q),64):
        block=q[start:start+64];dist=torch.full((len(block),),float('inf'));idx=torch.zeros(len(block),dtype=torch.long)
        for j in range(0,len(r),chunk_size):
            rb=r[j:j+chunk_size];d=(block.square().mean(1)[:,None]+rb.square().mean(1)[None,:]-2*block@rb.T/block.shape[1]).clamp_min(0)
            v,k=d.min(1);update=v<dist;dist[update]=v[update];idx[update]=k[update]+j
        best[start:start+len(block)]=dist;ids[start:start+len(block)]=idx
    return {'mse':best,'index':ids}

def save_neighbors(queries, reference, result, path):
    q=checked_images(queries);r=checked_images(reference);n=min(8,len(q))
    fig,axes=plt.subplots(n,2,figsize=(5,n*2),squeeze=False)
    for i in range(n):
        for ax,im,label in [(axes[i,0],q[i],f'Query {i+1}'),(axes[i,1],r[int(result['index'][i])],f"Nearest · MSE {result['mse'][i]:.4f}")]:
            ax.imshow(im.permute(1,2,0));ax.set_title(label,fontsize=10);ax.axis('off')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig);return Path(path)

def evaluate_samples(generated, train_reference, heldout_reference):
    """Same-size reference sets make NN comparisons less confounded by pool size."""
    g=checked_images(generated);t=checked_images(train_reference);h=checked_images(heldout_reference)
    if len(t)!=len(h):raise ValueError('Use equally sized train and held-out reference pools.')
    nt=nearest_pixels(g,t);nh=nearest_pixels(g,h)
    return {'generated_count':len(g),'reference_count_each':len(t),
            'generated_pairwise_pixel_mse':pairwise_pixel_mse(g),
            'heldout_pairwise_pixel_mse':pairwise_pixel_mse(h),
            'mean_nearest_train_pixel_mse':float(nt['mse'].mean()),
            'mean_nearest_heldout_pixel_mse':float(nh['mse'].mean())},nt,nh

def render_report(path, sections, config, metrics, figures):
    """Self-contained HTML; plain-text authored sections + embedded PNGs, printable to PDF.
    figures is a list of (path, caption). This renders evidence; it does not write conclusions.
    """
    esc=lambda x:html.escape(str(x))
    required=['Question','Dataset','Architecture','Training','Results and metrics','Limitations and conclusion','References']
    if any(not sections.get(k,'').strip() or '[WRITE' in sections[k] for k in required):
        raise ValueError('Write every report section; replace the [WRITE ...] placeholders.')
    document='<!doctype html><html lang="en"><meta charset="utf-8"><title>Anime face generation · Experimental report</title><style>body{max-width:960px;margin:3rem auto;padding:0 1.4rem;font:17px/1.6 system-ui;color:#17212b}h1{font-size:2rem}h2{margin-top:2rem;color:#2878c8}p{white-space:pre-wrap}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{border-collapse:collapse;width:100%}td,th{padding:.5rem;border-bottom:1px solid #ccc;text-align:left;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f5f1e8;padding:1rem}@media print{body{font-size:10pt;margin:0;max-width:none}figure,table{break-inside:avoid}h2{break-after:avoid}@page{size:A4;margin:18mm}}</style><body><h1>Anime face generation</h1><p>Experimental report · student-authored analysis</p>'
    for title in required:
        document+=f'<h2>{esc(title)}</h2><p>{esc(sections[title])}</p>'
        if title=='Training':document+='<pre>'+esc(json.dumps(config,indent=2))+'</pre>'
        if title=='Results and metrics':
            document+='<table><tr><th>Run</th><th>Measurement</th><th>Value</th></tr>'
            for run,values in metrics.items():
                for key,value in values.items():document+=f'<tr><td>{esc(run)}</td><td>{esc(key)}</td><td>{esc(value)}</td></tr>'
            document+='</table>'
            for file,caption in figures:
                file=Path(file)
                if file.suffix.lower()!='.png':raise ValueError('Report figures must be PNG.')
                encoded=base64.b64encode(file.read_bytes()).decode()
                document+=f'<figure><img src="data:image/png;base64,{encoded}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(document+'</body></html>');return path
