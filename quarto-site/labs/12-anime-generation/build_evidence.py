"""Build real-data previews and diagnostic controls; no generative training.
Usage: python build_evidence.py /path/to/12-downloaded-preview-images
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from PIL import Image
from support import rgb_image, save_grid, pairwise_pixel_mse
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'assets/preview';OUT.mkdir(parents=True,exist_ok=True)
files=sorted(Path(sys.argv[1]).glob('*.jpg'));assert len(files)==12
x=torch.stack([torch.from_numpy(np.asarray(rgb_image(p,64)).copy()).permute(2,0,1).float()/255 for p in files])
save_grid(x,OUT/'dataset.png','Actual Kaggle images · small file-list preview, not a random population sample',6)
save_grid(x[:1].repeat(12,1,1,1),OUT/'repeated.png','Constructed diagnostic control: the first real image repeated 12 times',6)
noise=torch.rand(x.shape,generator=torch.Generator().manual_seed(912))
save_grid(noise,OUT/'noise.png','Constructed noise control: pixel variation does not imply plausible faces',6)
record={'dataset':'https://www.kaggle.com/datasets/splcher/animefacedataset','retrieved':'2026-10-11',
        'selection':'First 12 filenames returned by Kaggle files listing; reordered lexicographically for display. Not representative and not generated outputs.',
        'listing_license':'DbCL-1.0 (dataset metadata; report the source and check its current terms)',
        'preprocessing':'EXIF transpose, RGB, center-square crop, bicubic resize to 64, divide by 255',
        'files':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':list(Image.open(p).size)} for p in files],
        'pairwise_pixel_mse':{'real_preview':pairwise_pixel_mse(x),'repeated_control':pairwise_pixel_mse(x[:1].repeat(12,1,1,1)),'noise_control':pairwise_pixel_mse(noise)},
        'scope':'Measured infrastructure examples only; no trained anime generator is supplied.'}
(OUT/'measurements.json').write_text(json.dumps(record,indent=2)+'\n')
# Vector diagrams preview student decisions and object/data ownership.
svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 510" role="img" aria-label="Provided data paths feed a student-designed experiment, then measured outputs feed the report"><rect width="1200" height="510" fill="#f5f1e8"/><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1L9 5L1 9" fill="none" stroke="#526575" stroke-width="1.5"/></marker></defs><g font-family="sans-serif" fill="#17212b"><text x="35" y="40" font-size="25" font-weight="bold">Provided infrastructure</text><text x="458" y="40" font-size="25" font-weight="bold">Your scientific decisions</text><text x="878" y="40" font-size="25" font-weight="bold">Your evidence</text><g fill="#e0ebf0" stroke="#9ebdd1"><rect x="30" y="70" width="290" height="100" rx="10"/><rect x="30" y="215" width="290" height="120" rx="10"/></g><g text-anchor="middle" font-size="21"><text x="175" y="108">Manifest: paths + hashes</text><text x="175" y="145">train / validation / test</text><text x="175" y="256">FaceDataset stores paths</text><text x="175" y="290">Loader decodes RGB tensors</text></g><g fill="#f6e1d9" stroke="#d7aea0"><rect x="430" y="70" width="320" height="100" rx="10"/><rect x="430" y="215" width="320" height="120" rx="10"/></g><g text-anchor="middle" font-size="21"><text x="590" y="108">Model + objective</text><text x="590" y="145">Architecture you justify</text><text x="590" y="255">Training + one change</text><text x="590" y="290">Separate model parameters</text></g><rect x="865" y="145" width="305" height="190" rx="10" fill="#e2efe8" stroke="#a0c4ac"/><g text-anchor="middle" font-size="21"><text x="1017" y="190">Samples + checkpoints</text><text x="1017" y="235">Metrics + limitations</text><text x="1017" y="280">Self-contained report</text></g><g fill="none" stroke="#526575" stroke-width="3" marker-end="url(#a)"><path d="M175 178V203"/><path d="M332 275H418"/><path d="M590 178V203"/><path d="M762 275H853"/></g><text x="35" y="405" font-size="21">Manifest entries reference files; they are not copied image tensors.</text><text x="35" y="445" font-size="21">Only the training split enters optimizer updates. Validation selects; test evaluates once.</text></g></svg>'''
(ROOT/'assets/ownership.svg').write_text(svg)
print(json.dumps(record['pairwise_pixel_mse'],indent=2))
