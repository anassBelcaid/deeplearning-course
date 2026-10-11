"""Verify provided infrastructure without completing the student's generative model.
Run: MPLCONFIGDIR=/tmp/anime-mpl python verify.py /path/to/12-preview-images
"""
from pathlib import Path
import ast,json,sys,tempfile,shutil,zipfile
import nbformat
import numpy as np
import torch
from PIL import Image
from support import *
R=Path(__file__).resolve().parent
nb=nbformat.read(R/'starter/anime_faces_homework.ipynb',as_version=4);nbformat.validate(nb)
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
    assert not any(ord(x)<32 and x not in '\n\r\t' for x in c.source), 'Unexpected control character'
# Standalone embedded support must match its source exactly.
assert any(c.cell_type=='code' and c.source.strip()==(R/'support.py').read_text().strip() for c in nb.cells)
assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
assert sum('raise NotImplementedError' in c.source for c in nb.cells if c.cell_type=='code')==2
with tempfile.TemporaryDirectory(prefix='anime-homework-check-') as temp:
    root=Path(temp)/'images';root.mkdir();out=Path(temp)/'outputs'
    for p in Path(sys.argv[1]).glob('*.jpg'):shutil.copy(p,root/p.name)
    # Same decoded image with different encoding: must not cross a split.
    first=next(root.glob('*.jpg'))
    with Image.open(first) as im:im.save(root/'copy.png')
    (root/'broken.jpg').write_bytes(b'not an image')
    manifest=prepare_data(root,out,size=32,max_images=None,seed=17,fractions=(.5,.25,.25))
    assert manifest['files_discovered']==14 and manifest['unique_retained']==12
    assert len(manifest['duplicates'])==len(manifest['rejected'])==1
    again=prepare_data(root,Path(temp)/'again',32,None,17,(.5,.25,.25))
    assert manifest==again
    datasets,loaders=make_loaders(root,manifest,batch_size=4,seed=19)
    hashes=[{r['pixel_sha256'] for r in rows} for rows in manifest['splits'].values()]
    assert not hashes[0]&hashes[1] and not hashes[0]&hashes[2] and not hashes[1]&hashes[2]
    x=collect(datasets['train'],6);v=collect(datasets['validation'],3)
    assert x.shape==(6,3,32,32) and x.min()>=0 and x.max()<=1
    a=next(iter(loaders['train']));_,same=make_loaders(root,manifest,4,19)
    torch.testing.assert_close(a,next(iter(same['train'])))
    brute=torch.stack([(x[i]-x[j]).double().square().mean() for i in range(6) for j in range(i+1,6)]).mean().item()
    assert abs(pairwise_pixel_mse(x)-brute)<1e-8
    assert pairwise_pixel_mse(x[:1].repeat(6,1,1,1))==0
    nn=nearest_pixels(x[:2],x,chunk_size=2)
    assert nn['index'].tolist()==[0,1] and nn['mse'].max()<1e-6
    m,nt,nh=evaluate_samples(x[:3],x[3:6],v)
    assert all(np.isfinite(z) for z in m.values())
    try:evaluate_samples(x,x,v)
    except ValueError:pass
    else:raise AssertionError('Unequal reference pools must be rejected')
    try:checked_images(x*4)
    except ValueError:pass
    else:raise AssertionError('Out-of-range outputs must be rejected')
    grid=save_grid(x,out/'grid.png','Infrastructure check: real data, no trained model')
    near=save_neighbors(x[:2],x,nn,out/'near.png')
    log=RunLog(out/'run',{'seed':17});log.record(1,diagnostic=.4);log.record(2,diagnostic=.3);log.plot()
    # Small arbitrary linear layer only tests serialization, not a generative reference solution.
    model=torch.nn.Linear(3,2);opt=torch.optim.Adam(model.parameters());inp=torch.ones(1,3)
    file=save_checkpoint(out/'state.pt',{'probe':model},{'probe':opt},{'purpose':'serialization check'},2)
    restored=torch.nn.Linear(3,2);restored.load_state_dict(torch.load(file,weights_only=True)['models']['probe'])
    torch.testing.assert_close(model(inp),restored(inp))
    sections={k:'Infrastructure verification only. This does not report a trained generator result.' for k in ['Question','Dataset','Architecture','Training','Results and metrics','Limitations and conclusion','References']}
    page=render_report(out/'report.html',sections,{'seed':17},{'probe':m},[(grid,'Real input test'),(near,'Identity-search check')]).read_text()
    assert page.count('data:image/png;base64,')==2 and '<table>' in page
    assert '[WRITE' not in page
with zipfile.ZipFile(R/'assets/anime-homework.zip') as z:
    assert set(z.namelist())=={'anime_faces_homework.ipynb','support.py','report_template.qmd','README.md'}
print(json.dumps({'notebook_cells':len(nb.cells),'data_checks':'passed: actual RGB files, corruption, duplicate removal, seeded disjoint split',
                  'metrics':'passed: pairwise brute-force equality, exact-match nearest search, range/pool guards',
                  'artifacts':'passed: grids, logs, weights round-trip, embedded report, portable ZIP',
                  'scope':'No student architecture or training solution supplied'},indent=2))
