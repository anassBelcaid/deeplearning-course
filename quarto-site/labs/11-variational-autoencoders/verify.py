"""Execute the completed student path, public checks, and every widget choice.
Writes validation artifacts outside the source tree. No long training is required.
"""
import os, runpy, tempfile, zipfile, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND','Agg')
os.environ['COURSE_OUTPUT_DIR']=tempfile.mkdtemp(prefix='vae-lab-check-')
os.environ['VAE_LAB_ASSETS']=str(ROOT/'assets')
source=runpy.run_path(str(ROOT/'build_notebook.py'))
ns={'__name__':'student_reference'}; count=0; start=time.perf_counter()
for cell in source['CELLS']:
    if cell['kind']!='code':continue
    code=cell.get('solution') or cell['text']
    exec(compile(code,f'notebook-code-{count}','exec'),ns)
    ns['plt'].show=lambda:None
    count+=1
assert len(ns['live_run']['history'])==2
widgets=ns['lab_controls']
if widgets:
    for model in ['beta0','beta1','beta20']:
        widgets['model'].value=model
        for category in range(10):
            widgets['garment'].value=category
            widgets['posterior'].click()
        widgets['prior'].click()
    assert (ns['OUT']/'interactive-posterior.png').is_file()
    assert (ns['OUT']/'interactive-prior.png').is_file()
# Verify the packaged fallback with course-path and server access disabled.
with tempfile.TemporaryDirectory(prefix='vae-lab-zip-') as directory:
    with zipfile.ZipFile(ROOT/'assets/vae-session.zip') as z:
        assert z.testzip() is None
        z.extractall(directory)
    os.environ['VAE_LAB_ASSETS']=str(Path(directory)/'assets')
    original=ns['urllib'].request.urlopen
    ns['urllib'].request.urlopen=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Unexpected network'))
    try: models,manifest=ns['load_reference_models']()
    finally:ns['urllib'].request.urlopen=original
    assert set(models)=={'ae','beta0','beta1','beta20'}
# Noise-independent posterior-mean outputs and different posterior draws.
model=ns['models']['beta1']; x=ns['val_x'][:4].to(ns['DEVICE'])
with ns['torch'].inference_mode():
    mu,_=model.encoder(x)
    ns['torch'].testing.assert_close(model.decoder(mu),model.decoder(mu),rtol=0,atol=0)
    assert not ns['torch'].equal(model(x)[0],model(x)[0])
report=dict(code_cells=count,seconds=time.perf_counter()-start,
    widget_posterior_combinations=30 if widgets else 0,widget_prior_combinations=3 if widgets else 0,
    offline_checkpoints=4,all_checks_passed=True,output=str(ns['OUT']))
Path('/tmp/vae-lab-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
