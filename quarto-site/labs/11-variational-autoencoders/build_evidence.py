"""Explicit, reproducible reference run. Never invoked by Quarto.
COURSE_DATA_DIR=/path/to/data python build_evidence.py [--reuse]
"""
from pathlib import Path
import os, sys, json, hashlib, platform
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND','Agg')
os.environ['COURSE_OUTPUT_DIR']=str(ROOT/'assets/expected')
os.environ['VAE_LAB_ASSETS']=str(ROOT/'assets')
ns={'__name__':'lab_reference'}
exec((ROOT/'support.py').read_text(),ns)
exec((ROOT/'reference_model.py').read_text(),ns)
ns['plt'].show=lambda:None
train_x,val_x,val_y,selected,val_ids=ns['prepare_data']()
manifest_path=ROOT/'assets/checkpoints/manifest.json'
if '--reuse' not in sys.argv:
    manifest=dict(seed=117,split_seed=41,train_size=12000,val_size=2000,epochs=20,
        batch_size=256,lr=.001,latent_dim=2,torch=str(ns['torch'].__version__),
        device=str(ns['DEVICE']),platform=platform.platform(),
        objective='Pixel-summed soft-target BCE + beta * latent-summed KL, batch mean',
        validation='One fixed posterior draw per validation image, noise seed 811',
        selected_original_ids=val_ids[selected].tolist(),models={})
    for key,beta,deterministic in [('ae',0.,True),('beta0',0.,False),('beta1',1.,False),('beta20',20.,False)]:
        print('\nTraining',key,flush=True)
        model=ns['fresh_model']()
        result=ns['fit'](model,train_x,val_x,beta=beta,epochs=20,seconds=float('inf'),deterministic=deterministic)
        path=ROOT/'assets/checkpoints'/f'{key}.pt'
        ns['torch'].save({k:v.cpu() for k,v in model.state_dict().items()},path)
        manifest['models'][key]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            beta=beta,deterministic=deterministic,**result)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
models,manifest=ns['load_reference_models']()
ns['opening_gallery'](models['ae'],val_x,selected)
ns['posterior_view'](models['beta1'],val_x,selected)
ns['prior_view'](models['beta1'])
ns['comparison'](models,val_x,val_y,selected)
ns['training_curves'](manifest)
probe=ns['latent_use'](models,val_x)
live=ns['fresh_model']()
print('\nBenchmark student live run',flush=True)
benchmark=ns['fit'](live,train_x[:4096],val_x[:512],epochs=2,seconds=60.)
ns['image_rows']([val_x[selected],live.decoder(live.encoder(val_x[selected].to(ns['DEVICE']))[0])],
    ['Input','Short live run'],'short-run.png',ns['NAMES'])
metrics=dict(reference={key:record['history'][-1] for key,record in manifest['models'].items()},
             latent_use=probe,live_benchmark=benchmark,
             scope='One seed, small architecture and training subset; not a benchmark ranking')
(ROOT/'assets/expected/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
