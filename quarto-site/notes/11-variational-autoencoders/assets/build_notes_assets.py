"""Rebuild the original vector cover and standalone lecture demos after rendering the deck.
Requires BeautifulSoup; never trains a model. Source visual evidence remains in lecture assets.
"""
from pathlib import Path
import base64
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parent
SITE=ROOT.parents[2]
LECTURE=SITE/'lectures/11-variational-autoencoders'
html=BeautifulSoup((SITE/'_site/lectures/11-variational-autoencoders/index.html').read_text(),'html.parser')
styles=['../../site_libs/revealjs/dist/reset.css','../../site_libs/revealjs/dist/reveal.css']+[e['href'] for e in html.select('link[rel="stylesheet"]') if '/dist/theme/quarto-' in e['href']]+['../01-what-is-learning/slides.css','week11.css']
demos={
 'generation':('a-fresh-draw-without-an-input-image','explorer.html'),
 'posterior':('posterior-experiment','explorer.html'),
 'ablation':('prior-penalty-experiment','loss-explorer.html'),
 'kl':('kl-explorer','loss-explorer.html'),
 'elbo':('elbo-evidence','loss-explorer.html'),
 'reparameterization':('reparameterization-graph','reparameterization.html'),
 'inspection':('vae-inspection','training-inspection.html'),
 'interpolation':('vae-latent-path','latent-explorer.html')}
# Resolve the ablation section by its stable lab id rather than its heading slug.
demos['ablation']=(html.find(id='loss-ablation-lab').find_parent('section')['id'],'loss-explorer.html')
for name,(sid,script) in demos.items():
 section=BeautifulSoup(str(html.find('section',id=sid)),'html.parser').section
 for e in section.select('aside.notes, h2, .vae-source, .vae-line'):e.decompose()
 if name=='inspection':
  section.select_one('.light-code-editor').decompose()
 if name=='elbo':
  for e in section.select('.elbo-identity'):e.decompose()
 section['id']='demo-'+name
 for e in section.select('.fragment'): e['class']=[c for c in e.get('class',[]) if c!='fragment']
 # Shared loss loader also fills these numeric labels in the lecture's next slide.
 hidden='<div hidden><span id="rec-cost-0"></span><span id="rec-cost-1"></span><span id="rec-cost-2"></span><span id="rec-average"></span></div>' if name=='ablation' else ''
 if name=='inspection': hidden+=''.join(f'<span hidden id="inspection-code-{mode}"></span>' for mode in ['mean','posterior','prior'])
 document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><base href="../../../../lectures/11-variational-autoencoders/"><title>VAE '+name+' experiment</title>'
 document+=''.join(f'<link rel="stylesheet" href="{s}">' for s in styles)
 document+='''<style>
html,body { margin:0; height:auto; overflow:visible; background:#f5f1e8; }
.reveal { font-size:28px; height:auto; width:100%; }
.reveal section { display:block; position:relative; box-sizing:border-box; padding:20px; width:100%; height:auto; text-align:left; }
.reveal .fragment { opacity:1; visibility:visible; }
.reveal .vae-live { grid-template-columns:1fr 1fr; gap:25px; margin:0; }
.reveal .vae-live img { width:100%; height:auto; max-height:none; }
.reveal .vae-live-controls h3 { font-size:29px; }
.reveal .posterior-stage { grid-template-columns:1fr 1.2fr 1fr; gap:15px; }
.reveal .posterior-stage figure img { width:100%; height:auto; }
.reveal .posterior-toolbar { margin-top:0; }
.reveal .training-inspection-layout { display:block; margin:0 auto; max-width:500px; }
.reveal .loss-ablation-layout { grid-template-columns:1fr; margin:0; }
.reveal .loss-ablation-controls p { font-size:22px; }
.reveal .loss-ablation-figure>figure { width:100%; height:auto; }
.reveal .loss-ablation-figure img { height:auto; width:100%; }
.reveal .elbo-example { grid-template-columns:1fr; gap:20px; margin:0; }
.reveal .reparam-controls { flex-wrap:wrap; margin:10px 0; gap:18px; }
.reveal .reparam-controls label { width:40%; }
.reveal .reparam-graph { height:auto; }
.reveal .reparam-rule p { font-size:23px; }
.reveal .latent-path-layout { gap:15px; }
.reveal .latent-endpoints p { font-size:20px; }
.reveal .latent-endpoints img { width:45px; height:45px; }
.reveal .latent-map-key { font-size:17px; }
.reveal #latent-path-map { height:auto; }
.reveal #latent-path-output { width:220px; height:220px; }
.reveal .kl-explorer { grid-template-columns:1fr; }
.reveal #kl-curves { height:auto; }
.reveal .kl-costs { gap:12px; }
.reveal .kl-costs strong { font-size:29px; }
.reveal .kl-costs span { font-size:13px; }
@media(max-width:700px) {
 .reveal .posterior-stage,.reveal .latent-path-layout,.reveal .vae-live { grid-template-columns:1fr; }
 .reveal .posterior-stage figure img { width:200px; margin:auto; }
 .reveal .vae-live img { width:260px; margin:auto; }
 .reveal .posterior-stage figure { max-width:100%; }
 .reveal .posterior-stage figcaption { font-size:21px; }
 .reveal .kl-costs { display:flex; flex-wrap:wrap; justify-content:center; }
}
</style></head><body><div class="reveal">'''+str(section)+hidden+'</div>'
 document+=(LECTURE/'assets'/script).read_text()
 document+='''<script src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS_HTML-full"></script>
<script>
function height(){parent.postMessage({type:'vae-notes-height',height:Math.ceil(document.querySelector('.reveal').getBoundingClientRect().height)+8},location.origin);}
new ResizeObserver(height).observe(document.querySelector('.reveal'));
window.addEventListener('load',height);
</script></body></html>'''
 (ROOT/'demos'/f'{name}.html').write_text(document)
# Purpose-made text-free cover: data-conditioned clouds feed the shared decoder,
# while prior draws reach it independently. Displayed images are actual course outputs.
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="680" viewBox="0 0 1400 680"><rect width="1400" height="680" fill="#17212b"/>']
for cx,cy,rx,ry in [(460,180,74,26),(445,350,48,42),(480,505,35,65)]:
 for scale,opacity in [(1.9,.09),(1.35,.15),(1,.28)]:svg.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx*scale}" ry="{ry*scale}" fill="#77b9e8" opacity="{opacity}"/>')
 svg.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="#ef805c"/><path d="M{cx+15} {cy} Q640 {cy} 760 340" fill="none" stroke="#77b9e8" stroke-width="3" opacity=".6"/>')
for i in range(4):svg.append(f'<rect x="{755+i*35}" y="{340-(75+i*45)/2}" width="20" height="{75+i*45}" rx="6" fill="#aa8fd1"/>')
svg.append('<path d="M900 340 H985" stroke="#d8d4ca" stroke-width="4"/><path d="M976 330 L991 340 L976 350" fill="none" stroke="#d8d4ca" stroke-width="4"/>')
for i,y in enumerate([100,270,440]):
 for x,file in [(95,f'posterior/input-{i}.png'),(1060,f'posterior/prior-{i}.png')]:
  data=base64.b64encode((LECTURE/'assets/evidence'/file).read_bytes()).decode()
  svg.append(f'<rect x="{x-9}" y="{y-9}" width="148" height="148" rx="12" fill="#f5f1e8" opacity=".12"/><image href="data:image/png;base64,{data}" x="{x}" y="{y}" width="130" height="130"/>')
 svg.append(f'<path d="M240 {y+65} H340" stroke="#77b9e8" stroke-width="3" opacity=".6"/>')
svg.append('<path d="M560 610 C680 660 745 610 802 490" fill="none" stroke="#ef805c" stroke-width="3" stroke-dasharray="8 9"/>')
for x,y in [(505,610),(531,623),(548,592),(573,630),(580,603)]:svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#ef805c"/>')
svg.append('</svg>');(ROOT/'chapter-cover.svg').write_text(''.join(svg))
print('Built eight standalone demos and original chapter cover.')

# Notes-sized static evidence and code inclusions stay tied to lecture sources.
import ast
source=(LECTURE/'assets/vae_training.py').read_text()
nodes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for filename,name in [('encoder','GaussianEncoder'),('model','VAE'),('loss','vae_loss'),('train','train_step')]:
    code=nodes[name]
    if filename=='encoder': code='import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'+code
    if filename=='train': code='model = VAE()\noptimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n\n'+code
    (ROOT/f'_{filename}.qmd').write_text('```python\n'+code+'\n```\n')
probe=html.find('section',id='sampling-break').find('pre').get_text()
(ROOT/'_sample-probe.qmd').write_text('```python\n'+probe+'\n```\n')
for name in demos:
    label={'generation':'Prior sampling','posterior':'Image-conditioned sampling','ablation':'Controlled objective comparison','kl':'Gaussian KL','elbo':'Exact evidence bound','reparameterization':'Fixed-noise gradient path','inspection':'Three decoder input paths','interpolation':'Latent interpolation'}[name]
    (ROOT/f'_demo-{name}.qmd').write_text(f'<div class="vae-demo"><iframe src="assets/demos/{name}.html" title="Interactive experiment: {label}" loading="lazy"></iframe><a href="assets/demos/{name}.html" target="_blank">Open {label.lower()} in a full window →</a></div>\n')
base='../../lectures/11-variational-autoencoders/assets/evidence/'
def strip(items,extra=''):
    return '<div class="evidence-strip '+extra+'">'+''.join(f'<figure><img src="{base+file}" alt="{alt}"><figcaption>{caption}</figcaption></figure>' for file,alt,caption in items)+'</div>\n'
(ROOT/'_ablation-gallery.qmd').write_text('![Reconstruction-only condition: first four inputs, posterior reconstructions, and prior outputs. All examples are retained; course experiment on Fashion-MNIST.]('+base+'loss/recon-gallery.png)\n\n![Reconstruction-plus-KL condition with the same inputs and noise. These galleries expose different output paths, not a general quality ranking.]('+base+'loss/vae-gallery.png)\n')
(ROOT/'_reconstruction-draws.qmd').write_text(strip([('loss/target.png','Fixed validation reconstruction target','Fixed target')]+[(f'loss/reconstruction-{i}.png',f'Posterior draw {i+1} reconstruction',f'Draw {i+1} · {cost}') for i,cost in enumerate(['338.11','341.37','339.67'])]))
(ROOT/'_prior-gallery.qmd').write_text(strip([(f'latent/prior-{i}.png',f'Unfiltered actual prior draw {i+1}',f'{i+1:02}') for i in range(16)],'evidence-sixteen')+'\nCourse experiment: actual mean outputs, first sixteen draws, seed 619.\n')
(ROOT/'_probe-gallery.qmd').write_text('\n'.join(f'**{label}**\n\n'+strip([(f'latent/probe-{key}-{i}.png',f'{label}, validation case {i+1}',f'Case {i+1}') for i in range(4)]) for key,label in [('input','Observed images'),('right','Correctly paired codes'),('wrong','Reassigned codes')]))
print('Built source-synchronized code, demo embeds, and complete evidence galleries.')
