"""Extract approved GAN visual evidence into self-contained notes panels.
Run after rendering the GAN lecture; no model training or inference is needed.
Requires BeautifulSoup. Assets retain the lecture as their single source of truth.
"""
from pathlib import Path
from bs4 import BeautifulSoup
import base64, json
R=Path(__file__).resolve().parent
SITE=R.parents[2]
L=SITE/'lectures/12-generative-adversarial-networks'
soup=BeautifulSoup((SITE/'_site/lectures/12-generative-adversarial-networks/index.html').read_text(),'html.parser')
theme=next(x['href'] for x in soup.select('link[rel="stylesheet"]') if '/dist/theme/quarto-' in x['href'])
panels={
 'target':('gan-target-experiment',1,['gan-judge-experiment']),
 'judge':('gan-judge-experiment',1,['gan-target-experiment']),
 'feedback':('gan-feedback-idea',0,[]),
 'architecture':('gan-architecture',0,[]),
 'routing':('gan-routing-exercise',2,[]),
 'generator':('gan-generator-code',0,[]),
 'discriminator':('gan-discriminator-code',0,[]),
 'source-loss':('gan-discriminator-loss',0,[]),
 'opposing':('gan-opposing-losses',3,[]),
 'gradient':('gan-gradient-loss',3,['gan-opposing-losses']),
 'updates':('gan-alternating-paths',4,[]),
 'conv-comparison':('gan-conv-architecture',0,[]),
 'spatial':('gan-spatial-architecture',0,[]),
 'transpose':('gan-transpose-mechanism',0,[]),
 'training':('gan-conv-training',5,[]),
 'evaluation':('gan-evaluate-gallery',6,['gan-evaluate-losses']),
 'loss-history':('gan-evaluate-losses',6,['gan-evaluate-gallery']),
 'neighbors':('gan-nearest-neighbors',0,[]),
}
styles=['../../site_libs/revealjs/dist/reset.css','../../site_libs/revealjs/dist/reveal.css',theme,'../01-what-is-learning/slides.css','week12.css']
for name,(sid,script,deps) in panels.items():
 section=BeautifulSoup(str(soup.find('section',id=sid)),'html.parser').section
 assert section, sid
 # Standalone panels have no Reveal lifecycle to mark the visible section active.
 section['class']=[c for c in section.get('class',[]) if c not in ['past','future']]+['present']
 for e in section.select('aside.notes, h2, .gan-source, .takeaway, .exercise-pause'):e.decompose()
 for e in section.select('.fragment'):
  e['class']=[c for c in e.get('class',[]) if c!='fragment']
 if name in ['generator','discriminator']:
  section.select_one('.light-code-editor').decompose()
  section.select_one('.gan-code-layout')['style']='display:block'
 dependency=''.join('<div hidden>'+str(soup.find('section',id=d))+'</div>' for d in deps)
 doc='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><base href="../../../../lectures/12-generative-adversarial-networks/"><title>GAN notes: '+name+'</title>'
 doc+=''.join(f'<link rel="stylesheet" href="{s}">' for s in styles)
 doc+='''<style>
 html,body { margin:0; padding:0; background:#f5f1e8; overflow:hidden; }
 #canvas { width:1280px; transform-origin:top left; }
 .reveal { width:1280px; height:auto; font-size:28px; }
 .reveal .slides { width:1280px!important; height:auto!important; position:relative; inset:auto; margin:0; transform:none!important; perspective:none; text-align:left; }
 .reveal .slides>section { display:block; position:relative; top:auto; left:auto; padding:22px; width:100%; height:auto; box-sizing:border-box; transform:none; opacity:1; pointer-events:auto; }
 .reveal .fragment { opacity:1; visibility:visible; }
 .reveal section [hidden] { display:none!important; }
 .reveal .gan-source,.reveal .slide-number { display:none; }
 .reveal .gan-code-layout { margin:0; }
 .reveal .gan-code-layout>div:empty { display:none; }
 .reveal .gan-routing-solution { position:relative; top:auto; left:auto; right:auto; margin-top:12px; box-shadow:none; }
 .reveal .gan-spatial-architecture svg { max-width:100%; }
 </style></head><body><div id="canvas"><div class="reveal"><div class="slides">'''+str(section)+'</div></div></div>'+dependency
 # Reveal event methods are not needed in a persistent notes panel.
 doc+='<script>window.Reveal={on:()=>{},layout:()=>{}};</script>'
 if script:doc+=(L/f'assets/section{script}-controls.html').read_text()
 doc+='''<script>
function resize(){const c=document.getElementById('canvas'),s=Math.min(1,innerWidth/1280);c.style.transform='scale('+s+')';const h=Math.ceil(c.getBoundingClientRect().height)+2;document.body.style.height=h+'px';parent.postMessage({type:'gan-notes-height',height:h},location.origin);}
new ResizeObserver(resize).observe(document.querySelector('#canvas .slides'));addEventListener('resize',resize);addEventListener('load',resize);document.fonts.ready.then(resize);resize();
</script></body></html>'''
 (R/'demos'/f'{name}.html').write_text(doc)
 (R/f'_demo-{name}.qmd').write_text(f'<div class="gan-demo"><iframe src="assets/demos/{name}.html" title="GAN visual evidence: {name.replace("-"," ")}" loading="lazy"></iframe><a href="assets/demos/{name}.html" target="_blank">Open this visual in a full window →</a></div>\n')
# Exact executable excerpts, native notes code rather than reduced slide screenshots.
for file,blocks in [('gan_modules.py',['GENERATOR','DISCRIMINATOR']),('conv_gan.py',['G','D'])]:
 source=(L/'assets'/file).read_text()
 for block in blocks:
  code=source.split('# BEGIN '+block+'\n')[1].split('# END '+block)[0]
  (R/f'_code-{block.lower()}.qmd').write_text('```python\n'+code+'```\n')
for name in ['d_step','g_step']:
 code=BeautifulSoup((L/f'assets/{name}-code.html').read_text(),'html.parser').select_one('pre code').get_text()
 (R/f'_code-{name}.qmd').write_text('```python\n'+code.strip()+'\n```\n')
# The blind exercise keeps answers in the chapter's native details solution.
gallery='<div class="blind-gallery">'
for i,letter in enumerate('ABCDEF'):
 gallery+=f'<figure><img src="../../lectures/12-generative-adversarial-networks/assets/evidence/section1/gallery-{i}.png" alt="Garment {letter}: source concealed for the exercise"><figcaption>{letter}</figcaption></figure>'
(R/'_blind-gallery.qmd').write_text(gallery+'</div>\n')
# Original text-free cover: noise expands through G; generated and real images
# merge into a contracting D; the lower coral curve represents training feedback.
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="640" viewBox="0 0 1400 640"><rect width="1400" height="640" fill="#17212b"/><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1L9 5L1 9" fill="none" stroke="#bfd4df" stroke-width="2"/></marker></defs>']
for x,y in [(100,320),(124,348),(92,373),(151,382),(162,322),(132,413)]:s.append(f'<circle cx="{x}" cy="{y}" r="7" fill="#7bb7e1"/>')
s.append('<path d="M190 365H275M490 365H565M760 365H840M760 135H810V340H840M1150 365H1240" fill="none" stroke="#bfd4df" stroke-width="3" marker-end="url(#arrow)"/>')
for i in range(4):
 h=70+36*i;x=290+48*i
 s.append(f'<rect x="{x}" y="{365-h/2}" width="30" height="{h}" rx="5" fill="#529cd2"/>')
 h=180-42*i;x=860+68*i
 s.append(f'<rect x="{x}" y="{365-h/2}" width="40" height="{h}" rx="5" fill="#dc886e"/>')
for y,file in [(65,'section1/gallery-0.png'),(285,'section5/epoch-60-sample-0.png')]:
 data=base64.b64encode((L/'assets/evidence'/file).read_bytes()).decode()
 s.append(f'<rect x="580" y="{y-8}" width="176" height="176" rx="12" fill="#ffffff18"/><image href="data:image/png;base64,{data}" x="588" y="{y}" width="160" height="160"/>')
s.append('<circle cx="1260" cy="365" r="16" fill="#efc6b4"/><path d="M1070 465C1030 590 440 600 360 495" fill="none" stroke="#ef805c" stroke-width="4" stroke-dasharray="9 9"/><path d="M350 511L359 489L379 503" fill="none" stroke="#ef805c" stroke-width="4"/></svg>')
(R/'chapter-cover.svg').write_text(''.join(s))
print(f'Built {len(panels)} panels, source-linked code, blind gallery, and original cover.')
