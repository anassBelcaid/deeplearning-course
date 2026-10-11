"""Reproduce loss arithmetic, autograd checks, and section 3 teaching figures.
No training or image generation. Run with the course PyTorch environment.
"""
from pathlib import Path
import json, math, html
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/section3';OUT.mkdir(parents=True,exist_ok=True)
bce=nn.BCEWithLogitsLoss()
record=json.loads((ROOT/'evidence/section2/contracts.json').read_text())
real_logits=torch.tensor(record['real_logits']).reshape(-1,1)
fake_logits=torch.tensor(record['fake_logits']).reshape(-1,1)
# BEGIN LOSSES
bce = nn.BCEWithLogitsLoss()
loss_D = (
    bce(real_logits, torch.ones_like(real_logits))
    + bce(fake_logits, torch.zeros_like(fake_logits))
)
loss_G = bce(fake_logits, torch.ones_like(fake_logits))
# END LOSSES
checks=[]
for p in [.01,.2,.5,.8,.99]:
    a=torch.tensor(math.log(p/(1-p)),dtype=torch.float64,requires_grad=True)
    minimax=-torch.nn.functional.softplus(a)
    nonsat=torch.nn.functional.softplus(-a)
    dm=torch.autograd.grad(minimax,a,retain_graph=True)[0].item()
    dn=torch.autograd.grad(nonsat,a)[0].item()
    assert abs(dm+p)<1e-12 and abs(dn-(p-1))<1e-12
    assert abs(bce(a,torch.ones_like(a)).item()+math.log(p))<1e-12
    checks.append(dict(p=p,logit=a.item(),minimax=minimax.item(),non_saturating=nonsat.item(),minimax_derivative=dm,non_saturating_derivative=dn))
assert abs(loss_D.item()-(-real_logits.sigmoid().log().mean()-(1-fake_logits.sigmoid()).log().mean()).item())<1e-6
assert abs(loss_G.item()+fake_logits.sigmoid().log().mean().item())<1e-6
(OUT/'loss-checks.json').write_text(json.dumps(dict(purpose='Exact scalar examples and losses on recorded untrained section 2 logits; no training',loss_D=loss_D.item(),loss_G=loss_G.item(),gradient_checks=checks,torch=str(torch.__version__)),indent=2)+'\n')
# Original graph using the approved shared node grammar.
def text(x,y,s,size=26,extra=''):return f'<text x="{x}" y="{y}" class="network-label" style="font-size:{size}px" {extra}>{s}</text>'
def node(x,y,w,h,label,kind):return f'<rect class="network-node network-{kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="15"/>'+text(x+w/2,y+h/2+9,label,27,'text-anchor="middle"')
def edge(path):return f'<path class="network-edge" d="{path}" marker-end="url(#loss-arrow)"/>'
body=''
for y,source,p,target,formula in [(70,'Real x','0.80','1','−ln(0.80) = 0.223'),(285,'Generated x̃','0.20','0','−ln(1 − 0.20) = 0.223')]:
    body+=node(25,y,230,100,source,'value')+edge(f'M273 {y+50}H325')+node(343,y,155,100,'D','discriminator')+edge(f'M516 {y+50}H568')
    body+=text(590,y+40,'p = '+p,32)+text(590,y+82,'p = sigmoid(logit)',20)
    body+=f'<g class="fragment" data-fragment-index="0">'+text(870,y+25,'Target '+target,24)+text(870,y+78,formula,27)+'</g>'
body+=text(28,455,'Same D · source labels come from how the image was obtained',25)
svg='<div class="course-network-diagram gan-loss-graph"><svg viewBox="0 0 1280 485" role="img" aria-label="Two images enter the same discriminator; real has target one, generated has target zero. Illustrative scores 0.8 and 0.2."><defs><marker id="loss-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 9 5 1 9" fill="none" stroke="#526575" stroke-width="1.5"/></marker></defs>'+body+'</svg></div>'
(ROOT/'loss-graph.html').write_text(svg)
# Analytic gradient curves; the dots are independently verified with PyTorch autograd.
body=text(90,34,'Gradient magnitude at the discriminator logit',27)
for val in [0,.25,.5,.75,1]:
    y=390-val*300
    body+=f'<path d="M90 {y}H1050" stroke="#d8dfe2"/>'+text(70,y+7,f'{val:g}',20,'text-anchor="end"')
for a in [-6,-4,-2,0,2,4,6]:body+=text(90+(a+6)*80,428,str(a),21,'text-anchor="middle"')
body+=text(570,475,'Discriminator logit a on a generated image',24,'text-anchor="middle"')
for color,ns in [('#c44834',False),('#2878c8',True)]:
    points=[]
    for k in range(241):
        a=-6+k/20;p=1/(1+math.exp(-a));v=1-p if ns else p
        points.append(f'{90+(a+6)*80:.2f},{390-v*300:.2f}')
    body+=f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="5"/>'
body+='<line id="gradient-cursor" x1="202.39" x2="202.39" y1="80" y2="390" stroke="#526575" stroke-dasharray="7 7" stroke-width="2"/>'
body+='<circle id="gradient-minimax" cx="202.39" cy="387" r="8" fill="#c44834"/><circle id="gradient-nonsat" cx="202.39" cy="93" r="8" fill="#2878c8"/>'
(ROOT/'gradient-plot.html').write_text('<div class="course-network-diagram gan-gradient-plot"><svg viewBox="0 0 1120 490" role="img" aria-label="Gradient magnitude versus discriminator logit. Minimax has weak gradient for confidently rejected generated images; the non-saturating loss has a strong gradient there.">'+body+'</svg></div>')
source=Path(__file__).read_text();code=source.split('# BEGIN LOSSES\n')[1].split('# END LOSSES')[0].strip()
# Hide only the target choices, preserving a clean single code surface.
lines=html.escape(code)
for token in ['torch.ones_like(real_logits)','torch.zeros_like(fake_logits)','torch.ones_like(fake_logits)']:
    lines=lines.replace(token,'<span class="gan-target-blank"><span class="gan-question">?</span><span class="fragment" data-fragment-index="0">'+token+'</span></span>')
(ROOT/'loss-code.html').write_text('<div class="light-code-editor"><div class="editor-bar"><span>PYTORCH · LOSS VALUES</span><strong>targets match the logits</strong></div><pre><code>'+lines+'</code></pre></div>')
print(json.dumps(dict(loss_D=loss_D.item(),loss_G=loss_G.item(),gradient_checks='5 passed'),indent=2))
