"""Reproduce the paired paper-style G/D feature-volume architecture view."""
from pathlib import Path
R=Path(__file__).resolve().parent

def text(x,y,s,size=24,anchor='middle',color='#17212b'):
 return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Manrope,sans-serif" font-size="{size}" fill="{color}">{s}</text>'
def volume(cx,cy,size,depth,fill,stroke,label,image=False):
 x,y=cx-size/2,cy-size/2
 out=''
 if depth:
  out+=f'<path d="M{x} {y}l{depth} {-depth/2}h{size}l{-depth} {depth/2}Z" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
  out+=f'<path d="M{x+size} {y}l{depth} {-depth/2}v{size}l{-depth} {depth/2}Z" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
  for t in range(8,depth,8):out+=f'<path d="M{x+size+t} {y-t/2}v{size}" stroke="{stroke}" opacity=".45"/>'
 out+=f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
 if image:
  out+=f'<image href="assets/evidence/section5/epoch-60-sample-0.png" x="{x+2}" y="{y+2}" width="{size-4}" height="{size-4}" style="image-rendering:pixelated"/>'
 elif size>=65:
  for i in range(1,7):
   out+=f'<path d="M{x+i*size/7} {y}v{size}M{x} {y+i*size/7}h{size}" stroke="{stroke}" opacity=".3"/>'
 else:out+=text(cx,cy+8,label,24)
 return out

def arrow(x1,x2,y,op,detail,color):
 mid=(x1+x2)/2
 return f'<path d="M{x1} {y}H{x2}" stroke="{color}" stroke-width="3" fill="none" marker-end="url(#spatial-arrow)"/>'+text(mid,y-29,op,22,color=color)+text(mid,y+35,detail,19)

b='<defs><marker id="spatial-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 9 5 1 9" fill="none" stroke="context-stroke" stroke-width="1.5"/></marker></defs>'
b+=text(25,32,'G · grow spatial resolution',29,'start','#2878c8')
centers=[110,440,790,1140];sizes=[26,65,115,170];depths=[18,48,30,4]
for i,(cx,size,dep,shape) in enumerate(zip(centers,sizes,depths,['16 × 1 × 1','64 × 7 × 7','32 × 14 × 14','1 × 28 × 28'])):
 b+=volume(cx,170,size,dep,'#dceeff','#6c9ec3','z',i==3)+text(cx,298,shape,26)
 if i<3:
  start=cx+size/2+dep+18;end=centers[i+1]-sizes[i+1]/2-18
  b+=arrow(start,end,170,'ConvT', 'k=7, s=1, p=0' if i==0 else 'k=4, s=2, p=1','#2878c8')
b+=text(110,258,'random code',20)+text(440,258,'BN + ReLU',20)+text(790,258,'BN + ReLU',20)+text(1140,327,'Tanh',20)
b+='<g class="fragment" data-fragment-index="0">'+text(25,377,'D · compress spatial evidence',29,'start','#b26950')
sizes=[170,115,65,26];depths=[4,30,48,0]
for i,(cx,size,dep,shape) in enumerate(zip(centers,sizes,depths,['1 × 28 × 28','32 × 14 × 14','64 × 7 × 7','1 logit'])):
 b+=volume(cx,504,size,dep,'#f6e1d9','#bd8d7b','a',i==0)+text(cx,630,shape,26)
 if i<3:
  start=cx+size/2+dep+18;end=centers[i+1]-sizes[i+1]/2-18
  b+=arrow(start,end,504,'Conv2d','k=7, s=1, p=0' if i==2 else 'k=4, s=2, p=1','#b26950')
b+=text(110,659,'real or generated',20)+text(440,591,'LeakyReLU',20)+text(790,591,'LeakyReLU',20)+text(1140,591,'flatten',20)+'</g>'
svg='<div class="gan-spatial-architecture"><svg viewBox="0 0 1280 685" role="img" aria-labelledby="spatial-gan-title spatial-gan-desc"><title id="spatial-gan-title">Generator expands feature maps; discriminator compresses them to a logit</title><desc id="spatial-gan-desc">Generator: 16 by 1 by 1 to 64 by 7 by 7 to 32 by 14 by 14 to a 1 by 28 by 28 image. Discriminator: image to 32 by 14 by 14 to 64 by 7 by 7 to one logit. Cuboid faces suggest spatial size, stack depth suggests channels; dimensions are labeled exactly and drawing sizes are schematic.</desc>'+b+'</svg></div>'
(R/'spatial-architecture.html').write_text(svg)
print('Built paired spatial architecture view.')
