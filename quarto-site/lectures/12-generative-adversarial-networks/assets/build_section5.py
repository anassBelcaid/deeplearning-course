"""Original layer-bar diagrams and exact-code editors for section 5."""
from pathlib import Path
import html
R=Path(__file__).resolve().parent
colors={'dense':'#53758c','convT':'#2878c8','conv':'#b26950','norm':'#458666','activation':'#986caa','reshape':'#71818c'}
def text(x,y,s,size=24,extra=''):return f'<text x="{x}" y="{y}" fill="#17212b" font-family="Manrope,sans-serif" font-size="{size}" {extra}>{s}</text>'
def bar(x,y,w,label,kind,h=48):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{colors[kind]}"/>'+f'<text x="{x+w/2}" y="{y+h/2+8}" text-anchor="middle" fill="white" font-family="Manrope,sans-serif" font-size="23">{label}</text>'
def svg(name,b,w=1280,h=550):return f'<div class="gan-layer-diagram"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{name}">{b}</svg></div>'
b=text(25,35,'Dense generator · pixels appear only at the final reshape',27)
for x,label,kind,shape in [(25,'Linear + ReLU','dense','16 → 128'),(335,'Linear + ReLU','dense','128 → 256'),(645,'Linear + sigmoid','dense','256 → 784'),(955,'Reshape','reshape','1 × 28 × 28')]:
 b+=bar(x,75,265,label,kind)+text(x+132,159,shape,24,'text-anchor="middle"')
 if x<955:b+=text(x+277,109,'→',27)
b+='<g class="fragment" data-fragment-index="0">'+text(25,245,'Convolutional generator · grow a spatial representation',27)
for x,label,shape,k in [(25,'ConvT 7 × 7','64 × 7 × 7',0),(455,'ConvT 4 × 4, s=2','32 × 14 × 14',1),(885,'ConvT 4 × 4, s=2','1 × 28 × 28',2)]:
 b+=bar(x,285,335,label,'convT')
 if k<2:b+=bar(x,338,335,'BatchNorm + ReLU','norm')
 else:b+=bar(x,338,335,'Tanh','activation')
 b+=text(x+167,432,shape,26,'text-anchor="middle"')
 if k<2:b+=text(x+370,320,'→',30)
b+=text(25,501,'Input: 16 × 1 × 1 · Shared local filters at each spatial stage',25)+'</g>'
(R/'conv-comparison.html').write_text(svg('Dense versus convolutional generator structure, excluding the batch axis',b))
source=(R/'conv_gan.py').read_text()
for name in ['G','D']:
 code=source.split('# BEGIN '+name+'\n')[1].split('# END '+name)[0].strip()
 call='z = torch.randn(4, 16, 1, 1)\nfake = G(z)  # (4, 1, 28, 28)' if name=='G' else 'logits = D(fake)  # (4, 1)\n# raw logits for BCEWithLogitsLoss'
 editor='<div class="light-code-editor"><div class="editor-bar"><span>PYTORCH · '+name+'</span><strong>kernel, stride, padding</strong></div><pre><code>'+html.escape(code)+'\n\n<span class="fragment" data-fragment-index="0">'+html.escape(call)+'</span></code></pre></div>'
 (R/f'conv-{name}-code.html').write_text(editor)
 b='';layers=[('16 × 1 × 1','Input'),('64 × 7 × 7','ConvT + BN + ReLU'),('32 × 14 × 14','ConvT + BN + ReLU'),('1 × 28 × 28','ConvT + Tanh')] if name=='G' else [('1 × 28 × 28','Input'),('32 × 14 × 14','Conv + LeakyReLU'),('64 × 7 × 7','Conv + LeakyReLU'),('1 × 1 × 1','Conv → flatten')]
 b+=text(190,30,layers[0][0],26,'text-anchor="middle"')+text(190,70,'↓',27,'text-anchor="middle"')
 for i,(shape,op) in enumerate(layers[1:]):
  y=95+i*145;b+=bar(15,y,350,op,'convT' if name=='G' else 'conv',48)
  b+=text(190,y+84,shape,26,'text-anchor="middle"')
  if i<2:b+=text(190,y+124,'↓',27,'text-anchor="middle"')
 (R/f'conv-{name}-diagram.html').write_text(svg(name+' tensor shapes excluding batch',b,380,530))
# Every numeric cell is rendered as HTML text, not a decorative image.
rows=[('Input 1 × kernel',[1,2,1,'·','·'],''),('Input 2 × kernel',['·','·',2,4,2],''),('Add overlaps',[1,2,3,4,2],'fragment')]
out='<div class="gan-overlap-grid">'
for label,values,cls in rows:
 out+=f'<div class="gan-overlap-row {cls}" data-fragment-index="0"><strong>{label}</strong>'
 for i,v in enumerate(values):out+=f'<span class="gan-overlap-cell {"overlap" if i==2 else ""}">{v}</span>'
 out+='</div>'
out+='</div>'
(R/'transpose-overlap.html').write_text(out)
print('Built architecture comparison, overlap example, and two synchronized code views.')
