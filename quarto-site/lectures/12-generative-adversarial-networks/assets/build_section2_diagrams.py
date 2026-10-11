"""Rebuild code-synchronized HTML editors and original section 2 SVG diagrams."""
from pathlib import Path
from textwrap import dedent
import html, re
ROOT=Path(__file__).resolve().parent

def svg(name,body,height=550):
    return f'''<div class="course-network-diagram gan-{name}"><svg viewBox="0 0 1280 {height}" role="img" aria-labelledby="{name}-title"><title id="{name}-title">{html.escape(name.replace('-',' '))}</title><defs><marker id="{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 9 5 1 9" fill="none" stroke="#526575" stroke-width="1.5"/></marker></defs>{body}</svg></div>'''

def node(x,y,w,h,label,kind='value',detail='',attrs=''):
    return f'<g {attrs}><rect class="network-node network-{kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="15"/><text class="network-label" x="{x+w/2}" y="{y+h/2-(6 if detail else -8)}" text-anchor="middle" font-weight="700">{label}</text>'+ (f'<text class="network-detail" x="{x+w/2}" y="{y+h/2+27}" text-anchor="middle">{detail}</text>' if detail else '')+'</g>'
def edge(name,path,attrs=''):
    return f'<path class="network-edge" d="{path}" marker-end="url(#{name}-arrow)" {attrs}/>'
def txt(x,y,text,size=24,attrs=''):
    return f'<text class="network-label" x="{x}" y="{y}" style="font-size:{size}px" {attrs}>{text}</text>'
def tile(x,y,file,size=90):return f'<image href="assets/evidence/section1/{file}" x="{x}" y="{y}" width="{size}" height="{size}"/>'
name='architecture-comparison';body=txt(35,32,'VAE · training path',26)
body+=tile(40,82,'gallery-0.png',110)+txt(95,225,'x',25,'text-anchor="middle"')
body+=node(205,91,175,94,'Encoder','encoder','μ, log variance')
body+=node(432,91,142,94,'Sample z','value')
body+=node(644,91,218,94,'Decoder','generator')
body+=node(941,91,110,94,'x̂')+txt(996,225,'Reconstruction',23,'text-anchor="middle"')
for path in ['M166 138H189','M396 138H416','M590 138H628','M878 138H925']:body+=edge(name,path)
body+=txt(1110,127,'Compare',23)+txt(1110,157,'with x',23)
body+='<g class="fragment" data-fragment-index="0">'+txt(35,304,'GAN · training paths',26)
body+=node(432,435,142,94,'Random z')+node(644,435,218,94,'Generator','generator')
body+=node(923,439,86,86,'x̃')+txt(966,558,'Generated',21,'text-anchor="middle"')
body+=tile(923,300,'gallery-0.png',86)+txt(890,326,'Real x',21,'text-anchor="end"')
body+=node(1100,389,155,142,'Judge','discriminator','shared D')
for path in ['M590 482H628','M878 482H907','M1025 343H1050V425H1084','M1025 482H1084']:body+=edge(name,path)
body+='</g>'
(ROOT/'architecture-comparison.html').write_text(svg(name,body,580))

name='routing';body=''
body+=node(35,52,205,91,'Real images','value','(4, 1, 28, 28)',attrs='data-path="real"')
body+=node(35,246,205,91,'Random z','value','(4, 16)',attrs='data-path="generator"')
body+=node(332,246,203,91,'G','generator',attrs='data-path="generator"')
body+=node(638,246,205,91,'Image batch','value',attrs='data-path="generator"')
body+=node(963,141,196,108,'D','discriminator',attrs='data-path="judge"')
body+=txt(1061,297,'One raw logit / image',23,'text-anchor="middle" data-path="judge"')
for path,kind in [('M258 97H863Q887 97 887 120V175H945','real'),('M258 291H314','generator'),('M553 291H620','generator'),('M861 291H883Q902 291 902 274V216H945','fake')]:body+=edge(name,path,f'data-path="{kind}"')
body+='<g class="fragment" data-fragment-index="0">'
body+=txt(740,373,'(4, 1, 28, 28)',25,'text-anchor="middle" data-path="generator"')
body+=txt(1061,335,'(4, 1)',25,'text-anchor="middle" data-path="judge"')
body+='</g>'
(ROOT/'routing.html').write_text(svg(name,body,410))

# Code editors are extracted from the executable source, preserving one source of truth.
source=(ROOT/'gan_modules.py').read_text()
def section(marker):return dedent(source.split('# BEGIN '+marker+'\n')[1].split('# END '+marker)[0]).strip()
def highlight(code):
    text=html.escape(code)
    text=re.sub(r'\b(import|from)\b',r'<span class="py-kw">\1</span>',text)
    return text
for key,label,forward in [('GENERATOR','GENERATOR','FORWARD_G'),('DISCRIMINATOR','DISCRIMINATOR','FORWARD_D')]:
    setup=section(key)
    if key=='GENERATOR':setup='import torch\nfrom torch import nn\n\n'+setup
    call=section(forward)
    # Display the measured print result without inserting it into executable Python.
    if key=='GENERATOR':call+='\n# torch.Size([4, 1, 28, 28])'
    else:call=call.replace('    ','')+'\n# torch.Size([4, 1])'
    out='<div class="light-code-editor"><div class="editor-bar"><span>PYTORCH · '+label+'</span><strong>untrained module contract</strong></div><pre><code>'+highlight(setup)+'\n\n<span class="fragment" data-fragment-index="0">'+highlight(call)+'</span></code></pre></div>'
    (ROOT/(key.lower()+'-code.html')).write_text(out+'\n')

name='generator-contract';body=txt(190,32,'z · (4, 16)',24,'text-anchor="middle"')
body+=node(60,95,260,100,'Generator G','generator','16 → 128 → 256 → 784')
body+=edge(name,'M190 48V79')
body+=txt(190,234,'Sigmoid + reshape',22,'text-anchor="middle"')
body+=edge(name,'M190 251V288')
body+='<image href="assets/evidence/section2/fake-0.png" x="112" y="305" width="156" height="156"/>'
body+=txt(190,504,'(4, 1, 28, 28)',24,'text-anchor="middle"')
body+=txt(190,544,'First output · untrained',20,'text-anchor="middle"')
(ROOT/'generator-contract.html').write_text(svg(name,body,570).replace('0 0 1280 570','0 0 380 570'))
name='discriminator-contract';body=''
body+='<image href="assets/evidence/section2/real-0.png" x="25" y="20" width="135" height="135"/>'
body+='<image href="assets/evidence/section2/fake-0.png" x="220" y="20" width="135" height="135"/>'
body+=txt(92,190,'Real',23,'text-anchor="middle"')+txt(287,190,'Generated',23,'text-anchor="middle"')
body+=edge(name,'M92 209V249H155V284')+edge(name,'M287 209V249H225V284')
body+=node(45,300,290,100,'One shared D','discriminator','784 → 256 → 128 → 1')
body+=edge(name,'M190 418V455')
body+=txt(190,496,'Logits · (4, 1)',26,'text-anchor="middle"')
body+=txt(190,539,'Any real value',22,'text-anchor="middle"')
(ROOT/'discriminator-contract.html').write_text(svg(name,body,570).replace('0 0 1280 570','0 0 380 570'))
print('Built four original diagrams and two synchronized code editors.')
