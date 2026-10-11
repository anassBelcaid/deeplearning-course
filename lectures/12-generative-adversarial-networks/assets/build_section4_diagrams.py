"""Original section 4 node diagrams, reusing the shared visual grammar."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def svg(name,body,w=1280,h=500):return f'<div class="course-network-diagram gan-{name}"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{name.replace("-"," ")}"><defs><marker id="{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1 9 5 1 9" fill="none" stroke="context-stroke" stroke-width="1.5"/></marker></defs>{body}</svg></div>'
def txt(x,y,s,size=25,extra=''):return f'<text class="network-label" x="{x}" y="{y}" style="font-size:{size}px" {extra}>{s}</text>'
def node(x,y,w,h,s,kind,extra=''):return f'<g {extra}><rect class="network-node network-{kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="15"/>'+txt(x+w/2,y+h/2+8,s,27,'text-anchor="middle"')+'</g>'
def edge(name,d,color='#526575',extra=''):return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#{name}-arrow)" {extra}/>'
name='update-flow';b=txt(35,58,'Forward: compute a loss',25)
for x,w,label,kind in [(30,100,'z','value'),(210,190,'G','generator'),(495,180,'image','value'),(780,190,'D','discriminator'),(1080,170,'loss','value')]:b+=node(x,190,w,110,label,kind)
for d in ['M148 245H192','M418 245H477','M693 245H762','M988 245H1062']:b+=edge(name,d)
b+='<g data-turn-only="D">'+node(495,60,180,75,'real x','value')+edge(name,'M693 97H875V172')+txt(720,337,'detach',23,'text-anchor="middle"')+'<path d="M725 220V270" stroke="#c44834" stroke-width="5"/></g>'
b+=txt(800,58,'Backward: gradients',25)
b+=edge(name,'M1165 319V385H875V319','#c44834')
b+='<g data-turn-only="G">'+edge(name,'M800 319V385H305V319','#c44834')+'</g>'
b+=txt(305,455,'G fixed',27,'text-anchor="middle" id="turn-g-label"')+txt(875,455,'D updates',27,'text-anchor="middle" id="turn-d-label"')
(ROOT/'update-flow.html').write_text(svg(name,b))
for name,turn in [('d-update-contract','D'),('g-update-contract','G')]:
    b=node(55,65,270,95,'Generator G','generator')+node(55,305,270,95,'Discriminator D','discriminator')
    b+=edge(name,'M190 267V287')
    if turn=='D':
        b+=txt(190,220,'detach()',25,'text-anchor="middle"')+txt(190,255,'stop G gradients',21,'text-anchor="middle"')
        b+=txt(190,35,'G: fixed parameters',23,'text-anchor="middle"')+txt(190,446,'D: backward + step',23,'text-anchor="middle"')
    else:
        b+=edge(name,'M345 354H365V110H343','#c44834')
        b+=txt(190,220,'Keep this graph',25,'text-anchor="middle"')+txt(190,255,'D still differentiates',21,'text-anchor="middle"')
        b+=txt(190,35,'G: backward + step',23,'text-anchor="middle"')+txt(190,446,'D: fixed parameters',23,'text-anchor="middle"')
    b+=txt(190,511,'Changed tensors after step',20,'text-anchor="middle"')+'<g class="fragment" data-fragment-index="0">'+txt(190,550,'G: 0 / 6 · D: 6 / 6' if turn=='D' else 'G: 6 / 6 · D: 0 / 6',25,'text-anchor="middle"')+'</g>'
    (ROOT/(name+'.html')).write_text(svg(name,b,380,580))
print('Built update routing and two code-contract diagrams.')
