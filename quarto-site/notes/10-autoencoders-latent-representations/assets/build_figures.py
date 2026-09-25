"""Rebuild notes figures from the approved lecture's measured data (no training)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parents[2] / 'lectures/10-autoencoders-latent-representations/assets'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
COLORS = ['#2878c8','#ef5b45','#008e9b','#7655ad','#d58b16','#3c8d63','#c44f86','#586a78','#9a6b35','#00a3b4']

def save(fig, name):
    fig.savefig(ROOT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def cells(file, columns):
    a = np.array(Image.open(file))
    return [a[y:y+28,x:x+28] for y in range(0,a.shape[0],28) for x in range(0,columns*28,28)]

def gallery(rows, labels, name, titles=None):
    fig, axs = plt.subplots(len(rows),len(rows[0]),figsize=(12,2.0*len(rows)),squeeze=False)
    for r,row in enumerate(rows):
        for c,im in enumerate(row):
            axs[r,c].imshow(im,cmap='gray',vmin=0,vmax=255)
            axs[r,c].set_xticks([]); axs[r,c].set_yticks([])
            if c == 0: axs[r,c].set_ylabel(labels[r],fontsize=12)
            if r == 0 and titles: axs[r,c].set_title(titles[c],fontsize=10)
    fig.tight_layout(); save(fig,name)

sweep=json.loads((SRC/'latent-sweep/manifest.json').read_text())
dims=sweep['latent_dimensions']
rows=[]
for sample in [2,5,8]:
    rows.append([np.array(Image.open(SRC/f'latent-sweep/sample-{sample}-input.png'))]+[np.array(Image.open(SRC/f'latent-sweep/d-{d}-sample-{sample}-reconstruction.png')) for d in dims])
gallery(rows,['Pullover','Sandal','Bag'],'width-gallery',['Input']+[f'd = {d}' for d in dims])
fig,axs=plt.subplots(1,3,figsize=(9,3))
for ax,file,title in zip(axs,['sample-2-input.png','d-8-sample-2-reconstruction.png','d-8-sample-2-error.png'],['Input','Reconstruction: d = 8','Absolute residual (0–1)']):
    ax.imshow(Image.open(SRC/'latent-sweep'/file),cmap='gray',vmin=0,vmax=255); ax.set_title(title); ax.axis('off')
fig.tight_layout(); save(fig,'residual')
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(dims,[sweep['models'][str(d)]['test_mse'] for d in dims],'o-',color=COLORS[0])
ax.set_xscale('log',base=2); ax.set_xticks(dims,labels=dims); ax.set(xlabel='Latent coordinates d',ylabel='Test MSE per pixel'); ax.grid(alpha=.2)
save(fig,'width-curve')

geo=json.loads((SRC/'latent-geometry/manifest.json').read_text()); p=geo['points']; z=np.array([v['z'] for v in p]); labels=np.array([v['label'] for v in p])
fig,axs=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
axs[0].scatter(*z.T,s=7,alpha=.5,color='#586a78'); axs[0].set_title('Observe geometry before labels')
for k in range(10): axs[1].scatter(*z[labels==k].T,s=9,alpha=.7,color=COLORS[k],label=geo['classes'][k])
axs[1].legend(fontsize=8,ncol=2,loc='upper right'); axs[1].set_title('Labels added only for inspection')
for ax in axs: ax.set(xlabel='z₁',ylabel='z₂')
fig.tight_layout(); save(fig,'latent-map')
ins=cells(SRC/'latent-geometry/inputs-sprite.png',40); rec=cells(SRC/'latent-geometry/reconstructions-sprite.png',40)
anchor=600; dist=np.linalg.norm(z-z[anchor],axis=1); near=np.argsort(dist)[1:8]; chosen=[anchor]+list(near)
gallery([[ins[i] for i in chosen],[rec[i] for i in chosen]],['Input','Reconstruction'],'neighbors',['Anchor']+[f'{geo["classes"][labels[i]]}\n{dist[i]:.2f}' for i in near])
gallery([cells(SRC/'latent-geometry/interpolation-footwear.png',9),cells(SRC/'latent-geometry/interpolation-tops.png',9)],['Sneaker → boot','Shirt → coat'],'interpolation',[f't = {i/8:g}' for i in range(9)])
gallery([cells(SRC/f'useful-reconstruction/denoise-{s}.png',6) for s in ['noisy','restored','clean']],['Noisy input','Prediction','Clean target'],'denoising')
gallery([cells(SRC/f'useful-reconstruction/anomaly-{s}.png',12) for s in ['inputs','reconstructions']],['Input','Prediction'],'anomalies')
anom=json.loads((SRC/'useful-reconstruction/manifest.json').read_text())['anomaly']
fig,ax=plt.subplots(figsize=(10,3.2))
for i,v in enumerate(anom['gallery']):
    y=(1 if v['anomaly'] else 0)+(i%3-1)*.08
    ax.scatter(v['score'],y,color=COLORS[1] if v['anomaly'] else COLORS[5],s=65)
ax.axvline(anom['threshold_95pct'],color=COLORS[4],ls='--',label='Demonstration percentile: 0.03971')
ax.set_yticks([0,1],labels=['Selected footwear','Selected high-error garments']); ax.set(xlim=(0,.2),ylim=(-.4,1.6),xlabel='Mean squared error per pixel'); ax.legend(loc='upper left',fontsize=9)
fig.tight_layout(); save(fig,'anomaly-scores')
sam=json.loads((SRC/'sampling-failure/manifest.json').read_text())
fig,ax=plt.subplots(figsize=(8,5))
pts=np.array([[v['x'],v['y']] for v in sam['encoded_points']]); rnd=np.array([[v['x'],v['y']] for v in sam['random_points']])
ax.scatter(*pts.T,s=8,c='gray',alpha=.3,label='Encoded test examples')
sc=ax.scatter(*rnd.T,c=[v['distance'] for v in sam['random_points']],s=16,cmap='plasma'); fig.colorbar(sc,ax=ax,label='Distance to nearest displayed code (raw coordinates)')
ax.set(xlabel='Scaled z₁',ylabel='Scaled z₂'); ax.legend(); save(fig,'sampling-support')

# An hourglass uses state heights to encode relative coordinate counts.
fig,ax=plt.subplots(figsize=(12,4)); ax.set_xlim(-.5,12.5); ax.set_ylim(-2.2,2.5); ax.axis('off')
heights=[3.5,2.6,1.6,.65,1.6,2.6,3.5]
for i,h in enumerate(heights):
    x=2*i
    ax.add_patch(FancyBboxPatch((x-.38,-h/2),.76,h,boxstyle='round,pad=.04',color=COLORS[0] if i<3 else COLORS[1] if i==3 else COLORS[3]))
    ax.text(x,0,['x','h₁','h₂','z','h̃₂','h̃₁','x̂'][i],ha='center',va='center',color='white',fontsize=18)
    if i<6: ax.add_patch(FancyArrowPatch((x+.5,0),(x+1.5,0),arrowstyle='-|>',mutation_scale=15,color='#586a78'))
ax.text(2,2.1,'Encoder fθ',ha='center',fontsize=15); ax.text(10,2.1,'Decoder gφ',ha='center',fontsize=15)
save(fig,'hourglass')

# Original text-free chapter cover: real images, contracting states, a code cloud,
# and expanding states ending at reconstructions.
fig=plt.figure(figsize=(12,5),facecolor='#f5f2e9'); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,12),ylim=(0,5)); ax.axis('off')
for i,h in enumerate([3.4,2.4,1.3,.6,1.3,2.4,3.4]):
    x=3+i
    ax.add_patch(FancyBboxPatch((x-.15,2.5-h/2),.3,h,boxstyle='round,pad=.07',color=COLORS[0] if i<3 else COLORS[1] if i==3 else COLORS[3],alpha=.85))
    if i<6: ax.plot([x+.24,x+.76],[2.5,2.5],color='#89959b',lw=2)
for col,images in [(0.035,ins),(0.82,rec)]:
    for j,idx in enumerate([200,700,800]):
        a=fig.add_axes([col,.13+j*.255,.145,.22]); a.imshow(images[idx],cmap='gray'); a.axis('off')
fig.savefig(ROOT/'chapter-cover.png',dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight'); plt.close(fig)
