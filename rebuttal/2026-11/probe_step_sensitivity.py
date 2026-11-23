"""Matched CelebA pilot; official 100-step results are read-only."""
import sys, os, json
import numpy as np
import celeba_latent_probe as base

B=base.O
O=B.parent/'celeba_probe_steps_pilot_v1'
O.mkdir(exist_ok=True)

def hsic():
    from scipy.spatial.distance import pdist,squareform
    packs=[np.load(B/f'pack{i}.npz') for i in range(4)]
    ids=np.concatenate([p['id_2'] for p in packs])
    y=np.concatenate([p['y_2'] for p in packs])
    assert len(set(ids))==len(ids)==1000
    assert ids.tolist()==[r[0] for r in json.loads((B/'manifest.json').read_text())[2][:1000]]
    # Delta kernel on the joint categorical label (not four marginal tests).
    L=(y[:,None,:]==y[None,:,:]).all(-1).astype(float)
    def center(k): return k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()
    L=center(L); results={}
    for key in ('z0','zt'):
        z=np.concatenate([p[f'{key}_2'] for p in packs])
        dist=pdist(z,'sqeuclidean'); s2=np.median(dist)
        assert s2>0 and np.isfinite(s2)
        K=center(np.exp(-squareform(dist)/(2*s2)))
        def stat(l):return np.sum(K*l)/(len(y)-1)**2
        obs=stat(L); rng=np.random.default_rng(42); null=[]
        for _ in range(999):
            idx=rng.permutation(len(y));null.append(stat(L[np.ix_(idx,idx)]))
        results[key]=dict(hsic=obs,median_squared_distance=s2,p=(1+sum(v>=obs for v in null))/1000,null95=np.percentile(null,[2.5,97.5]).tolist())
    (O/'hsic100.json').write_text(json.dumps(dict(n=1000,kernels='median RBF; joint label equality',metrics=results),indent=2))

def extract(n,i):
    import torch
    from evaluation import evaluate_latent_prompt_lora_celeba_n2048 as ev
    orig=ev.PromptLoraEditor
    class Editor(orig):
        def __init__(self,*a,**kw):
            a=list(a);assert a[3]==100;a[3]=n
            super().__init__(*a,**kw)
            self.grid=self.scheduler.timesteps.flip(0).cpu().tolist()
            assert self.grid==list(range(0,1000,1000//n)) and 990 in self.grid
        def invert(self,z,pe,ne):
            traj=self.invert_trajectory(z,pe,ne)
            return traj[1+self.grid.index(990)]
    ev.PromptLoraEditor=Editor
    base.O=O/str(n);base.O.mkdir(exist_ok=True)
    manifest=base.O/'manifest.json'
    if not manifest.exists():
        try:manifest.symlink_to(B/'manifest.json')
        except FileExistsError:pass
    base.extract(i)
    with np.load(B/f'pack{i}.npz') as a,np.load(base.O/f'pack{i}.npz') as b:
        for j in range(3):
            for k in ('id','y'):assert np.array_equal(a[f'{k}_{j}'],b[f'{k}_{j}'])
            assert np.allclose(a[f'z0_{j}'],b[f'z0_{j}'],atol=.005,rtol=.005)

def fit(n):
    from sklearn.metrics import roc_auc_score
    base.O=O/str(n);base.probe(4)
    y=np.concatenate([np.load(B/f'pack{i}.npz')['y_2'] for i in range(4)])
    res={}
    for kind in ['linear','mlp']:
        a,b=[np.load(d/f'pred_4_zt_{kind}.npy') for d in [B,base.O]]
        def delta(i):return roc_auc_score(y[i],b[i],average=None)-roc_auc_score(y[i],a[i],average=None)
        rng=np.random.default_rng(42);boot=[]
        for _ in range(999):
            i=rng.integers(len(y),size=len(y))
            if np.all(y[i].min(0)!=y[i].max(0)):boot.append(delta(i))
        res[kind]=dict(delta=delta(np.arange(len(y))).tolist(),ci95=np.percentile(boot,[2.5,97.5],axis=0).tolist())
    (base.O/'paired_delta_vs100.json').write_text(json.dumps(res,indent=2))

if __name__=='__main__':
    if sys.argv[1]=='hsic':hsic()
    elif sys.argv[1]=='extract':extract(int(sys.argv[2]),int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:fit(int(sys.argv[2]))
