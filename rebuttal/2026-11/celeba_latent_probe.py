"""Held-out factual inversion probe. No SCM intervention and no CF generation."""
from pathlib import Path
import json, random, sys, os
import numpy as np

R=Path('/work/nvme/bhje/hyu29'); P=R/'repos/cf-diffusion-celeba'
D=R/'datasets/celebA'; O=R/'evaluations/celeba_latent_probe_v1'
A=['Young','Male','No_Beard','Bald']; cfgpath=P/'configs/joint_label_lora_celeba_20k_bs2_seed42_eval_deltaai.yaml'
def prepare():
    parts=[[] for _ in range(3)]
    for line in (D/'list_eval_partition.txt').read_text().splitlines():
        name,i=line.split(); parts[int(i)].append(name)
    lines=(D/'list_attr_celeba.txt').read_text().splitlines(); cols=[1+lines[1].split().index(a) for a in A]
    labels={r[0]:[int(r[k]) for k in cols] for r in map(str.split,lines[2:])}
    rng=random.Random(42); splits=[]
    for files,n in zip(parts,[5000,1000,5000]):
        splits.append([(f,labels[f]) for f in rng.sample(sorted(files),n)])
    assert len({f for split in splits for f,y in split})==11000
    O.mkdir(exist_ok=False);(O/'manifest.json').write_text(json.dumps(splits))
    print('test positives', (np.array([y for f,y in splits[2]])>0).sum(0))

def extract(idx):
    import torch,yaml
    from evaluation.evaluate_latent_prompt_lora_celeba_n2048 import PromptLoraEditor,COMPLEX_ATTRS
    assert list(COMPLEX_ATTRS)==A
    cfg=yaml.safe_load(cfgpath.read_text()); ck=R/'ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000'
    ed=PromptLoraEditor(cfg,ck/'lora',torch.device('cuda'),100,3.,invert_guidance_scale=1.,negative_prompt='ugly, blurry, low res, unrealistic',scheduler_mode='ca',vae_latent_mode='mean',condition_mode='joint_label',pai_checkpoint=ck/'mapper/pai_mapper.pt')
    splits=json.loads((O/'manifest.json').read_text()); meta={}
    with torch.inference_mode():
        for j,split in enumerate(splits):
            h=len(split)//20; rows=split[idx*h:(idx+1)*h]; z0=[]; zt=[]
            for b in range(0,h,4):
                names,ys=zip(*rows[b:b+4])
                y=torch.tensor(ys,device='cuda',dtype=torch.float)
                pe,ne=ed.condition_embeds(y); z=ed.encode_images(D,names); t=ed.invert(z,pe,ne)
                assert z.shape==t.shape and torch.isfinite(z).all() and torch.isfinite(t).all() and not torch.equal(z,t)
                for buf,u in [(z0,z),(zt,t)]:buf.append(u.flatten(1).float().cpu().numpy())
            meta[f'z0_{j}']=np.concatenate(z0);meta[f'zt_{j}']=np.concatenate(zt)
            meta[f'y_{j}']=(np.array([y for f,y in rows])>0).astype(int)
            meta[f'id_{j}']=np.array([f for f,y in rows])
    tmp=O/f'pack{idx}.tmp.npz'; np.savez(tmp,**meta);tmp.replace(O/f'pack{idx}.npz')

def probe(n):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score as auc,average_precision_score as ap
    packs=[np.load(O/f'pack{i}.npz') for i in range(n)]
    def arr(k):return np.concatenate([p[k] for p in packs])
    splits=json.loads((O/'manifest.json').read_text())
    for j,rows in enumerate(splits):
        rows=rows[:len(rows)*n//20]
        assert [f for f,y in rows]==arr(f'id_{j}').tolist()
        assert np.array_equal(np.array([y for f,y in rows])>0,arr(f'y_{j}'))
    y,v,test=[arr(f'y_{j}') for j in range(3)]
    res=dict(n=[len(y),len(v),len(test)],test_positive=test.sum(0).tolist(),attributes=A,metrics={})
    for z in ['z0','zt']:
        train,val,te=[arr(f'{z}_{j}') for j in range(3)]
        norm=StandardScaler().fit(train);train,val,te=[norm.transform(x) for x in [train,val,te]]
        for kind in ['linear','mlp']:
            if kind=='linear':
                pred=[]
                for j in range(4):
                    models=[LogisticRegression(C=c,solver='liblinear',max_iter=500,random_state=42).fit(train,y[:,j]) for c in [.001,.01,.1]]
                    model=max(models,key=lambda m:auc(v[:,j],m.decision_function(val)))
                    pred.append(model.decision_function(te))
                pred=np.array(pred).T
            else:
                model=MLPClassifier(hidden_layer_sizes=(64,),max_iter=40,alpha=.01,random_state=42,batch_size=128)
                model.fit(train,y);pred=model.predict_proba(te)
            rng=np.random.default_rng(42); null=[]; boot=[]
            for i in range(999):
                # Conditional randomization of held-out label rows for a frozen probe.
                yp=test[rng.permutation(len(test))]
                null.append(auc(yp,pred,average=None))
                ix=rng.integers(0,len(test),len(test))
                if np.all(test[ix].min(0)!=test[ix].max(0)):boot.append(auc(test[ix],pred[ix],average=None))
            null=np.array(null); obs=auc(test,pred,average=None)
            res['metrics'][z+'_'+kind]=dict(auc=obs.tolist(),auprc=ap(test,pred,average=None).tolist(),ci95=np.percentile(boot,[2.5,97.5],axis=0).tolist(),null95=np.percentile(null,[2.5,97.5],axis=0).tolist(),permutation_p=((1+(null>=obs).sum(0))/1000).tolist())
            np.save(O/f'pred_{n}_{z}_{kind}.npy',pred)
            (O/f'partial_{n}.json').write_text(json.dumps(res))
    (O/f'result_{n}.json').write_text(json.dumps(res,indent=2))

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='extract':extract(int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:probe(int(sys.argv[2]))
