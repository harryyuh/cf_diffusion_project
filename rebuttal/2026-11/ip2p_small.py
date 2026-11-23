"""Paired graph-free CelebA baseline. Targets are used only for scoring."""
from pathlib import Path
import os, json, random, sys
R=Path('/work/nvme/bhje/hyu29'); S=R/'evaluations/rebuttal_cfg_20261118_v1'
O=R/'evaluations/ip2p_celeba_rebuttal_v1'; P=R/'repos/cf-diffusion-celeba'
A=['Young','Male','No_Beard','Bald']; G=[1.,1.3,1.6,1.9,2.2]
T=[['Make the person look older.','Make the person look younger.'],
   ['Make the person look female.','Make the person look male.'],
   ['Give the person a beard.','Remove the person\'s beard.'],
   ['Give the person hair on their head.','Make the person bald.']]
REV='31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2))
def refrows(m,g,j):
    import torch
    rows={}
    for h in range(2*j,2*j+2):
        for f in sorted((S/f'formal/celeba/{m}/{g}/{h}/tensors').glob('*.pt')):
            z=torch.load(f,map_location='cpu',weights_only=False)
            for p,i in enumerate(z['source_indices'].tolist()):
                assert i not in rows
                rows[i]=([float(z['targets'][a].flatten()[p]) for a in A],[float(z['predictions'][a].flatten()[p]) for a in A])
    assert len(rows)==512
    return rows
if sys.argv[1]=='prepare':
    O.mkdir(exist_ok=False)
    rng=random.Random(42)
    save(O/'ids.json',{a:rng.sample(sum([load(S/f'{a}_{h}.json')['male_indices'] for h in range(2)],[]),128) for a in A})
    sys.exit()
if sys.argv[1]=='merge':
    summary=[]
    for j,a in enumerate(A):
        ids=load(O/'ids.json')[a]; truth=refrows(0,1,j)
        for m,gs in [('ip2p',G),('ca',[1,2,3,5,7.5]),('label',[1,2,3,5,7.5])]:
            for g in gs:
                if m=='ip2p':
                    records=load(O/f'{a}_{g}.json')
                    rows={r['id']:(r['target'],r['logits']) for r in records}
                else: rows=refrows(int(m=='label'),g,j)
                f1=[]
                for k in range(4):
                    tp=fp=fn=0
                    for i in ids:
                        y,logits=rows[i]; assert y==truth[i][0]
                        pred=logits[k]>0; pos=y[k]>0
                        if k==2: pos,pred=not pos,not pred
                        tp+=pred and pos;fp+=pred and not pos;fn+=not pred and pos
                    f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
                nd=[[1],[0],[0,1,3],[0,1,2]][j]
                summary.append(dict(method=m,scale=g,do=a,f1=f1,target=f1[j],preservation=sum(f1[k] for k in nd)/len(nd) if all(f1[k]!=None for k in nd) else None))
    save(O/'paired_summary.json',summary);sys.exit()
import torch
from PIL import Image
from torchvision.transforms.functional import center_crop,to_tensor
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from evaluation.evaluate_latent_prompt_lora_celeba_n2048 import _build_predictors
sm=sys.argv[1]=='smoke'; idx=int(os.getenv('SLURM_ARRAY_TASK_ID','0'));j=idx%4; g=G[idx//4]
pipe=StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix',revision=REV,variant='fp16',torch_dtype=torch.float16).to('cuda')
pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
bench=P/'external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark'
heads=_build_predictors(bench,torch.device('cuda'))
d=R/'datasets/celebA'
names=sorted(x.split()[0] for x in (d/'list_eval_partition.txt').read_text().splitlines() if x.split()[1]=='2')
lines=(d/'list_attr_celeba.txt').read_text().splitlines()
cols=[lines[1].split().index(a)+1 for a in A]
classmap={r[0]:[int(r[k]) for k in cols] for r in map(str.split,lines[2:])}
with torch.inference_mode():
    for j in (range(4) if sm else [j]):
        a=A[j];ids=load(O/'ids.json')[a]; ids=ids[:2] if sm else ids
        truth=refrows(0,1,j); records=[]
        dr=O/('smoke' if sm else 'images')/a/str(g);dr.mkdir(parents=True,exist_ok=True)
        for i in ids:
            target,unused=truth[i];direction=int(classmap[names[i]][j]<0);assert direction==target[j]
            im=center_crop(Image.open(d/'img_align_celeba'/names[i]).convert('RGB'),150).resize((512,512),Image.Resampling.BILINEAR)
            pred=pipe(T[j][direction],image=im,num_inference_steps=50,guidance_scale=7.5,image_guidance_scale=g,generator=torch.Generator(device='cuda').manual_seed(42+i))
            pic=pred.images[0];pic.save(dr/f'{i}.png')
            x=to_tensor(pic.resize((64,64),Image.Resampling.BICUBIC))[None].cuda()
            lab={k:torch.tensor([[float(v)]],device='cuda') for k,v in zip(A,target)}
            logits=[]
            for k in A:
                clf=heads[k];ctx=clf.conditions[k] if clf.conditions is not None else None
                z=clf(x,torch.cat([lab[t] for t in ctx],dim=1)) if ctx is not None else clf(x)
                assert torch.isfinite(z).all();logits.append(z.item())
            records.append(dict(id=i,target=target,logits=logits,instruction=T[j][direction],flagged=bool(pred.nsfw_content_detected[0]) if pred.nsfw_content_detected else False))
        save(O/f'{"smoke_" if sm else ""}{a}_{g}.json',records)
