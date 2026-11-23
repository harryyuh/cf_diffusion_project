"""DeltaAI rebuttal sweep. Run from project root in the cf-celeba environment."""
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

R = Path('/work/nvme/bhje/hyu29')
P = R/'repos/cf-diffusion-celeba'
C = P/'external/Causal-Adapter'
O = R/'evaluations/rebuttal_cfg_20261118_v1'
attrs = ['Young','Male','No_Beard','Bald']
pen = ['pendulum','light','shadow_length','shadow_position']
ref = R/'evals/pendulum/ca_author_official20k_seed42_n256_ddim100/ca/selection.json'

def run(args):
    print(shlex.join(list(map(str,args))),flush=True)
    subprocess.run(list(map(str,args)),check=True)

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2))

def prepare():
    O.mkdir(exist_ok=True)
    assert not (O/'pendulum.py').exists() and not (O/'formal').exists()
    plans=R/'evaluations/celeba/ca_official_fulltest_ddim100/plans/manifest.json'
    rows=json.loads(plans.read_text())['assignment']
    assert len({r['source_index'] for r in rows})==len(rows)==19962
    rng=random.Random(42)
    for a in attrs:
        ids=rng.sample([r['source_index'] for r in rows if r['intervention_attr']==a],512)
        for j in range(2):
            save(O/f'{a}_{j}.json',dict(male_indices=ids[j*256:j*256+256],female_indices=[],intervention_attr=a))
        save(O/f'{a}_smoke.json',dict(male_indices=ids[:4],female_indices=[],intervention_attr=a))
    src=(P/'evaluation/evaluate_pendulum.py').read_text()
    for k,v in [('self.steps = args.steps','self.steps = args.steps\n        self.sweep_cfg = args.guidance_scale'),('start_latents=trajectory[-1].clone(), guidance_scale=1.0','start_latents=trajectory[-1].clone(), guidance_scale=self.sweep_cfg')]:
        assert src.count(k)==1,k
        src=src.replace(k,v)
    compile(src,'pendulum.py','exec')
    (O/'pendulum.py').write_text(src)

def job(ds,m,g,j,smoke):
    out=O/('smoke' if smoke else 'formal')/ds/str(m)/str(g)/str(j)
    out.mkdir(parents=True,exist_ok=True)
    if ds=='celeba':
        a=attrs[j//2]
        sel=O/f'{a}_{("smoke" if smoke else j%2)}.json'
        template=(P/'slurm/evaluation/celeba_ca_label_fulltest_sharded_deltaai.slurm').read_text()
        s=template.split('if [[ "${METHOD}" == ca ]]; then\n')[1].split('\nfi')[0].split('\nelse\n')[m]
        vals=dict(CA_ROOT=C,OFFICIAL=R/'ckpts/celeba/ca_author_official_200k',LABEL=R/'ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000',SEL=sel,COUNT=4 if smoke else 256,ATTR=a,METHOD_OUT=out)
        for k,v in vals.items(): s=s.replace('${'+k+'}',str(v))
        assert '${' not in s
        args=shlex.split(s.replace('\\\n',' '))
        args[args.index('--guidance-scale' if m else '--ca-guidance-scale')+1]=str(g)
        (out/'tensors').mkdir(exist_ok=True)
        run(args)
    else:
        d=R/'models/Causal-Adapter/pendulum'
        method='joint_label' if m else 'ca'
        size=2 if smoke else 32
        args=[sys.executable,O/'pendulum.py','--method',method,'--data-root',R/'datasets/pendulum','--ca-root',C,'--ca-scm',d/'scm/best_model.pt','--regressor',R/'ckpts/pendulum/regressor/best.pt','--steps',100,'--seed',42,'--max-samples',256,'--shard-size',size,'--shard-index',j,'--grid-samples',0,'--guidance-scale',g,'--output-dir',out]
        if m: args+=['--joint-config','configs/joint_label_lora_pendulum_equal_budget_official_protocol.yaml','--joint-root',R/'ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed42/final']
        else: args+=['--base-model','lambdalabs/miniSD-diffusers','--ca-controlnet',d/'controlnet/controlnet-steps-20000.safetensors','--ca-embedding',d/'controlnet/learned_embeds-steps-20000.safetensors']
        run(args)
        sub=out/method
        cur=json.loads((sub/'selection.json').read_text())
        old=json.loads(ref.read_text())
        for key in ['test_indices',*map(str,range(4))]:
            v=lambda x:x[key] if key=='test_indices' else x['intervention_train_indices'][key]
            assert v(cur)==v(old)[j*size:(j+1)*size]
        run([sys.executable,'-m','evaluation.reevaluate_pendulum_saved_ca_regressors','--input-dir',sub,'--data-root',R/'datasets/pendulum','--ca-root',C,'--ca-scm',d/'scm/best_model.pt','--checkpoint-dir',C/'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/checkpoints/pendulum/trained_classifiers','--batch-size',32])
    return out

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare(); sys.exit()
    ds=sys.argv[1]
    gs=[1,2,3,5,7.5] if ds=='celeba' else [1,1.5,2,3,5]
    i=int(os.environ['SLURM_ARRAY_TASK_ID'])
    if sys.argv[2]=='smoke':
        hashes=[]
        for g in ([1,3,7.5] if ds=='celeba' else [1,2,5]):
            p=job(ds,i,g,6 if ds=='celeba' else 0,True)
            if ds=='pendulum':
                b=next(p.glob('*/images/light/*.png')).read_bytes()
            else:
                import torch
                t=torch.load(next((p/'tensors').glob('*.pt')),map_location='cpu',weights_only=False)
                key=next(k for k,v in t.items() if isinstance(v,torch.Tensor) and v.ndim==4 and v.shape[1]==3)
                b=t[key].contiguous().numpy().tobytes()
            hashes.append(hashlib.sha256(b).hexdigest())
        assert len(set(hashes))==3,hashes
    else: job(ds,i//40,gs[(i//8)%5],i%8,False)
