"""Standalone DeltaAI launcher: paired CelebA training and six checkpoint evaluations."""
import os, sys, json, random, shlex, subprocess, hashlib, time, math
from pathlib import Path
R=Path('/work/nvme/bhje/hyu29')
P=R/'repos/cf-diffusion-celeba'
C=P/'external/Causal-Adapter'
D=R/'datasets/celebA'
O=R/'ckpts/celeba/rebuttal_paired20k_v1'
S=R/'evaluations/rebuttal_cfg_20261118_v1'
steps=[1000,2000,5000,10000,15000,20000]

def prepare():
    from collections import Counter
    names={n for n,v in (s.split() for s in (D/'list_eval_partition.txt').read_text().splitlines()) if v=='0'}
    lines=(D/'list_attr_celeba.txt').read_text().splitlines()
    cols=[lines[1].split().index(x)+1 for x in ['No_Beard','Bald']]
    data={r[0]:tuple(r[i] for i in cols) for r in map(str.split,lines[2:]) if r[0] in names}
    ns=sorted(data); assert len(ns)==len(names)==162770
    counts=[Counter(v[j] for v in data.values()) for j in range(2)]
    w=[sum(1/counts[j][data[n][j]] for j in range(2)) for n in ns]
    ids=random.Random(42).choices(ns,weights=w,k=40000)
    O.mkdir(parents=True,exist_ok=False)
    (O/'manifest.json').write_text(json.dumps(ids))
    print('presentations',len(ids),'unique',len(set(ids)),'positive counts',[sum(data[n][j]=='1' for n in ids) for j in range(2)])

def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)

def train(method,smoke):
    import yaml
    import torch
    stage='smoke' if smoke else 'train'
    n=4 if smoke else 20000
    every=2 if smoke else 1000
    out=O/stage/method
    out.mkdir(parents=True,exist_ok=False)
    ids=json.loads((O/'manifest.json').read_text())[:2*n]
    def paired(names):
        lookup={k:i for i,k in enumerate(names)}
        assert len(names)==len(lookup)==162770
        ii=[lookup[k] for k in ids]
        digest=hashlib.sha256('\n'.join(names[i] for i in ii).encode()).hexdigest()
        (out/'pairing.json').write_text(json.dumps(dict(count=len(ii),sha256=digest,first=ids[:8])))
        return ii
    start=time.monotonic()
    def record(step):
        with (out/'time.jsonl').open('a') as f:
            print(json.dumps(dict(step=step,elapsed=time.monotonic()-start,gpu=torch.cuda.get_device_name())),file=f)
    if method=='label':
        cfg=yaml.safe_load((P/'configs/joint_label_lora_celeba_20k_bs2_seed42_eval_deltaai.yaml').read_text())
        cfg.update(run_name='model',output_dir=str(out),save_every=every,max_train_steps=n,random_horizontal_flip=False,lambda_ctc=0.,log_every=1 if smoke else 100)
        cf=out/'config.yaml'; cf.write_text(yaml.safe_dump(cfg))
        path=P/'training/train_joint_label_lora_celeba.py'
        s=path.read_text()
        s=replace(s,'    if sampler is not None:', '    sampler = paired(dataset._filenames)\n    if sampler is not None:')
        s=replace(s,'                    save_hybrid_checkpoint','                    record(global_step)\n                    save_hybrid_checkpoint')
        argv=['--config',str(cf)]
        os.chdir(P)
    else:
        path=C/'train.py'
        s=replace(path.read_text(),'sampler = build_balanced_sampler(args.dataset, train_dataset)',"""from torchvision.transforms import RandomHorizontalFlip
    for transform in train_dataset.image_transforms.transforms:
        if isinstance(transform, RandomHorizontalFlip): transform.p = 0.0
    sampler = paired(train_dataset.adapter.data.filename)""")
        s=replace(s,'if global_step % args.save_steps == 0:', 'if global_step % args.save_steps == 0:\n                        record(global_step)')
        argv=shlex.split(f'--seed 42 --output_name paired20k --output_dir {out} --pretrained_model_name_or_path lambdalabs/miniSD-diffusers --train_data_dir {D} --dataset celeA_complex --resolution 256 --train_batch_size 2 --gradient_accumulation_steps 1 --max_train_steps {n} --learning_rate 5e-6 --lr_scheduler constant --lr_warmup_steps 0 --adam_beta1 0.9 --adam_beta2 0.999 --adam_weight_decay 0.01 --adam_epsilon 1e-8 --max_grad_norm 1 --mixed_precision no --save_steps {every} --checkpointing_steps 999999 --placeholder_string "a human is @ and * and & and #" --presudo_words "@,*,&,#" --presudo_words_infonce "@,*,&,#" --infonce_temperature 0.2 --infonce_scale 0.0005 --task_cond generation_text_global_after --scm_path {R}/ckpts/celeba/causal_adapter_scm/best_model.pt')
        os.chdir(C)
    (out/'launcher_snapshot.py').write_text(s)
    sys.argv=[str(path),*argv]; sys.path.insert(0,str(path.parent))
    exec(compile(s,str(path),'exec'),dict(__name__='__main__',__file__=str(path),paired=paired,record=record))
    if method=='ca':
        import ast
        losses,=out.rglob('logs.txt')
        last=losses.read_text().splitlines()[-1]
        assert last.startswith(f'Step {n}:') and math.isfinite(ast.literal_eval(last.split(': ',1)[1])['recons_loss'])
    for k in range(every,n+1,every):
        if method=='ca':
            for prefix in ('controlnet','learned_embeds'):
                assert len(list(out.rglob(f'{prefix}-steps-{k}.safetensors')))==1
        else:
            q=out/f'model/checkpoints/step_{k:06d}'
            assert (q/'mapper/pai_mapper.pt').is_file() and any((q/'lora').glob('*'))

def evaluate(i):
    method=['ca','label'][i//48]; step=steps[i//8%6]; j=i%8
    a=['Young','Male','No_Beard','Bald'][j//2]
    out=O/'eval'/method/str(step)/str(j); out.mkdir(parents=True,exist_ok=True)
    src=(P/'slurm/evaluation/celeba_ca_label_fulltest_sharded_deltaai.slurm').read_text()
    s=src.split('if [[ "${METHOD}" == ca ]]; then\n')[1].split('\nfi')[0].split('\nelse\n')[int(method=='label')]
    train=O/'train'/method
    vals=dict(CA_ROOT=C,OFFICIAL=R/'ckpts/celeba/ca_author_official_200k',LABEL=train/f'model/checkpoints/step_{step:06d}',SEL=S/f'{a}_{j%2}.json',COUNT=256,ATTR=a,METHOD_OUT=out)
    for k,v in vals.items(): s=s.replace('${'+k+'}',str(v))
    args=shlex.split(s.replace('\\\n',' '))
    if method=='ca':
        for flag,prefix in [('--ca-controlnet-path','controlnet'),('--ca-text-embedding-path','learned_embeds')]:
            q,=train.rglob(f'{prefix}-steps-{step}.safetensors')
            args[args.index(flag)+1]=str(q)
    (out/'tensors').mkdir(exist_ok=True)
    subprocess.run(args,cwd=P,check=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='eval': evaluate(int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:
        method=['ca','label'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
        if sys.argv[1]=='train':
            assert (O/'smoke/ca/pairing.json').read_bytes()==(O/'smoke/label/pairing.json').read_bytes()
        train(method,sys.argv[1]=='smoke')
