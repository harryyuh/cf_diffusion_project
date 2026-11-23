"""Matched 32-pair smoke, sampling CFG 3, no diffusion."""
import json, random, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop, to_tensor, to_pil_image
from PIL import Image, ImageDraw
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, CLIPModel, CLIPProcessor

R=Path('/work/nvme/bhje/hyu29')
D=R/'datasets/celebA'
S=R/'evaluations/rebuttal_cfg_20261118_v1'
O=S/('bg_smoke_'+os.environ['SLURM_JOB_ID']); O.mkdir()
A=['Young','Male','No_Beard','Bald']
phrases=[('an older person','a young person'),('a woman','a man'),('a person with a beard','a person without a beard'),('a person with hair on their head','a bald person')]
names=sorted(l.split()[0] for l in (D/'list_eval_partition.txt').read_text().splitlines() if l.split()[1]=='2')
lines=(D/'list_attr_celeba.txt').read_text().splitlines()
cols=[1+lines[1].split().index(a) for a in A]
y={r[0]:[int(r[j])>0 for j in cols] for r in map(str.split,lines[2:])}
rng=random.Random(42); used=set(); rows=[]
grid=Image.new('RGB',(4*104*4,8*124),'white'); pen=ImageDraw.Draw(grid)
segid='jonathandinu/face-parsing'; cid='openai/clip-vit-large-patch14'
with torch.inference_mode():
    sp=SegformerImageProcessor.from_pretrained(segid)
    seg=SegformerForSemanticSegmentation.from_pretrained(segid).eval().cuda()
    assert seg.config.id2label[0].lower()=='background'
    cp=CLIPProcessor.from_pretrained(cid); cm=CLIPModel.from_pretrained(cid).eval().cuda()
    for ai,a in enumerate(A):
        plans=[json.loads((S/f'{a}_{j}.json').read_text())['male_indices'] for j in range(2)]
        pool=sum(plans,[]); assert len(set(pool))==512
        ids=[]
        for bit in (False,True):
            ids+=rng.sample([i for i in pool if i not in used and y[names[i]][ai]==bit],4)
        used.update(ids)
        packs=[{},{}]
        for m in range(2):
            for j,plan in enumerate(plans):
                folder=S/'formal/celeba'/str(m)/'3'/str(2*ai+j)
                assert json.loads((folder/'metrics.json').read_text())['indices']==plan
                for ix in set(ids)&set(plan):
                    b,p=divmod(plan.index(ix),4)
                    t=torch.load(folder/'tensors'/f'batch_{b:05}.pt',map_location='cpu',weights_only=True)
                    assert t['source_indices'].flatten()[p].item()==ix
                    assert (float(t['targets'][a].flatten()[p])>0)!=y[names[ix]][ai]
                    packs[m][ix]=t['images'][p].float()
        src=[center_crop(Image.open(D/'img_align_celeba'/names[i]).convert('RGB'),150) for i in ids]
        refs=torch.stack([to_tensor(im.resize((64,64),Image.Resampling.BILINEAR)) for im in src])
        logits=seg(**sp(images=src,return_tensors='pt').to('cuda')).logits
        bg=(F.interpolate(logits,(64,64),mode='bilinear',align_corners=False).argmax(1)==0).float()
        masks=(1-F.max_pool2d((1-bg)[:,None],3,1,1))[:,0].bool().cpu()
        imgs=[refs]+[torch.stack([p[i] for i in ids]) for p in packs]
        assert all(x.shape==(8,3,64,64) and x.min()>=0 and x.max()<=1 for x in imgs)
        out=cm(**cp(text=['A portrait of '+v+'.' for v in phrases[ai]],images=[to_pil_image(x) for batch in imgs for x in batch],padding=True,return_tensors='pt').to('cuda'))
        emb=out.image_embeds.reshape(3,8,-1); texts=out.text_embeds
        for n,i in enumerate(ids):
            orig=int(y[names[i]][ai]); mask=masks[n]; frac=mask.float().mean().item()
            ref=refs[n]; overlay=ref.clone(); overlay[1,mask]=1
            assert (ref-ref).square().sum()==0
            xx=ai*416; yy=n*124
            pen.text((xx,yy),f'{a} {orig}->{1-orig}  {names[i]}  BG {frac:.1%}',fill='black')
            for z,img in enumerate([ref,overlay,imgs[1][n],imgs[2][n]]):
                grid.paste(to_pil_image(img).resize((100,100)),(xx+104*z,yy+18))
            for m in range(2):
                mse=(imgs[1+m][n]-ref).square()[:,mask].mean().item() if frac>=.05 else None
                delta=emb[1+m,n]-emb[0,n]; dt=texts[1-orig]-texts[orig]
                sim=F.cosine_similarity(delta,dt,dim=0).item() if delta.norm()>1e-6 else None
                rows.append(dict(attr=a,source=names[i],target=1-orig,method=['CA','Label-only'][m],background_fraction=frac,masked_mse=mse,psnr=-10*math.log10(mse) if mse and mse>0 else None,directional_clip=sim))
    assert len(rows)==64 and len(used)==32
    grid.save(O/'source_mask_CA_label.png')
    (O/'rows.json').write_text(json.dumps(rows,allow_nan=False,indent=2))
    print('PASS',O,flush=True)
