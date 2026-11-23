"""Read-only scoring of Bald sweep outputs. Source-derived scene masks, not edit masks."""
import json, sys, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import center_crop, to_tensor, to_pil_image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, CLIPModel, CLIPProcessor

root=Path('/work/nvme/bhje/hyu29')
d=root/'datasets/celebA'
s=root/'evaluations/rebuttal_cfg_20261118_v1'
out=s/'bald_clip_psnr_v1'
out.mkdir(exist_ok=True)
smoke=sys.argv[1]=='smoke'
plans=[json.loads((s/f'Bald_{j}.json').read_text())['male_indices'] for j in range(2)]
pool=sum(plans,[])
assert len(pool)==len(set(pool))==512
qa=random.Random(42).sample(plans[0],30)
use=qa if smoke else pool
names=sorted(x.split()[0] for x in (d/'list_eval_partition.txt').read_text().splitlines() if x.split()[1]=='2')
lines=(d/'list_attr_celeba.txt').read_text().splitlines()
k=1+lines[1].split().index('Bald')
bald={v[0]:int(v[k]) for v in map(str.split,lines[2:])}
segid='jonathandinu/face-parsing'; clipid='openai/clip-vit-large-patch14'
sp=AutoImageProcessor.from_pretrained(segid)
seg=AutoModelForSemanticSegmentation.from_pretrained(segid).eval().cuda()
assert seg.config.id2label[0].lower()=='background'
cp=CLIPProcessor.from_pretrained(clipid)
cm=CLIPModel.from_pretrained(clipid).eval().cuda()
prompts=['A portrait of a person with hair on their head.','A portrait of a bald person.']
refs={}; masks={}; selected=set(use)
with torch.inference_mode():
    for i in range(0,len(use),8):
        ids=use[i:i+8]
        ims=[center_crop(Image.open(d/'img_align_celeba'/names[x]).convert('RGB'),150).resize((256,256),Image.Resampling.BILINEAR) for x in ids]
        logits=seg(**sp(images=ims,return_tensors='pt').to('cuda')).logits
        bg=(F.interpolate(logits,(64,64),mode='bilinear',align_corners=False).argmax(1)==0).float()
        bg=(1-F.max_pool2d((1-bg)[:,None],3,1,1))[:,0].bool().cpu()
        for x,im,b in zip(ids,ims,bg):
            refs[x]=to_tensor(im.resize((64,64),Image.Resampling.BICUBIC));masks[x]=b
    if smoke:
        grid=Image.new('RGB',(8*128,4*64))
        for j,x in enumerate(qa):
            a=refs[x];v=a.clone();v[1,masks[x]]=1
            p=(j%8*128,j//8*64);grid.paste(to_pil_image(a),p);grid.paste(to_pil_image(v),(p[0]+64,p[1]))
        grid.save(out/'qa_30.png')
    for method in ([0] if smoke else [0,1]):
        for scale in ([1] if smoke else [1,2,3,5,7.5]):
            rows={}
            for j in ([0] if smoke else [0,1]):
                directory=s/'formal/celeba'/str(method)/str(scale)/str(6+j)
                seen=[]
                for f in sorted((directory/'tensors').glob('batch_*.pt')):
                    t=torch.load(f,map_location='cpu',weights_only=False)
                    ids=t['source_indices'].tolist();seen+=ids
                    take=[p for p,x in enumerate(ids) if x in selected]
                    if not take: continue
                    ims=t['images'][take].float()
                    assert ims.shape[1:]==(3,64,64) and ims.min()>=0 and ims.max()<=1
                    batch=cp(text=prompts,images=[to_pil_image(z) for z in ims],return_tensors='pt',padding=True).to('cuda')
                    res=cm(**batch)
                    cos=F.normalize(res.image_embeds) @ F.normalize(res.text_embeds).T
                    for u,p in enumerate(take):
                        x=ids[p];target=int(bald[names[x]]==-1)
                        assert (float(torch.as_tensor(t['targets']['Bald']).flatten()[p])>0)==bool(target)
                        b=masks[x]; frac=b.float().mean().item()
                        mse=(ims[u]-refs[x]).square()[:,b].mean().item() if b.any() else None
                        db=-10*math.log10(max(mse,1e-12)) if frac>=.05 else None
                        rows[x]=dict(filename=names[x],target=target,background_fraction=frac,background_psnr=db,clip_cosine=cos[u,target].item())
                assert sorted(seen)==sorted(plans[j]),directory
            assert set(rows)==selected
            ok=[v for v in rows.values() if v['background_psnr'] is not None]
            assert ok,'no valid scene background'
            record=dict(clip_model=clipid,parser=segid,captions=prompts,n=len(rows),n_valid=len(ok),mean_background_psnr=np.mean([v['background_psnr'] for v in ok]),mean_clip_cosine_same_subset=np.mean([v['clip_cosine'] for v in ok]),rows=rows)
            (out/f'{"smoke" if smoke else "full"}_m{method}_cfg{scale}.json').write_text(json.dumps(record,allow_nan=False))
