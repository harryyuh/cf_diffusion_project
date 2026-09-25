"""Re-evaluate saved Pendulum counterfactual PNGs with CA's four regressors."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from data.pendulum_dataset import PENDULUM_ATTRS, PendulumDataset, denormalize_pendulum
from evaluation.evaluate_pendulum import load_scm, scm_target, raw_to_ca

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir', action='append', required=True,
                   help='Leaf evaluation directory containing images/ and selection.json')
    p.add_argument('--data-root', required=True)
    p.add_argument('--ca-root', required=True)
    p.add_argument('--ca-scm', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    return p.parse_args()

def load_predictors(ca_root, ckpt_dir, device):
    bench=Path(ca_root)/'counterfactual-benchmark'/'counterfactual_benchmark'
    sys.path.insert(0,str(bench))
    from models.classifiers.pendulum_classifier import PendClassifier
    graph={'pendulum':['shadow_length','shadow_position'],
           'light':['shadow_length','shadow_position'],
           'shadow_length':[], 'shadow_position':[]}
    out={}
    for attr in PENDULUM_ATTRS:
        matches=sorted(Path(ckpt_dir).glob(f'{attr}_classifier-*.ckpt'))
        if len(matches)!=1: raise RuntimeError(f'{attr}: expected one checkpoint, found {matches}')
        model=PendClassifier(attr=attr,width=8,num_outputs=1,context_dim=len(graph[attr]),lr=1e-4).to(device)
        payload=torch.load(matches[0],map_location=device)
        model.load_state_dict(payload['state_dict'])
        model.eval(); out[attr]=(model,matches[0])
    return out

def evaluate(root, args, predictors, scm, device):
    root=Path(root); selection=json.loads((root/'selection.json').read_text())
    test=PendulumDataset(args.data_root,'test',image_size=256)
    train=PendulumDataset(args.data_root,'train',image_size=256)
    ids=[int(x) for x in selection['test_indices']]
    tf=transforms.Compose([transforms.Resize((96,96),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor()])
    result={'input_dir':str(root),'n':len(ids),'evaluator':'CA PendClassifier, four independent checkpoints',
            'checkpoint_dir':args.checkpoint_dir,'interventions':{}}
    all_diag=[]
    with torch.no_grad():
      for do_idx,do_name in enumerate(PENDULUM_ATTRS):
        pred_chunks={a:[] for a in PENDULUM_ATTRS}; target_chunks=[]
        train_ids=selection['intervention_train_indices'][str(do_idx)] if isinstance(selection['intervention_train_indices'],dict) and str(do_idx) in selection['intervention_train_indices'] else selection['intervention_train_indices'][do_idx]
        for start in range(0,len(ids),args.batch_size):
            batch_ids=ids[start:start+args.batch_size]
            source_raw=torch.stack([test[i]['raw_factors'] for i in batch_ids]).to(device)
            intervention_raw=source_raw.clone()
            chosen=train_ids[start:start+len(batch_ids)]
            intervention_raw[:,do_idx]=torch.stack([train[int(i)]['raw_factors'][do_idx] for i in chosen]).to(device)
            _,target_raw,_=scm_target(scm,source_raw,do_idx,intervention_raw)
            target_ca=raw_to_ca(target_raw)
            imgs=torch.stack([tf(Image.open(root/'images'/do_name/f'{i:05d}.png').convert('RGB')) for i in batch_ids]).to(device)
            for attr,(model,_) in predictors.items(): pred_chunks[attr].append(model(imgs).reshape(-1).cpu())
            target_chunks.append(target_ca.cpu())
        target=torch.cat(target_chunks)
        normalized={}; raw={}
        for j,attr in enumerate(PENDULUM_ATTRS):
            pred=torch.cat(pred_chunks[attr])
            normalized[attr]=float((pred-target[:,j]).abs().mean())
            # Convert both through CA's Gaussian scale for human-readable raw MAE.
            scale=torch.tensor([42.,44.,4.5,8.])[j]; raw[attr]=float((pred-target[:,j]).abs().mean()*scale)
        result['interventions'][do_name]={'mae_ca_normalized':normalized,'mae_raw':raw,
            'intervened_variable_mae_ca_normalized':normalized[do_name]}
        all_diag.append(normalized[do_name])
    result['mean_intervened_variable_mae_ca_normalized']=float(np.mean(all_diag))
    out=root/'metrics_ca_official_regressors.json'; out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); print('wrote',out)

def main():
    args=parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictors=load_predictors(args.ca_root,args.checkpoint_dir,device)
    scm=load_scm(args.ca_root,args.ca_scm,device)
    for root in args.input_dir: evaluate(root,args,predictors,scm,device)
if __name__=='__main__': main()
