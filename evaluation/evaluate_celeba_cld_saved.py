"""Compute benchmark VAE-based CLD/minimality from saved CelebA shards."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from data.celeba_dataset import CelebADataset

ATTRS = ("Young", "Male", "No_Beard", "Bald")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-root", required=True)
    ap.add_argument("--method", required=True, choices=("ca", "label_only"))
    ap.add_argument("--celeba-root", required=True)
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--vae-checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    bench = Path(args.benchmark_root)
    sys.path.insert(0, str(bench))
    from models.vaes import CelebaCondVAE
    import importlib.util
    spec = importlib.util.spec_from_file_location("benchmark_minimality", bench / "evaluation/metrics/minimality.py")
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    minimality, kl_divergence = metric_module.minimality, metric_module.kl_divergence

    params = {"latent_dim":16, "hidden_dim":256, "n_chan":[3,32,64,128,256,256],
              "beta":5, "lr":0.0005, "weight_decay":0, "fixed_logvar":"False"}
    attr_size = {a:1 for a in ATTRS}
    vae = CelebaCondVAE(params, attr_size, unconditional=True).eval().cuda()
    ck = torch.load(args.vae_checkpoint, map_location="cpu")
    vae.load_state_dict(ck["state_dict"])

    records = []
    for shard_dir in sorted((Path(args.saved_root)/args.method).glob("plan_*_*"),
                            key=lambda p:int(p.name.split("_")[1])):
        attr = next(a for a in ATTRS if shard_dir.name.endswith("_"+a))
        for pt in sorted((shard_dir/"tensors").glob("batch_*.pt")):
            obj = torch.load(pt, map_location="cpu")
            for image, real, index, factual, cf in zip(obj["images"].float(), obj["real_images"].float(),
                    obj["source_indices"].tolist(), obj["factual_labels"], obj["counterfactual_labels"]):
                records.append((int(index), attr, image, real, factual, cf))
    if not records: raise RuntimeError("No saved tensors found")
    if len({r[0] for r in records}) != len(records): raise RuntimeError("Duplicate source indices")
    if {r[0] for r in records} != set(range(19962)):
        raise RuntimeError('Incomplete full-test source coverage')
    if args.smoke:
        records = records[:32]

    factual_feats, cf_feats, factual_labels, cf_labels, interventions = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start+args.batch_size]
            real = torch.stack([x for _,_,_,x,_,_ in chunk]).cuda()
            fake = torch.stack([x for _,_,x,_,_,_ in chunk]).cuda()
            rmu, rlog = vae.encoder(real, None); fmu, flog = vae.encoder(fake, None)
            factual_feats.append(torch.stack([rmu,rlog],dim=1).cpu().numpy())
            cf_feats.append(torch.stack([fmu,flog],dim=1).cpu().numpy())
            for _, attr, _, _, factual, cf in chunk:
                j = ATTRS.index(attr)
                factual_labels.append([int(float(factual[j]) > 0)])
                cf_labels.append([int(float(cf[j]) > 0)])
                interventions.append(attr)

    rf = np.concatenate(factual_feats); ff = np.concatenate(cf_feats)
    factual_labels = np.asarray(factual_labels)
    cf_labels = np.asarray(cf_labels)
    intervention_names = np.asarray(interventions)
    scores = np.full(len(records), np.nan, dtype=np.float64)
    prob1 = np.full(len(records), np.nan, dtype=np.float64)
    prob2 = np.full(len(records), np.nan, dtype=np.float64)
    # Benchmark minimality() groups reference images by a scalar parent value.
    # Pooling all four binary attributes would mix, e.g., Male=1 with Bald=1.
    # Invoke it separately for each intervention, preserving its implementation.
    for attr in ATTRS:
        ids = np.flatnonzero(intervention_names == attr)
        if len(ids) == 0:
            continue
        if len(np.unique(factual_labels[ids, 0])) != 2:
            raise RuntimeError(f'{attr}: both factual reference classes are required')
        factuals = list(zip(rf[ids], factual_labels[ids]))
        counterfactuals = list(zip(ff[ids], cf_labels[ids]))
        attr_scores, attr_prob1, attr_prob2 = minimality(
            factuals, counterfactuals, [attr] * len(ids), None, 'vae'
        )
        if len(attr_scores) != len(ids):
            raise RuntimeError(f'{attr}: CLD skipped samples; score alignment is invalid')
        scores[ids] = attr_scores
        prob1[ids] = attr_prob1
        prob2[ids] = attr_prob2
    if not np.isfinite(scores).all():
        raise RuntimeError('CLD left samples without a score')
    raw_kl = kl_divergence(ff, rf)
    per_attr = {}
    for attr in ATTRS:
        ids = intervention_names == attr
        attr_scores = scores[ids]
        per_attr[attr] = {"n":int(ids.sum()), "raw_vae_kl_mean":float(raw_kl[ids].mean()),
                          "raw_vae_kl_std":float(raw_kl[ids].std()),
                          "cld_minimality_mean":float(attr_scores.mean()) if len(attr_scores) else None}
    out = {"method":args.method, "n":len(records), "vae_checkpoint":args.vae_checkpoint,
           "smoke_only":args.smoke,
           "official_code_compatibility":"uses benchmark minimality.py and its KL implementation unchanged",
           "reference_grouping":"separate factual reference pool for each intervention attribute",
           "cld_minimality_mean":float(np.mean(scores)), "cld_minimality_std":float(np.std(scores)),
           "prob1_mean":float(np.mean(prob1)), "prob2_mean":float(np.mean(prob2)),
           "raw_vae_kl_mean":float(np.mean(raw_kl)), "raw_vae_kl_std":float(np.std(raw_kl)),
           "per_intervention":per_attr}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))

if __name__ == "__main__": main()
