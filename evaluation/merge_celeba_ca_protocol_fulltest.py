"""Merge CA-official full-test outputs from sample-level saved data."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset

DATA = Path("/work/nvme/bhje/hyu29/datasets/celebA")
ATTRS = ("Young", "Male", "No_Beard", "Bald")
EXPECTED = 19962

class OfficialRealTrain(Dataset):
    def __init__(self):
        ds = CelebADataset(str(DATA), "train", CELEBA_ATTR_ORDER, image_size=64)
        self.names = list(ds._filenames)
        # Match CA: dataset_SD CenterCrop(150)->BILINEAR 256, followed by
        # ddim_modules.resize_tensor(..., mode='bicubic') to 64.
        self.tf = transforms.Compose([
            transforms.CenterCrop(150),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        return self.tf(Image.open(DATA / "img_align_celeba" / self.names[i]).convert("RGB"))

def pred_labels(x):
    x = np.asarray(x)
    # CelebaComplexClassifier.forward returns logits, including when all
    # observed values happen to lie between zero and one.
    return x > 0.0

def f1(pred, target):
    p, t = pred_labels(pred).reshape(-1), (np.asarray(target).reshape(-1) > 0.5)
    tp, fp, fn = np.logical_and(p,t).sum(), np.logical_and(p,~t).sum(), np.logical_and(~p,t).sum()
    d = 2*tp + fp + fn
    return {"f1": float(2*tp/d) if d else 0.0, "tp": int(tp), "fp": int(fp), "fn": int(fn), "n": int(len(t))}

def records(root, method):
    files = sorted((root / method).glob("plan_*_*/tensors/batch_*.pt"))
    for path in files:
        obj = torch.load(path, map_location="cpu")
        n = len(obj["images"])
        if not all(k in obj for k in ("real_images", "predictions", "targets", "intervention_attr")):
            raise RuntimeError(f"legacy/incomplete tensor shard: {path}")
        yield path, obj, n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--root', type=Path, default=Path('/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root
    manifest_path = args.manifest or (root / 'plans/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["assignment"]) == EXPECTED
    device = torch.device("cuda")
    real_loader = DataLoader(OfficialRealTrain(), batch_size=128, num_workers=8, pin_memory=True)
    result = {"protocol": {
        "implementation": "CA official full-test protocol with frozen seed-42 batch intervention manifest",
        "real_reference": "complete CelebA train split",
        "real_preprocess": "CenterCrop150 -> bilinear256 -> bicubic64",
        "fake_cohort": "complete 19,962-image test split; one intervention per source",
        "ddim_steps": 100,
        "global_f1": "recomputed once from concatenated sample-level predictions and targets",
    }}
    for method in ("ca", "label_only"):
        fid_metric = FrechetInceptionDistance(normalize=True, reset_real_features=False).set_dtype(torch.float32).to(device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        for real_idx, real in enumerate(real_loader):
            fid_metric.update(real.to(device, non_blocking=True), real=True)
            if args.smoke:
                break
        preds = {a: [] for a in ATTRS}; targets = {a: [] for a in ATTRS}
        by_do = {d: {a: [[], []] for a in ATTRS} for d in ATTRS}
        per_intervention = {a: 0 for a in ATTRS}; seen = []; count = 0
        for _, obj, n in records(root, method):
            fake = obj["images"].float().to(device)
            real = obj["real_images"].float().to(device)
            fid_metric.update(fake, real=False)
            lpips_metric.update(fake, real)
            for a in ATTRS:
                preds[a].append(torch.as_tensor(obj["predictions"][a]).cpu().numpy())
                targets[a].append(torch.as_tensor(obj["targets"][a]).cpu().numpy())
                by_do[str(obj['intervention_attr'])][a][0].append(preds[a][-1])
                by_do[str(obj['intervention_attr'])][a][1].append(targets[a][-1])
            per_intervention[str(obj["intervention_attr"])] += n
            seen.extend(torch.as_tensor(obj["source_indices"]).tolist())
            count += n
            if args.smoke and count >= 16:
                break
        if not args.smoke and (count != EXPECTED or set(seen) != set(range(EXPECTED))):
            raise RuntimeError(f"{method}: expected {EXPECTED} unique samples, found {count}/{len(set(seen))}")
        result[method] = {
            "num_fake": count,
            "per_intervention_counts": per_intervention,
            "fid": float(fid_metric.compute().cpu()),
            "lpips_identity": float(lpips_metric.compute().cpu()),
            "effectiveness": {a: f1(np.concatenate(preds[a]), np.concatenate(targets[a])) for a in ATTRS},
            "effectiveness_by_do": {d: {a: f1(np.concatenate(p), np.concatenate(t))
                for a, (p, t) in values.items() if p} for d, values in by_do.items()},
        }
    result['smoke_only'] = args.smoke
    out = args.output or (root / ('smoke_metrics_fid_f1_lpips.json' if args.smoke else 'merged_metrics_fid_f1_lpips.json'))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
