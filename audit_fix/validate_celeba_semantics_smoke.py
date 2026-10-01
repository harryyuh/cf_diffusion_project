import argparse
from pathlib import Path

import torch


def load_records(root: Path):
    out = {}
    for path in sorted(root.glob("tensors/*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        ids = torch.as_tensor(obj["source_indices"]).reshape(-1).tolist()
        factual = torch.as_tensor(obj["factual_labels"]).float().reshape(-1, 4)
        target = torch.as_tensor(obj["counterfactual_labels"]).float().reshape(-1, 4)
        for idx, f, t in zip(ids, factual, target):
            out[int(idx)] = (f, t)
    return out


def binary(x: torch.Tensor):
    return x > (0.5 if x.min().item() >= 0 else 0.0)


p = argparse.ArgumentParser()
p.add_argument("--ca-dir", type=Path, required=True)
p.add_argument("--label-dir", type=Path, required=True)
args = p.parse_args()

ca = load_records(args.ca_dir)
label = load_records(args.label_dir)
if not ca:
    raise RuntimeError("CA smoke output is empty")
missing = sorted(set(ca) - set(label))
if missing:
    raise RuntimeError(f"Label reference lacks CA smoke ids: {missing}")
for idx in sorted(ca):
    ca_f, ca_t = ca[idx]
    lo_f, lo_t = label[idx]
    if not torch.equal(binary(ca_f), binary(lo_f)):
        raise RuntimeError(f"Factual mismatch for source {idx}: {ca_f.tolist()} vs {lo_f.tolist()}")
    if not torch.equal(binary(ca_t), binary(lo_t)):
        raise RuntimeError(f"Target mismatch for source {idx}: {ca_t.tolist()} vs {lo_t.tolist()}")
print(f"validated {len(ca)} paired sources: factual and propagated targets match")
