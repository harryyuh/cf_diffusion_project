from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path("/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100")


def binary(x: torch.Tensor) -> torch.Tensor:
    return x > (0.5 if x.min().item() >= 0 else 0.0)


records = defaultdict(dict)
stats = defaultdict(lambda: {"files": 0, "rows": 0, "values": Counter()})
for method in ("ca", "label_only"):
    for path in sorted((ROOT / method).glob("plan_*_*/tensors/*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        cf = torch.as_tensor(obj["counterfactual_labels"]).float().reshape(-1, 4)
        factual = torch.as_tensor(obj["factual_labels"]).float().reshape(-1, 4)
        ids = torch.as_tensor(obj["source_indices"]).reshape(-1).tolist()
        attr = str(obj["intervention_attr"])
        stats[(method, attr)]["files"] += 1
        stats[(method, attr)]["rows"] += len(ids)
        stats[(method, attr)]["values"].update(float(v) for v in cf.unique().tolist())
        for idx, f, c in zip(ids, factual, cf):
            if idx in records[method]:
                raise RuntimeError(f"duplicate source {idx} in {method}")
            records[method][idx] = (attr, f, c, str(path))

for key in sorted(stats):
    value = stats[key]
    print(key, "files", value["files"], "rows", value["rows"], "values", dict(value["values"]))

common = sorted(set(records["ca"]) & set(records["label_only"]))
print("coverage", {m: len(records[m]) for m in records}, "common", len(common))
by_attr = defaultdict(lambda: {"n": 0, "fact_mismatch": 0, "cf_mismatch": 0})
examples = []
for idx in common:
    ca_attr, ca_f, ca_cf, ca_path = records["ca"][idx]
    lo_attr, lo_f, lo_cf, lo_path = records["label_only"][idx]
    if ca_attr != lo_attr:
        raise RuntimeError(f"attribute mismatch {idx}: {ca_attr} vs {lo_attr}")
    row = by_attr[ca_attr]
    row["n"] += 1
    row["fact_mismatch"] += int(not torch.equal(binary(ca_f), binary(lo_f)))
    mismatch = not torch.equal(binary(ca_cf), binary(lo_cf))
    row["cf_mismatch"] += int(mismatch)
    if mismatch and len(examples) < 20:
        examples.append((idx, ca_attr, ca_f.tolist(), ca_cf.tolist(), lo_f.tolist(), lo_cf.tolist(), ca_path, lo_path))

print("paired", dict(by_attr))
print("examples")
for x in examples:
    print(x)
