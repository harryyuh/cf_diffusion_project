import json
from pathlib import Path

source = Path("/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json")
out = Path("/work/nvme/bhje/hyu29/evaluations/celeba/ca_protocol_n2048_ddim100/plans")
out.mkdir(parents=True, exist_ok=True)
raw = json.loads(source.read_text())
indices = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
if len(indices) < 2048:
    indices = [int(x) for x in raw.get("indices", [])]
assert len(indices) >= 2048, len(indices)
indices = indices[:2048]
assert len(set(indices)) == 2048
for shard in range(8):
    part = indices[shard * 256:(shard + 1) * 256]
    (out / f"shard_{shard}.json").write_text(json.dumps({"male_indices": part, "female_indices": []}, indent=2))
(out / "manifest.json").write_text(json.dumps({"num_samples": 2048, "num_shards": 8, "indices": indices,
    "intervention_assignment": {str(i): ["Young", "Male", "No_Beard", "Bald"][i % 4] for i in range(8)}}, indent=2))
print(out / "manifest.json")
