"""Split all 19,962 CelebA test indices into restartable CA-protocol shards."""
import json
from pathlib import Path

SOURCE = Path("/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json")
OUT = Path("/work/nvme/bhje/hyu29/evaluations/celeba/ca_protocol_fulltest_ddim100/plans")
SHARD_SIZE = 250

raw = json.loads(SOURCE.read_text())
indices = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
if len(indices) != 19962:
    indices = [int(x) for x in raw["indices"]]
assert len(indices) == 19962 and len(set(indices)) == 19962
OUT.mkdir(parents=True, exist_ok=True)
attrs = ["Young", "Male", "No_Beard", "Bald"]
shards = []
for shard, start in enumerate(range(0, len(indices), SHARD_SIZE)):
    part = indices[start:start + SHARD_SIZE]
    attr = attrs[shard % len(attrs)]
    path = OUT / f"shard_{shard}.json"
    path.write_text(json.dumps({"male_indices": part, "female_indices": []}, indent=2) + "\n")
    shards.append({"shard": shard, "count": len(part), "attribute": attr, "path": str(path)})
(OUT / "manifest.json").write_text(json.dumps({
    "num_samples": len(indices), "shard_size": SHARD_SIZE, "num_shards": len(shards),
    "indices": indices, "shards": shards,
    "intervention_assignment": "deterministic uniform cycling over four attributes; pooled as in CA FID",
}, indent=2) + "\n")
print(OUT / "manifest.json", len(shards))
