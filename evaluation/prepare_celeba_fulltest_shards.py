"""Create a deterministic CA-paper-style full-test intervention manifest."""
import json
import random
from pathlib import Path

SOURCE = Path("/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json")
OUT = Path("/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100/plans")
ATTRS = ("Young", "Male", "No_Beard", "Bald")
SEED = 42
OFFICIAL_BATCH_SIZE = 4
SHARD_SIZE = 250

raw = json.loads(SOURCE.read_text())
indices = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
if len(indices) != 19962:
    indices = [int(x) for x in raw["indices"]]
assert len(indices) == 19962 and len(set(indices)) == 19962
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)
assignment = []
for start in range(0, len(indices), OFFICIAL_BATCH_SIZE):
    attr = rng.choice(ATTRS)
    assignment.extend({"source_index": i, "intervention_attr": attr} for i in indices[start:start + OFFICIAL_BATCH_SIZE])

plans = []
for attr in ATTRS:
    selected = [x["source_index"] for x in assignment if x["intervention_attr"] == attr]
    for start in range(0, len(selected), SHARD_SIZE):
        part = selected[start:start + SHARD_SIZE]
        plan_id = len(plans)
        path = OUT / f"plan_{plan_id:03d}.json"
        path.write_text(json.dumps({"male_indices": part, "female_indices": [],
                                    "intervention_attr": attr, "plan_id": plan_id}, indent=2) + "\n")
        plans.append({"plan_id": plan_id, "count": len(part), "intervention_attr": attr, "path": str(path)})
(OUT / "manifest.json").write_text(json.dumps({
    "protocol": "CA official evaluate_SD.py: one random intervention per evaluation batch",
    "seed": SEED, "official_batch_size": OFFICIAL_BATCH_SIZE,
    "num_samples": len(indices), "num_plans": len(plans), "attributes": ATTRS,
    "assignment": assignment, "plans": plans,
}, indent=2) + "\n")
print(OUT / "manifest.json", len(plans), len(assignment))
