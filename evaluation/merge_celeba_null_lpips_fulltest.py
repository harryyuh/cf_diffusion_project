"""Merge one-cycle CelebA null-edit composition from paired saved shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected", type=int, default=19962)
    args = p.parse_args()

    plan_files = sorted(args.plan_root.glob("plan_[0-9][0-9][0-9].json"))
    if len(plan_files) != 82:
        raise RuntimeError(f"Expected 82 plans, found {len(plan_files)}")
    expected_indices = set(range(args.expected))
    by_method = {}
    for method in ("ca", "label_only"):
        seen = []
        lpips_sum = 0.0
        mae_sum = 0.0
        for plan in plan_files:
            stem = plan.stem
            path = args.root / method / f"{stem}.json"
            result = json.loads(path.read_text())
            if result["method"] != method or result["smoke_only"]:
                raise RuntimeError(f"Incorrect method or smoke output: {path}")
            expected = json.loads(plan.read_text())
            indices = [int(i) for key in ("male_indices", "female_indices") for i in expected[key]]
            if result["source_indices"] != indices or result["n"] != len(indices):
                raise RuntimeError(f"Source mismatch in {path}")
            seen.extend(indices)
            lpips_sum += result["lpips_composition"] * len(indices)
            mae_sum += result["mae_01"] * len(indices)
        if len(seen) != args.expected or set(seen) != expected_indices:
            raise RuntimeError(f"{method} does not cover the complete test set exactly once")
        by_method[method] = {
            "n": len(seen),
            "lpips_composition": lpips_sum / len(seen),
            "mae_01": mae_sum / len(seen),
        }
    result = {
        "protocol": "one-cycle null intervention; original attributes unchanged; DDIM100; CFG invert1/sample3; benchmark AlexNet LPIPS; matched 19,962-source test set",
        "ca": by_method["ca"],
        "label_only": by_method["label_only"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
