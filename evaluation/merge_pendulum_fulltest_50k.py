"""Validate and merge paired CA/Label-only Pendulum full-directory shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTRS = ("pendulum", "light", "shadow_length", "shadow_position")
METHODS = ("ca", "joint_label")


def load(root: Path, method: str, shards: int):
    rows = []
    for i in range(shards):
        leaf = root / f"shard_{i}" / method
        selection = json.loads((leaf / "selection.json").read_text())
        metric = json.loads((leaf / "metrics_ca_official_regressors.json").read_text())
        ids = selection["test_indices"]
        assert len(ids) == metric["n"], (leaf, len(ids), metric["n"])
        assert selection["shard_index"] == i, leaf
        assert selection["output_resolution"] == 256, leaf
        rows.append((selection, metric))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=29)
    ap.add_argument("--expected-n", type=int, default=1826)
    args = ap.parse_args()
    results = {method: load(args.root, method, args.shards) for method in METHODS}
    reference = results["ca"]
    for method, rows in results.items():
        ids = [idx for selection, _ in rows for idx in selection["test_indices"]]
        assert len(ids) == args.expected_n and sorted(ids) == list(range(args.expected_n)), method
        assert all(s["test_size"] == args.expected_n for s, _ in rows), method
        for (selection, _), (ref_selection, _) in zip(rows, reference):
            assert selection["test_indices"] == ref_selection["test_indices"], method
            assert selection["intervention_train_indices"] == ref_selection["intervention_train_indices"], method

    merged = {
        "n": args.expected_n,
        "test_population": "all PNGs in the current DeltaAI test directory",
        "steps": 100,
        "output_resolution": 256,
        "evaluator_resolution": 96,
        "evaluator": "four CA benchmark PendClassifier checkpoints",
        "methods": {},
    }
    for method, rows in results.items():
        matrix = {}
        for do in ATTRS:
            matrix[do] = {}
            for target in ATTRS:
                matrix[do][target] = sum(
                    metric["n"] * metric["interventions"][do]["mae_ca_normalized"][target]
                    for _, metric in rows
                ) / args.expected_n
        merged["methods"][method] = {"mae_ca_normalized": matrix}
    output = args.root / "merged_ca_official_regressors.json"
    output.write_text(json.dumps(merged, indent=2) + "\n")
    print(output)
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
