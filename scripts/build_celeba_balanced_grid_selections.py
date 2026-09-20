#!/usr/bin/env python3
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.celeba_dataset import CelebADataset


ATTRS = ("Young", "Male", "No_Beard", "Bald")
DATA_ROOT = Path("/work/nvme/bhje/hyu29/datasets/celebA")
OUT_ROOT = Path("/work/nvme/bhje/hyu29/evaluations/celeba/selection/balanced_grid_n32_seed42")


def main() -> None:
    ds = CelebADataset(
        root=str(DATA_ROOT), split="test", factor_cols=list(ATTRS), image_size=64
    )
    labels = ds.factor_matrix(ATTRS)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for column, attr in enumerate(ATTRS):
        rng = np.random.default_rng(42)
        negative = rng.choice(np.flatnonzero(labels[:, column] < 0), 16, replace=False)
        positive = rng.choice(np.flatnonzero(labels[:, column] > 0), 16, replace=False)
        # Interleave the two intervention directions in the displayed grid.
        indices = np.column_stack((negative, positive)).reshape(-1).tolist()
        payload = {
            "schema_version": 1,
            "data_root": str(DATA_ROOT),
            "split": "test",
            "image_size": 64,
            "selection": "16 negative + 16 positive, interleaved",
            "balanced_attribute": attr,
            "seed": 42,
            # Existing evaluators read these legacy keys and concatenate them.
            # Keep the interleaved attribute-balanced order in the first list.
            "male_indices": indices,
            "female_indices": [],
        }
        output = OUT_ROOT / f"selection_test_{attr}_balanced_n32_seed42.json"
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(attr, len(indices), output)


if __name__ == "__main__":
    main()
