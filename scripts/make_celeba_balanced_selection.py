from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CELEBA_ATTR_ORDER = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/scratch/gilbreth/yu1331/datasets/celebA")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-each", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", required=True)
    p.add_argument("--subset-from-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.data_root)
    split_code = {"train": 0, "val": 1, "test": 2}[args.split]
    partition = {}
    with open(root / "list_eval_partition.txt", "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                partition[parts[0]] = int(parts[1])
    attr_values = {}
    with open(root / "list_attr_celeba.txt", "r") as f:
        _ = f.readline()
        cols = f.readline().split()
        male_col = cols.index("Male")
        for line in f:
            parts = line.split()
            if len(parts) == len(cols) + 1:
                attr_values[parts[0]] = int(parts[1 + male_col])
    filenames = sorted(name for name in attr_values if partition.get(name) == split_code)

    if args.subset_from_json:
        with open(args.subset_from_json, "r") as f:
            parent = json.load(f)
        male_pool = [int(x) for x in parent["male_indices"]]
        female_pool = [int(x) for x in parent["female_indices"]]
    else:
        male_pool = []
        female_pool = []
        for idx, name in enumerate(filenames):
            if attr_values[name] > 0:
                male_pool.append(idx)
            else:
                female_pool.append(idx)

    if len(male_pool) < args.n_each or len(female_pool) < args.n_each:
        raise ValueError(
            f"Need {args.n_each} per group, got male={len(male_pool)} female={len(female_pool)}"
        )

    male_indices = sorted(rng.sample(male_pool, args.n_each))
    female_indices = sorted(rng.sample(female_pool, args.n_each))
    out = {
        "schema_version": 1,
        "data_root": args.data_root,
        "split": args.split,
        "image_size": 64,
        "celeba_attr_cols": list(CELEBA_ATTR_ORDER),
        "parent_keys": ["Male"],
        "n_each": args.n_each,
        "seed": args.seed,
        "subset_from_json": args.subset_from_json,
        "male_indices": male_indices,
        "female_indices": female_indices,
        "male_filenames": [filenames[i] for i in male_indices],
        "female_filenames": [filenames[i] for i in female_indices],
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
