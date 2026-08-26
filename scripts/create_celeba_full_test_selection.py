"""Create the full CelebA test-index selection consumed by our evaluator."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=19962)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "data_root": "/work/nvme/bhje/hyu29/datasets/celebA",
        "split": "test",
        "image_size": 64,
        "celeba_attr_cols": [],
        "parent_keys": ["Male"],
        "n_each": 0,
        "seed": 42,
        "subset_from_json": "",
        "male_indices": list(range(args.size)),
        "female_indices": [],
        "male_filenames": [],
        "female_filenames": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
