"""
Select father_dims and rest_dims from p-values; save as JSON.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select latent dimensions by p-value.")
    parser.add_argument("--config", type=str, required=True, help="Path to logistic config YAML.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    pvalues_path = output_dir / "pvalues.csv"
    if not pvalues_path.exists():
        raise FileNotFoundError(f"Run fit_logistic_regression first. Expected {pvalues_path}")

    pvalues_df = pd.read_csv(pvalues_path, index_col=0)
    pvalues = pvalues_df["pvalue"].values
    pvalue_threshold = cfg.get("pvalue_threshold", 0.05)

    # father_dims = indices with p-value < threshold
    father_dims: List[int] = [i for i, p in enumerate(pvalues) if p < pvalue_threshold]
    rest_dims: List[int] = [i for i in range(len(pvalues)) if i not in father_dims]

    (output_dir / "father_dims.json").write_text(json.dumps(father_dims))
    (output_dir / "rest_dims.json").write_text(json.dumps(rest_dims))

    summary = {
        "pvalue_threshold": pvalue_threshold,
        "n_father_dims": len(father_dims),
        "n_rest_dims": len(rest_dims),
        "total_dims": len(pvalues),
    }
    (output_dir / "latent_selection_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"father_dims: {len(father_dims)}, rest_dims: {len(rest_dims)}")
    print(f"Saved father_dims.json, rest_dims.json, latent_selection_summary.json to {output_dir}")


if __name__ == "__main__":
    main()
