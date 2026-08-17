from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--attr", choices=("No_Beard", "Bald"), required=True)
    p.add_argument("--include-self-sweep", action="store_true")
    return p.parse_args()


def load_rows(root: Path, attr: str, include_self: bool):
    patterns = [
        root / "round1_cross" / attr / "*" / f"metrics_do_{attr}_n32.json",
    ]
    if include_self:
        patterns.append(root / "round2_self" / attr / "*" / f"metrics_do_{attr}_n32.json")
    rows = []
    for pattern in patterns:
        for filename in glob.glob(str(pattern)):
            data = json.loads(Path(filename).read_text())
            args = data["args"]
            rows.append(
                {
                    "path": filename,
                    "cross": float(args["ptp_cross_replace_steps"]),
                    "self": float(args["ptp_self_replace_steps"]),
                    "target_f1": float(data["effectiveness"][attr]),
                    "lpips": float(data["lpips_identity"]),
                    "mae": float(data["pixel_mae_identity"]["mean"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No completed metrics found for {attr}: {patterns}")
    return rows


def main():
    args = parse_args()
    rows = load_rows(Path(args.root), args.attr, args.include_self_sweep)
    # Preserve at least 90% of the best observed effectiveness, then prefer
    # the least-changing candidate. This avoids selecting a high-F1 setting
    # whose advantage comes from substantially larger global drift.
    best_f1 = max(row["target_f1"] for row in rows)
    eligible = [row for row in rows if row["target_f1"] >= 0.9 * best_f1]
    eligible.sort(key=lambda row: (row["lpips"], row["mae"], -row["target_f1"]))
    best = eligible[0]
    print(f'{best["cross"]:.8g} {best["self"]:.8g}')


if __name__ == "__main__":
    main()
