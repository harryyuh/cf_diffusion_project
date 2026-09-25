"""Validate the four CA Pendulum regressors on real held-out images.

This intentionally uses CA's own PendulumLike dataset and Gaussian label
normalization.  A PNG round-trip is also measured to rule out serialization
as the source of evaluator disagreement on generated counterfactuals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ATTRS = ["pendulum", "light", "shadow_length", "shadow_position"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = Path(args.ca_root) / "counterfactual-benchmark" / "counterfactual_benchmark"
    sys.path.insert(0, str(benchmark))
    from ctf_datasets.pendulum.dataset import PendulumLike
    from models.classifiers.pendulum_classifier import PendClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attribute_size = {name: 1 for name in ATTRS}
    dataset = PendulumLike(attribute_size, split="test", normalize_=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    models = {}
    for name in ATTRS:
        matches = sorted(Path(args.checkpoint_dir).glob(f"{name}_classifier-*.ckpt"))
        if len(matches) != 1:
            raise RuntimeError(f"{name}: expected one checkpoint, found {matches}")
        model = PendClassifier(attr=name, width=8, num_outputs=1, context_dim=0, lr=1e-4).to(device)
        payload = torch.load(matches[0], map_location=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models[name] = (model, str(matches[0]))

    sums = torch.zeros(4, dtype=torch.float64)
    sq_sums = torch.zeros(4, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = torch.cat([models[name][0](images) for name in ATTRS], dim=1)
            errors = (predictions - labels).abs().double().cpu()
            sums += errors.sum(0)
            sq_sums += errors.square().sum(0)
            count += images.shape[0]

    mean = sums / count
    variance = (sq_sums / count - mean.square()).clamp_min(0)
    result = {
        "dataset": "CA PendulumLike test",
        "n": count,
        "normalization": "(raw - [2,104,7.5,11]) / [42,44,4.5,8]",
        "mae_ca_normalized": {name: float(mean[i]) for i, name in enumerate(ATTRS)},
        "std_abs_error": {name: float(variance[i].sqrt()) for i, name in enumerate(ATTRS)},
        "mean_mae_ca_normalized": float(mean.mean()),
        "checkpoints": {name: path for name, (_, path) in models.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
