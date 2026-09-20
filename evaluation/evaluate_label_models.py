"""Evaluate the two label-side components used by the counterfactual pipeline.

This script intentionally keeps the two error sources separate:
  1. causal model (labels -> labels), evaluated on held-out observational labels;
  2. anti-causal predictor (image -> labels), evaluated on real test images.

It supports the local CelebA full-label SCM, the released CA Pendulum SCM,
the released counterfactual-benchmark CelebA classifiers, and our Pendulum
four-output image regressor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.celeba_dataset import CelebADataset
from data.pendulum_dataset import PENDULUM_ATTRS, PENDULUM_MINMAX, PendulumDataset
from models.full_label_scm import FULL_LABEL_KEYS, FullLabelSCM
from training.train_pendulum_regressor import PendulumRegressor


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(np.int64)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "bce": float(F.binary_cross_entropy(torch.from_numpy(p), torch.from_numpy(y.astype(np.float32))).item()),
        "positive_rate": float(y.mean()),
    }
    out["auroc"] = roc_auc_score(y, p) if np.unique(y).size == 2 else None
    return out


def evaluate_celeba_scm(args) -> dict:
    ds = CelebADataset(args.celeba_root, "test", FULL_LABEL_KEYS, image_size=64)
    labels = torch.from_numpy(ds.factor_matrix(FULL_LABEL_KEYS)).float()
    ckpt = torch.load(args.celeba_scm, map_location="cpu")
    hidden = int(ckpt.get("config", {}).get("hidden_dim", 64))
    model = FullLabelSCM(hidden_dim=hidden)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    labels01 = (labels > 0).float()
    with torch.no_grad():
        probs = torch.sigmoid(model.logits_from_01(labels01)).numpy()
    per_attr = {}
    for j in model.child_indices:
        per_attr[FULL_LABEL_KEYS[j]] = binary_metrics(labels01[:, j].numpy().astype(np.int64), probs[:, j])
    return {"n": len(ds), "checkpoint": args.celeba_scm, "children_only": per_attr,
            "roots_passed_through": [FULL_LABEL_KEYS[j] for j in model.root_indices]}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate_celeba_anticausal(args) -> dict:
    # Match the benchmark's CenterCrop(150) -> Resize(64) preprocessing exactly.
    ds = CelebADataset(args.celeba_root, "test", FULL_LABEL_KEYS, image_size=64, center_crop_size=150)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    bench = Path(args.benchmark_root)
    sys.path.insert(0, str(bench))
    mod = load_module(bench / "models/classifiers/celeba_complex_classifier.py", "ca_celeba_classifier")
    models = {}
    for attr in FULL_LABEL_KEYS:
        path = next((bench / "methods/deepscm/checkpoints/celeba/complex/trained_classifiers").glob(f"{attr}_classifier-*.ckpt"))
        context_dim = 2 if attr in ("Young", "Male") else 0
        models[attr] = mod.CelebaComplexClassifier.load_from_checkpoint(
            str(path), attr=attr, in_shape=(3, 64, 64), n_chan=[3, 8, 16, 32, 32, 64, 1],
            context_dim=context_dim, version="standard", map_location=args.device,
        ).to(args.device).eval()
    ys = {a: [] for a in FULL_LABEL_KEYS}
    ps = {a: [] for a in FULL_LABEL_KEYS}
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(args.device)
            y01 = {a: (batch[a].view(-1).to(args.device) > 0).float() for a in FULL_LABEL_KEYS}
            cond = torch.stack([y01["No_Beard"], y01["Bald"]], dim=1)
            for attr in FULL_LABEL_KEYS:
                logits = models[attr](x, cond if attr in ("Young", "Male") else None).view(-1)
                ys[attr].append(y01[attr].cpu())
                ps[attr].append(torch.sigmoid(logits).cpu())
    per_attr = {}
    for attr in FULL_LABEL_KEYS:
        y = torch.cat(ys[attr]).numpy().astype(np.int64)
        p = torch.cat(ps[attr]).numpy()
        per_attr[attr] = binary_metrics(y, p)
    return {"n": len(ds), "preprocess": "CenterCrop(150), Resize(64), ToTensor", "per_attr": per_attr}


def evaluate_pendulum_anticausal(args) -> dict:
    ds = PendulumDataset(args.pendulum_root, "test", image_size=128)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    ckpt = torch.load(args.pendulum_regressor, map_location="cpu")
    model = PendulumRegressor().to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    pred, target = [], []
    with torch.no_grad():
        for batch in loader:
            pred.append(model(batch["image"].to(args.device)).cpu())
            target.append(batch["factors"])
    p = torch.cat(pred).numpy(); y = torch.cat(target).numpy()
    scale = PENDULUM_MINMAX[:, 1] - PENDULUM_MINMAX[:, 0]
    per_attr = {}
    for j, attr in enumerate(PENDULUM_ATTRS):
        per_attr[attr] = {
            "mae_normalized": mean_absolute_error(y[:, j], p[:, j]),
            "rmse_normalized": mean_squared_error(y[:, j], p[:, j]) ** 0.5,
            "mae_raw_units": mean_absolute_error(y[:, j], p[:, j]) * float(scale[j]),
            "r2": r2_score(y[:, j], p[:, j]),
        }
    return {"n": len(ds), "checkpoint": args.pendulum_regressor, "per_attr": per_attr}


def evaluate_pendulum_scm(args) -> dict:
    ca = Path(args.ca_root)
    sys.path.insert(0, str(ca))
    control = load_module(ca / "causal_modules/control_heads.py", "ca_control_heads")
    common = load_module(ca / "SCM_modeling/common.py", "ca_scm_common")
    _, test, adj = common.load_dataset_splits("pendulum", args.pendulum_root)
    model = control.ControlNetConditioningEmbedding(4, mask=adj, dataset_name="pendulum")
    state = torch.load(args.pendulum_scm, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.mask = torch.as_tensor(adj, dtype=torch.float32)
    model.eval()
    y = torch.as_tensor(test, dtype=torch.float32)
    with torch.no_grad():
        output, _ = model.pretrain(y)
    p = output.squeeze(-1).numpy(); yn = y.numpy()
    child = np.flatnonzero(np.asarray(adj).sum(axis=0) > 0)
    per_attr = {}
    for j in child:
        per_attr[PENDULUM_ATTRS[j]] = {
            "mae_model_space": mean_absolute_error(yn[:, j], p[:, j]),
            "rmse_model_space": mean_squared_error(yn[:, j], p[:, j]) ** 0.5,
            "r2": r2_score(yn[:, j], p[:, j]),
        }
    roots = [PENDULUM_ATTRS[j] for j in range(4) if j not in child]
    return {"n": len(y), "checkpoint": args.pendulum_scm, "normalization": "CA gaussian label normalization",
            "children_only": per_attr, "roots_passed_through": roots, "adjacency": np.asarray(adj).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--celeba-root", required=True)
    p.add_argument("--pendulum-root", required=True)
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--ca-root", required=True)
    p.add_argument("--celeba-scm", required=True)
    p.add_argument("--pendulum-scm", required=True)
    p.add_argument("--pendulum-regressor", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    result = {
        "celeba_scm": evaluate_celeba_scm(args),
        "celeba_anticausal_real_test": evaluate_celeba_anticausal(args),
        "pendulum_scm": evaluate_pendulum_scm(args),
        "pendulum_anticausal_real_test": evaluate_pendulum_anticausal(args),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
