"""
DCI-style metrics (disentanglement, completeness, informativeness) from a CelebA-trained VAE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset, expand_env_in_cfg
from models.vae_factory import build_vae_from_train_cfg
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute DCI metrics for CelebA VAE.")
    p.add_argument("--config", type=str, default="configs/dci_celeba.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_r2_matrix(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, k = z.shape
    _, l = y.shape
    R = np.zeros((k, l), dtype=np.float64)
    for i in range(k):
        xi = z[:, i : i + 1]
        for j in range(l):
            lr = LinearRegression()
            lr.fit(xi, y[:, j])
            pred = lr.predict(xi)
            r2 = r2_score(y[:, j], pred)
            R[i, j] = max(0.0, float(r2))
    return R


def disentanglement_scores(R: np.ndarray) -> Tuple[np.ndarray, float]:
    eps = 1e-11
    Rp = np.maximum(R, eps)
    P_row = Rp / Rp.sum(axis=1, keepdims=True)
    k, l = P_row.shape
    H = np.zeros(k)
    for i in range(k):
        H[i] = float(-np.sum(P_row[i] * np.log(P_row[i] + eps)))
    max_h = np.log(l)
    d_per_code = 1.0 - H / max_h
    total = Rp.sum()
    if total < eps:
        rho = np.ones(k) / k
    else:
        rho = Rp.sum(axis=1) / total
    d_global = float(np.sum(rho * d_per_code))
    return d_per_code, d_global


def completeness_scores(R: np.ndarray) -> Tuple[np.ndarray, float]:
    eps = 1e-11
    Rp = np.maximum(R, eps)
    P_col = Rp / Rp.sum(axis=0, keepdims=True)
    k, l = P_col.shape
    H = np.zeros(l)
    for j in range(l):
        H[j] = float(-np.sum(P_col[:, j] * np.log(P_col[:, j] + eps)))
    max_h = np.log(k)
    c_per_factor = 1.0 - H / max_h
    total = Rp.sum()
    if total < eps:
        w = np.ones(l) / l
    else:
        w = Rp.sum(axis=0) / total
    c_global = float(np.sum(w * c_per_factor))
    return c_per_factor, c_global


def informativeness_r2_per_output(z: np.ndarray, y: np.ndarray) -> List[float]:
    _, l = y.shape
    scores: List[float] = []
    lr = LinearRegression()
    for j in range(l):
        lr.fit(z, y[:, j])
        pred = lr.predict(z)
        scores.append(max(0.0, float(r2_score(y[:, j], pred))))
    return scores


def informativeness_r2(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(informativeness_r2_per_output(z, y)))


def informativeness_r2_z_rest(
    z: np.ndarray, n_parent_dims: int, y: np.ndarray
) -> Optional[float]:
    n_parent_dims = int(n_parent_dims)
    if n_parent_dims <= 0 or n_parent_dims >= z.shape[1]:
        return None
    zr = z[:, n_parent_dims:]
    return informativeness_r2(zr, y)


def informativeness_r2_z_rest_per_output(
    z: np.ndarray, n_parent_dims: int, y: np.ndarray
) -> Optional[List[float]]:
    n_parent_dims = int(n_parent_dims)
    if n_parent_dims <= 0 or n_parent_dims >= z.shape[1]:
        return None
    zr = z[:, n_parent_dims:]
    return informativeness_r2_per_output(zr, y)


@torch.no_grad()
def collect_mu(vae: nn.Module, loader: DataLoader, device: torch.device, use_mu: bool) -> np.ndarray:
    out: List[np.ndarray] = []
    for batch in tqdm(loader, desc="encode"):
        x = batch["image"].to(device)
        mu, logvar = vae.encode(x)
        z = mu if use_mu else vae.reparameterize(mu, logvar)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def _load_vae(vae_cfg_path: Path, ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, int]:
    with open(vae_cfg_path, "r") as f:
        vcfg = yaml.safe_load(f)
    expand_env_in_cfg(vcfg)
    n_parent_dims = int(vcfg.get("n_parent_dims", 0))
    vae = build_vae_from_train_cfg(vcfg)
    load_checkpoint(ckpt_path, model=vae, map_location="cpu", strict=False)
    vae = vae.to(device)
    vae.eval()
    return vae, n_parent_dims


def _metrics_from_z(
    z: np.ndarray,
    y: np.ndarray,
    n_parent_dims: int,
    factor_names: List[str],
    focus_names: Optional[List[str]],
) -> Dict[str, Any]:
    R = compute_r2_matrix(z, y)
    d_per, d_global = disentanglement_scores(R)
    c_per, c_global = completeness_scores(R)

    info = informativeness_r2(z, y)
    info_z_rest = informativeness_r2_z_rest(z, n_parent_dims, y)
    info_z_rest_per = informativeness_r2_z_rest_per_output(z, n_parent_dims, y)
    info_z_rest_by: Optional[Dict[str, float]] = None
    if info_z_rest_per is not None:
        info_z_rest_by = {factor_names[j]: info_z_rest_per[j] for j in range(len(factor_names))}

    info_z_rest_focus: Optional[float] = None
    if focus_names and info_z_rest_by:
        vals = [info_z_rest_by[k] for k in focus_names if k in info_z_rest_by]
        if vals:
            info_z_rest_focus = float(np.mean(vals))

    return {
        "latent_dim": z.shape[1],
        "n_parent_dims": n_parent_dims,
        "z_rest_dim": z.shape[1] - n_parent_dims if n_parent_dims > 0 else z.shape[1],
        "num_factor_outputs": y.shape[1],
        "disentanglement": d_global,
        "completeness": c_global,
        "informativeness_mean_r2": info,
        "informativeness_z_rest_mean_r2": info_z_rest,
        "informativeness_z_rest_r2_by_factor_output": info_z_rest_by,
        "informativeness_z_rest_mean_r2_focus": info_z_rest_focus,
        "disentanglement_per_latent_dim": d_per.tolist(),
        "completeness_per_factor_output": c_per.tolist(),
        "factor_output_names": factor_names,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    expand_env_in_cfg(cfg)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output_json is not None:
        cfg["output_json"] = args.output_json

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 123))

    data_root = cfg["data_root"]
    split = cfg.get("split", "test")
    if cfg.get("use_all_celeba_attrs"):
        factor_cols = list(CELEBA_ATTR_ORDER)
    else:
        factor_cols = list(cfg["celeba_attr_cols"])
    image_size = int(cfg.get("image_size", 64))
    continuous_cols: List[str] = list(cfg.get("continuous_factor_cols", factor_cols))

    ds = CelebADataset(
        root=data_root,
        split=split,
        factor_cols=factor_cols,
        image_size=image_size,
    )
    y = ds.factor_matrix(continuous_cols)

    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    use_mu = bool(cfg.get("use_encoder_mu", True))
    focus_names: Optional[List[str]] = cfg.get("informativeness_z_rest_focus_cols")

    vae_runs = cfg.get("vae_runs")
    if vae_runs:
        runs_out: Dict[str, Dict[str, Any]] = {}
        for i, spec in enumerate(vae_runs):
            if not isinstance(spec, dict):
                raise ValueError(f"vae_runs[{i}] must be a mapping")
            name = str(spec.get("name", f"run_{i}"))
            ckpt_path = Path(spec["vae_checkpoint"])
            vae_cfg_path = Path(spec["vae_config"])
            vae, n_parent_dims = _load_vae(vae_cfg_path, ckpt_path, device)
            z = collect_mu(vae, loader, device, use_mu=use_mu)
            m = _metrics_from_z(z, y, n_parent_dims, continuous_cols, focus_names)
            m["vae_checkpoint"] = str(ckpt_path.resolve())
            m["vae_config"] = str(vae_cfg_path.resolve())
            runs_out[name] = m
        out: Dict[str, Any] = {
            "data_root": str(Path(data_root).resolve()),
            "split": split,
            "continuous_factor_cols": continuous_cols,
            "use_encoder_mu": use_mu,
            "runs": runs_out,
        }
    else:
        vae_cfg_path = Path(cfg["vae_config"])
        ckpt_path = Path(cfg["vae_checkpoint"])
        vae, n_parent_dims = _load_vae(vae_cfg_path, ckpt_path, device)
        z = collect_mu(vae, loader, device, use_mu=use_mu)
        m = _metrics_from_z(z, y, n_parent_dims, continuous_cols, focus_names)
        out = {
            "vae_checkpoint": str(ckpt_path.resolve()),
            "vae_config": str(vae_cfg_path.resolve()),
            "data_root": str(Path(data_root).resolve()),
            "split": split,
            **m,
            "continuous_factor_cols": continuous_cols,
        }

    print(json.dumps(out, indent=2))

    out_path = cfg.get("output_json")
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
