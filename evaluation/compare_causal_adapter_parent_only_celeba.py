"""Compare parent-only diffusion and Causal-Adapter on CelebA do(Male).

Uses the Causal-Adapter repository's benchmark metric implementations where
practical: ``effectiveness_SD.effectiveness`` and ``minimality_SD.minimality``.
FID uses the same torchmetrics backend as the official ``fid_SD`` helper.

Evaluation protocol (shared across methods):
- Parent-only **generation** stays native 64x64 DDIM (unchanged).
- **Real** images for FID/minimality: CenterCrop(150) -> 256 -> bicubic 64.
- **Counterfactual** images for classifier/FID: each method's output at 64x64.
- **Effectiveness targets**: P2P ``causal_cond`` from Causal-Adapter (same for PO and CA).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from importlib import import_module
from json import load as json_load
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
COMPLEX_ATTRS = ["Young", "Male", "No_Beard", "Bald"]


def _add_ca_paths(ca_root: Path) -> Path:
    bench_root = ca_root / "counterfactual-benchmark" / "counterfactual_benchmark"
    for p in (ca_root / "diffusers" / "src", ca_root / "notebook_benchmarks", ca_root, bench_root):
        sp = str(p.resolve())
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)
    return bench_root


def _patch_safety_checker_compat() -> None:
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    except Exception:
        return
    if not hasattr(StableDiffusionSafetyChecker, "all_tied_weights_keys"):
        StableDiffusionSafetyChecker.all_tied_weights_keys = {}


def _load_official_metric(bench_root: Path, module_name: str, attr_name: str) -> Any:
    import importlib.util

    path = bench_root / "evaluation" / "metrics" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"ca_benchmark_{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official Causal-Adapter metric from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--causal-adapter-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter")
    p.add_argument("--parent-config", default="configs/diffusion_celeba_parent_only.yaml")
    p.add_argument("--selection-json", default="")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--parent-start-t", type=int, default=300)
    p.add_argument("--ca-base-model-path", default="lambdalabs/miniSD-diffusers")
    p.add_argument("--ca-controlnet-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/controlnet-steps-200000.safetensors")
    p.add_argument("--ca-text-embedding-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/learned_embeds-steps-200000.safetensors")
    p.add_argument("--ca-scm-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt")
    p.add_argument("--ca-editing", choices=["p2p", "standard"], default="p2p")
    p.add_argument("--intervention-attr", choices=COMPLEX_ATTRS, default="Male")
    p.add_argument("--ca-steps", type=int, default=50)
    p.add_argument("--ca-guidance-scale", type=float, default=3.0)
    p.add_argument("--ca-invert-guidance-scale", type=float, default=1.0)
    p.add_argument(
        "--ca-batched-standard",
        action="store_true",
        help="Run standard CA DDIM on the full DataLoader batch instead of serializing images.",
    )
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-grid", default="")
    p.add_argument("--skip-parent-only", action="store_true")
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    from data.celeba_dataset import expand_env_in_cfg

    expand_env_in_cfg(cfg)
    return cfg


def _load_selection_indices(path: str, n: int) -> Optional[List[int]]:
    if not path:
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    idxs = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
    return idxs[:n] if n > 0 else idxs


def _label01_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in COMPLEX_ATTRS:
        out[k] = (batch[k].to(device).float().view(-1, 1) > 0).float()
    return out


class FilenameSubset(Dataset):
    """Subset wrapper that keeps the original CelebA filename available."""

    def __init__(self, dataset: Dataset, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.filenames = getattr(dataset, "_filenames", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        source_idx = self.indices[idx]
        item = dict(self.dataset[source_idx])
        item["source_index"] = torch.tensor(source_idx, dtype=torch.long)
        if self.filenames is not None:
            item["filename"] = self.filenames[source_idx]
        return item


class ParentOnlyGenerator:
    def __init__(self, cfg_path: Path, start_t: int, device: torch.device) -> None:
        old_path = list(sys.path)
        old_model_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "models" or name.startswith("models.")
        }
        for name in list(old_model_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(REPO))
        try:
            from inference.generate_celeba_counterfactual_male import _ddim_cf, _load_models, _parent_matrix
        finally:
            sys.path[:] = old_path

        self.cfg_path = cfg_path
        self.cfg = _load_yaml(cfg_path)
        self.device = device
        self.start_t = int(start_t)
        self.vae, self.unet, self.diffusion, self.n_parent_dims, self.include_z_rest, self.ckpt, self.subdir, *_ = _load_models(
            self.cfg, device, None
        )
        for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
            sys.modules.pop(name, None)
        sys.modules.update(old_model_modules)
        self._ddim_cf = _ddim_cf
        self._parent_matrix = _parent_matrix
        self.parent_keys = list(self.cfg.get("parent_keys") or ["Male"])

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["image"].to(self.device)
        f_obs = self._parent_matrix(batch, self.parent_keys, self.device)
        f_tgt = -f_obs
        x_cf = self._ddim_cf(
            self.vae,
            self.unet,
            self.diffusion,
            x,
            f_obs,
            f_tgt,
            self.n_parent_dims,
            self.start_t,
            self.device,
            include_z_rest=self.include_z_rest,
        )
        out = _label01_from_batch(batch, self.device)
        out["Male"] = (f_tgt > 0).float()
        out["image"] = x_cf.clamp(0, 1)
        return out


class CausalAdapterGenerator:
    def __init__(self, args: argparse.Namespace, device: torch.device, data_root: Path) -> None:
        from inference_utils import build_transforms, load_causal_adapter  # type: ignore
        from causal_modules.ddim_modules import P2P_editing, ddim_editing  # type: ignore

        self.args = args
        self.device = device
        self.data_root = data_root
        self.img_dir = data_root / "img_align_celeba"
        self.image_tfm, self.original_tfm, _ = build_transforms("celeA_complex", size=256)
        self.assets = load_causal_adapter(
            "celeA_complex",
            base_model_path=args.ca_base_model_path,
            controlnet_path=args.ca_controlnet_path,
            text_embedding_path=args.ca_text_embedding_path,
            scm_path=args.ca_scm_path or None,
            prompt="a human is @ and * and & and #",
            presudo_words="@,*,&,#",
            device=device,
            torch_dtype=torch.float32,
        )
        self.p2p = P2P_editing
        self.ddim = ddim_editing

    def _source_to_ca(self, x01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x01, size=(256, 256), mode="bilinear", align_corners=False)
        return (x * 2.0 - 1.0).clamp(-1, 1)

    def _pil_to_tensor64(self, img: Image.Image) -> torch.Tensor:
        return transforms.ToTensor()(img.convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)).to(self.device)

    def _official_source(self, filename: str) -> torch.Tensor:
        pil = Image.open(self.img_dir / filename).convert("RGB")
        return self.image_tfm(pil).unsqueeze(0).to(self.device, dtype=torch.float32)

    def official_real64(self, batch: Dict[str, Any]) -> torch.Tensor:
        filenames = batch.get("filename")
        if filenames is None:
            return batch["image"].to(self.device)
        imgs = []
        for filename in filenames:
            pil = Image.open(self.img_dir / str(filename)).convert("RGB")
            preview = self.original_tfm(pil)
            imgs.append(self._pil_to_tensor64(preview))
        return torch.stack(imgs, dim=0).to(self.device)

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        labels = _label01_from_batch(batch, self.device)
        filenames = batch.get("filename")
        xs = batch["image"].to(self.device)
        outs: List[torch.Tensor] = []
        cf_label_rows: List[torch.Tensor] = []
        intervention_attr = self.args.intervention_attr
        intervention_idx = COMPLEX_ATTRS.index(intervention_attr)
        target_attr = 1.0 - labels[intervention_attr]

        # The upstream CA helper already supports batched latents, prompts and
        # labels.  The old evaluator nevertheless called it once per image,
        # making DataLoader batch_size=4 effectively batch_size=1.  Keep the
        # serial P2P path unchanged; standard DDIM can safely use the native
        # batched implementation.
        if self.args.ca_batched_standard:
            if self.args.ca_editing != "standard":
                raise ValueError("--ca-batched-standard currently supports only --ca-editing standard")
            if filenames is not None:
                image_t = torch.cat([self._official_source(str(name)) for name in filenames], dim=0)
            else:
                image_t = self._source_to_ca(xs)
            label = torch.cat([labels[k] for k in COMPLEX_ATTRS], dim=1)
            target_labels = label.clone()
            target_labels[:, intervention_idx] = target_attr[:, 0]
            images, _, causal_cond, _ = self.ddim(
                self.assets.pipe,
                image_t,
                label,
                self.assets.presudo_token_ids,
                [self.assets.prompt] * image_t.shape[0],
                num_steps=self.args.ca_steps,
                invert_guidance_scale=self.args.ca_invert_guidance_scale,
                set_guidance_scale=self.args.ca_guidance_scale,
                intervention_indx=None,
                intervention_values=None,
                return_PIL=True,
                DSCM_labels=target_labels.unsqueeze(2),
            )
            if len(images) != image_t.shape[0]:
                raise RuntimeError(f"Expected {image_t.shape[0]} CA images, got {len(images)}")
            cf_labels = (causal_cond.squeeze(-1) > 0.5).float().to(self.device)
            out = {k: cf_labels[:, j : j + 1].clone() for j, k in enumerate(COMPLEX_ATTRS)}
            out["image"] = torch.stack([self._pil_to_tensor64(img) for img in images], dim=0).clamp(0, 1)
            return out

        for i in range(xs.shape[0]):
            if filenames is not None:
                image_t = self._official_source(str(filenames[i]))
            else:
                image_t = self._source_to_ca(xs[i : i + 1])
            label = torch.cat([labels[k][i : i + 1] for k in COMPLEX_ATTRS], dim=1)
            target_value = target_attr[i, 0]
            if self.args.ca_editing == "p2p":
                images, _, causal_cond, _ = self.p2p(
                    self.assets.pipe,
                    image_t,
                    label,
                    self.assets.presudo_token_ids,
                    self.assets.prompt,
                    self.assets.presudo_list,
                    num_steps=self.args.ca_steps,
                    invert_guidance_scale=self.args.ca_invert_guidance_scale,
                    set_guidance_scale=self.args.ca_guidance_scale,
                    intervention_indx=intervention_idx,
                    intervention_values=target_value,
                    return_PIL=True,
                    blend_word=True,
                    blend_params={"start_blend": 0.0, "th": (0.3, 0.5)},
                    disentangle=False,
                    cross_replace_steps=0.2,
                    self_replace_steps=0.2,
                    DSCM_labels=None,
                )
            else:
                target_labels = label.clone()
                target_labels[:, intervention_idx] = target_value
                images, _, causal_cond, _ = self.ddim(
                    self.assets.pipe,
                    image_t,
                    label,
                    self.assets.presudo_token_ids,
                    self.assets.prompt,
                    num_steps=self.args.ca_steps,
                    invert_guidance_scale=self.args.ca_invert_guidance_scale,
                    set_guidance_scale=self.args.ca_guidance_scale,
                    intervention_indx=None,
                    intervention_values=None,
                    return_PIL=True,
                    DSCM_labels=target_labels.unsqueeze(2),
                )
            outs.append(self._pil_to_tensor64(images[-1]))
            cf_label_rows.append((causal_cond.squeeze(-1) > 0.5).float().view(1, -1))
        cf_labels = torch.cat(cf_label_rows, dim=0).to(self.device)
        out = {k: cf_labels[:, j : j + 1].clone() for j, k in enumerate(COMPLEX_ATTRS)}
        out["image"] = torch.stack(outs, dim=0).clamp(0, 1)
        return out


def _build_predictors(bench_root: Path, device: torch.device) -> Dict[str, torch.nn.Module]:
    cfg_path = bench_root / "methods" / "deepscm" / "configs" / "celeba" / "complex" / "classifier.json"
    with open(cfg_path, "r") as f:
        cfg = json_load(f)
    ckpt_dir = Path(cfg["ckpt_path"])
    if not ckpt_dir.is_absolute():
        rel = Path(str(ckpt_dir).lstrip("./"))
        candidates = [
            bench_root / "methods" / "deepscm" / rel,
            cfg_path.parent / rel,
            Path("/home/yu1331/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm") / rel,
        ]
        ckpt_dir = next((p for p in candidates if p.exists()), candidates[0])
    from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier  # type: ignore

    predictors: Dict[str, torch.nn.Module] = {}
    for atr in COMPLEX_ATTRS:
        mod = CelebaComplexClassifier(
            attr=atr,
            context_dim=len(list(cfg["anticausal_graph"][atr])),
            num_outputs=int(cfg["attribute_size"][atr]),
            lr=float(cfg["lr"]),
            version=str(cfg["version"]),
        )
        fname = next(fn for fn in os.listdir(ckpt_dir) if fn.startswith(atr))
        sd = torch.load(ckpt_dir / fname, map_location=device)
        mod.load_state_dict(sd["state_dict"])
        predictors[atr] = mod.to(device).eval()
    return predictors


def _fid(real_images: Iterable[torch.Tensor], fake_images: Iterable[torch.Tensor], device: torch.device) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(normalize=True, reset_real_features=False).set_dtype(torch.float32).to(device)
    for x in real_images:
        metric.update(x.to(device), real=True)
    for x in fake_images:
        metric.update(x.to(device), real=False)
    return float(metric.compute().detach().cpu())


def _pixel_embedding(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(x.shape[0], -1)


def _binary_pred_labels(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    # torchmetrics treats out-of-[0,1] float predictions as logits; match that
    # behavior while still supporting probability-valued predictions.
    if np.nanmin(pred) < 0.0 or np.nanmax(pred) > 1.0:
        return pred > 0.0
    return pred > 0.5


def _binary_f1_from_arrays(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bin = _binary_pred_labels(pred).astype(bool).reshape(-1)
    target_bin = (np.asarray(target).reshape(-1) > 0.5)
    tp = np.logical_and(pred_bin, target_bin).sum(dtype=np.float64)
    fp = np.logical_and(pred_bin, ~target_bin).sum(dtype=np.float64)
    fn = np.logical_and(~pred_bin, target_bin).sum(dtype=np.float64)
    denom = (2.0 * tp) + fp + fn
    return float((2.0 * tp / denom) if denom > 0 else 0.0)


def _rate_from_arrays(values: np.ndarray, *, is_prediction: bool) -> float:
    if is_prediction:
        return float(_binary_pred_labels(values).mean())
    return float((np.asarray(values).reshape(-1) > 0.5).mean())


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    ca_root = Path(args.causal_adapter_root)
    bench_root = _add_ca_paths(ca_root)
    _patch_safety_checker_compat()
    effectiveness = _load_official_metric(bench_root, "effectiveness_SD", "effectiveness")
    minimality = _load_official_metric(bench_root, "minimality_SD", "minimality")
    from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba  # type: ignore
    from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictors = _build_predictors(bench_root, device)

    parent_cfg_path = Path(args.parent_config)
    if not parent_cfg_path.is_absolute():
        parent_cfg_path = REPO / parent_cfg_path
    parent_cfg = _load_yaml(parent_cfg_path)
    ds = CelebADataset(
        root=parent_cfg["data_root"],
        split=args.split,
        factor_cols=list(CELEBA_ATTR_ORDER),
        image_size=64,
    )
    indices = _load_selection_indices(args.selection_json, args.max_samples)
    if indices is None:
        indices = list(range(len(ds)))
        if args.max_samples > 0:
            indices = indices[: args.max_samples]
    subset: Dataset = FilenameSubset(ds, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    po_gen = None if args.skip_parent_only else ParentOnlyGenerator(parent_cfg_path, args.parent_start_t, device)
    ca_gen = CausalAdapterGenerator(args, device, Path(parent_cfg["data_root"]))
    method_names = ("causal_adapter",) if args.skip_parent_only else ("parent_only_diffusion", "causal_adapter")
    accum: Dict[str, Dict[str, Any]] = {
        name: {
            "eff_predictions": {a: [] for a in COMPLEX_ATTRS},
            "eff_targets": {a: [] for a in COMPLEX_ATTRS},
            "all_real": [],
            "all_fake": [],
            "factuals": [],
            "counterfactuals": [],
            "interventions": [],
        }
        for name in method_names
    }
    results: Dict[str, Any] = {
        "args": vars(args),
        "num_samples": len(indices),
        "intervention_attr": args.intervention_attr,
        "indices": indices,
        "eval_protocol": {
            "real_preprocess": "CenterCrop(150)->256->bicubic64",
            "effectiveness_targets": "causal_adapter_p2p_causal_cond",
            "parent_only_generation": "native_64x64_ddim_unchanged",
        },
        "methods": {},
    }
    for batch in tqdm(loader, desc="compare"):
        real = ca_gen.official_real64(batch)
        ca_cf = ca_gen(batch)
        method_outputs = [("causal_adapter", ca_cf)]
        if po_gen is not None:
            po_cf = {k: ca_cf[k].detach() for k in COMPLEX_ATTRS}
            po_cf["image"] = po_gen(batch)["image"].detach()
            method_outputs.insert(0, ("parent_only_diffusion", po_cf))

        for name, cf in method_outputs:
            e, _raw = effectiveness(cf, unnormalize_celeba, predictors, "celeba")
            slot = accum[name]
            for a in COMPLEX_ATTRS:
                slot["eff_predictions"][a].append(np.asarray(_raw["predictions"][a]))
                slot["eff_targets"][a].append(np.asarray(_raw["targets"][a]))
            fake = cf["image"].detach()
            slot["all_real"].append(real.detach())
            slot["all_fake"].append(fake.detach())
            slot["factuals"].extend(zip(_pixel_embedding(real), (batch["Male"].numpy() > 0).astype(np.int64)))
            intervention_attr = args.intervention_attr
            slot["counterfactuals"].extend(
                zip(_pixel_embedding(fake), cf[intervention_attr].detach().cpu().numpy().astype(np.int64))
            )
            slot["interventions"].extend([intervention_attr] * real.shape[0])

    for name in method_names:
        slot = accum[name]
        min_scores, p1, p2 = minimality(
            slot["factuals"], slot["counterfactuals"], slot["interventions"], bins={}, embedding=""
        )
        results["methods"][name] = {
            "effectiveness": {
                a: _binary_f1_from_arrays(
                    np.concatenate(slot["eff_predictions"][a], axis=0),
                    np.concatenate(slot["eff_targets"][a], axis=0),
                )
                for a in COMPLEX_ATTRS
            },
            "effectiveness_debug": {
                a: {
                    "target_pos_rate": _rate_from_arrays(
                        np.concatenate(slot["eff_targets"][a], axis=0), is_prediction=False
                    ),
                    "pred_pos_rate": _rate_from_arrays(
                        np.concatenate(slot["eff_predictions"][a], axis=0), is_prediction=True
                    ),
                }
                for a in COMPLEX_ATTRS
            },
            "fid": _fid(slot["all_real"], slot["all_fake"], device),
            "minimality": {
                "mean": float(np.mean(min_scores)) if min_scores else None,
                "std": float(np.std(min_scores)) if min_scores else None,
                "prob1_mean": float(np.mean(p1)) if p1 else None,
                "prob2_mean": float(np.mean(p2)) if p2 else None,
            },
        }
        print(name, json.dumps(results["methods"][name], indent=2))

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    if args.output_grid:
        from torchvision.utils import save_image

        real = torch.cat(accum["causal_adapter"]["all_real"], dim=0)[:32].detach().cpu()
        fake = torch.cat(accum["causal_adapter"]["all_fake"], dim=0)[:32].detach().cpu()
        paired = torch.stack((real, fake), dim=1).flatten(0, 1)
        grid_out = Path(args.output_grid)
        grid_out.parent.mkdir(parents=True, exist_ok=True)
        save_image(paired, grid_out, nrow=8, padding=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
