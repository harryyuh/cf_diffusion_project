"""
Run the existing CelebA complex counterfactual-benchmark metrics with Causal-Adapter.

This is a thin bridge around the project-local
``evaluation/run_celeba_complex_benchmark_metrics.py``. The benchmark loop,
classifier loading, metrics, and JSON output stay the same; only the model
adapter changes from the local cf-diffusion DDIM model to the official
Causal-Adapter inference functions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from json import load as json_load
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
_OLD_EVAL = _REPO / "evaluation" / "run_celeba_complex_benchmark_metrics.py"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.run_celeba_complex_benchmark_metrics import (  # noqa: E402
    COMPLEX_ATTRS,
    _ensure_classifier_json,
    _import_benchmark,
    _import_benchmark_symbol,
    _resolve_embedding_helpers,
    produce_counterfactuals,
)

ATTR_TO_INDEX = {name: i for i, name in enumerate(COMPLEX_ATTRS)}


def _add_causal_adapter_to_path(causal_adapter_root: Path) -> None:
    root = str(causal_adapter_root.resolve())
    notebooks = str((causal_adapter_root / "notebook_benchmarks").resolve())
    for p in (notebooks, root):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _to_causal_adapter_image(x01: torch.Tensor, size: int) -> torch.Tensor:
    """Benchmark [0, 1] BCHW image -> Causal-Adapter [-1, 1] at ``size``."""
    if x01.shape[-1] != size or x01.shape[-2] != size:
        x01 = F.interpolate(x01, size=(size, size), mode="bilinear", align_corners=False)
    return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)


def _from_pil_list(images: Any, device: torch.device, out_size: int) -> torch.Tensor:
    """Causal-Adapter PIL output -> benchmark [0, 1] BCHW image."""
    from torchvision.transforms.functional import pil_to_tensor

    img = images[-1] if isinstance(images, list) else images
    t = pil_to_tensor(img).float().div(255.0).unsqueeze(0).to(device)
    if t.shape[-1] != out_size or t.shape[-2] != out_size:
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return t.clamp(0.0, 1.0)


class CausalAdapterCelebABenchmarkAdapter(nn.Module):
    """Benchmark SCM-like adapter backed by official Causal-Adapter inference."""

    def __init__(
        self,
        *,
        causal_adapter_root: Path,
        base_model_path: str,
        controlnet_path: str,
        text_embedding_path: str,
        scm_path: Optional[str],
        device: torch.device,
        ca_size: int = 256,
        benchmark_size: int = 64,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        invert_guidance_scale: float = 1.0,
        blend_word: bool = True,
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _add_causal_adapter_to_path(causal_adapter_root)

        from inference_utils import load_causal_adapter  # type: ignore
        from causal_modules.ddim_modules import P2P_editing  # type: ignore

        self._p2p_editing = P2P_editing
        self.assets = load_causal_adapter(
            "celeA_complex",
            base_model_path=base_model_path,
            controlnet_path=controlnet_path,
            text_embedding_path=text_embedding_path,
            scm_path=scm_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        self.device = device
        self.ca_size = int(ca_size)
        self.benchmark_size = int(benchmark_size)
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.invert_guidance_scale = float(invert_guidance_scale)
        self.blend_word = bool(blend_word)
        self.blend_presets = {
            0: dict(blend_params={"start_blend": 0.0, "th": (0.3, 0.3)}, cross_replace_steps=0.1, self_replace_steps=0.1),
            1: dict(blend_params={"start_blend": 0.0, "th": (0.3, 0.5)}, cross_replace_steps=0.2, self_replace_steps=0.2),
            2: dict(blend_params={"start_blend": 0.0, "th": (0.3, 0.5)}, cross_replace_steps=0.0, self_replace_steps=0.0),
            3: dict(blend_params={"start_blend": 0.0, "th": (0.5, 0.3)}, cross_replace_steps=0.0, self_replace_steps=0.0),
        }

    def encode(self, **factual_batch: torch.Tensor) -> Dict[str, Any]:
        return {"_ca_factual": {k: v for k, v in factual_batch.items()}}

    @torch.no_grad()
    def decode(self, repl: Optional[Dict[str, torch.Tensor]] = None, **kwargs: Any) -> Dict[str, torch.Tensor]:
        factual = kwargs["_ca_factual"]
        device = self.device
        out: Dict[str, torch.Tensor] = {
            k: v.to(device).clone() if isinstance(v, torch.Tensor) else v for k, v in factual.items()
        }
        if repl:
            for k, v in repl.items():
                out[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        if not repl:
            return out
        active = [k for k in repl if k in ATTR_TO_INDEX]
        if len(active) != 1:
            return out

        attr = active[0]
        inter_id = ATTR_TO_INDEX[attr]
        cfg = self.blend_presets[inter_id]
        pipe = self.assets.pipe
        xs = factual["image"].to(device)

        generated: List[torch.Tensor] = []
        for i in range(xs.shape[0]):
            label = torch.cat([factual[a][i : i + 1].to(device).float().view(1, 1) for a in COMPLEX_ATTRS], dim=1)
            target_value = repl[attr][i : i + 1].to(device).float().view(-1)[0]
            image_t = _to_causal_adapter_image(xs[i : i + 1], self.ca_size)
            images, *_ = self._p2p_editing(
                pipe,
                image_t,
                label,
                self.assets.presudo_token_ids,
                self.assets.prompt,
                self.assets.presudo_list,
                num_steps=self.num_steps,
                invert_guidance_scale=self.invert_guidance_scale,
                set_guidance_scale=self.guidance_scale,
                intervention_indx=inter_id,
                intervention_values=target_value,
                return_PIL=True,
                blend_word=self.blend_word,
                blend_params=cfg["blend_params"],
                disentangle=False,
                cross_replace_steps=cfg["cross_replace_steps"],
                self_replace_steps=cfg["self_replace_steps"],
                DSCM_labels=None,
            )
            generated.append(_from_pil_list(images, device, self.benchmark_size))
        out["image"] = torch.cat(generated, dim=0)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CelebA complex benchmark metrics for official Causal-Adapter.")
    p.add_argument("--causal-adapter-root", default=str(_REPO / "external" / "Causal-Adapter"))
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--controlnet-path", required=True)
    p.add_argument("--text-embedding-path", required=True)
    p.add_argument("--scm-path", default="")
    p.add_argument("--benchmark-root", default=str(_REPO.parent / "counterfactual-benchmark" / "counterfactual_benchmark"))
    p.add_argument("--classifier-json", default="")
    p.add_argument("--celeba-root", required=True)
    p.add_argument("--benchmark-data-dir", default="")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ca-size", type=int, default=256)
    p.add_argument("--benchmark-size", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--invert-guidance-scale", type=float, default=1.0)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--metrics", nargs="+", choices=["composition", "effectiveness", "fid", "minimality"], default=["composition", "effectiveness", "fid", "minimality"])
    p.add_argument("--cycles", nargs="+", type=int, default=[1, 10])
    p.add_argument("--embeddings", default="", choices=["", "vgg", "clfs", "vae", "lpips", "clip"])
    p.add_argument("--intervention-attributes", nargs="+", default=COMPLEX_ATTRS)
    p.add_argument("--effectiveness-parents", nargs="+", default=COMPLEX_ATTRS)
    p.add_argument("--output-json", default="")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bench_root = Path(args.benchmark_root).resolve()
    if not bench_root.is_dir():
        sys.exit(f"benchmark-root not found: {bench_root}")

    clf_path = Path(args.classifier_json).resolve() if args.classifier_json.strip() else (
        bench_root / "methods" / "deepscm" / "configs" / "celeba" / "complex" / "classifier.json"
    )
    if not clf_path.is_file():
        sys.exit(f"classifier-json not found: {clf_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = CausalAdapterCelebABenchmarkAdapter(
        causal_adapter_root=Path(args.causal_adapter_root),
        base_model_path=args.base_model_path,
        controlnet_path=args.controlnet_path,
        text_embedding_path=args.text_embedding_path,
        scm_path=args.scm_path.strip() or None,
        device=device,
        ca_size=args.ca_size,
        benchmark_size=args.benchmark_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        invert_guidance_scale=args.invert_guidance_scale,
        torch_dtype=_dtype(args.dtype),
    ).to(device)

    M = _import_benchmark(bench_root)
    attribute_size = {k: 1 for k in COMPLEX_ATTRS}
    transform = M["ReturnDictTransform"](attribute_size)
    data_dir = args.benchmark_data_dir.strip() or args.celeba_root
    train_set = M["Celeba"](attribute_size, split="train", transform=transform, data_dir=data_dir)
    test_set = M["Celeba"](attribute_size, split="test", transform=transform, data_dir=data_dir)

    with open(clf_path, "r") as f:
        clf_cfg = _ensure_classifier_json(json_load(f))
    ckpt_dir = clf_cfg["ckpt_path"]
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = str((clf_path.parent / ckpt_dir).resolve())

    CelebaComplexClassifier = _import_benchmark_symbol(
        bench_root, "models.classifiers.celeba_complex_classifier", "CelebaComplexClassifier"
    )
    predictors: Dict[str, nn.Module] = {}
    for atr in attribute_size:
        predictors[atr] = CelebaComplexClassifier(
            attr=atr,
            context_dim=len(list(clf_cfg["anticausal_graph"][atr])),
            num_outputs=int(clf_cfg.get(f"{atr}_num_out", clf_cfg["attribute_size"][atr])),
            lr=float(clf_cfg.get("lr", 1e-4)),
            version=str(clf_cfg.get("version", "standard")),
        )
    for key, mod in predictors.items():
        fname = next((fn for fn in os.listdir(ckpt_dir) if fn.startswith(key)), None)
        if fname is None:
            sys.exit(f"No checkpoint starting with {key!r} in {ckpt_dir}")
        sd = torch.load(os.path.join(ckpt_dir, fname), map_location=device)
        mod.load_state_dict(sd["state_dict"])
        mod.to(device).eval()

    emb_arg: Optional[str] = args.embeddings if args.embeddings else None
    _emb_model, emb_fn = _resolve_embedding_helpers(bench_root, emb_arg, str(clf_path), M["unnormalize_celeba"])

    bs = int(args.batch_size)
    results: Dict[str, Any] = {}

    if "composition" in args.metrics:
        scores_all = []
        loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        for batch in tqdm(loader, desc="composition"):
            b = {k: v.to(device) for k, v in batch.items()}
            sc, _ = M["composition"](b, M["unnormalize_celeba"], adapter, cycles=args.cycles, embedding=emb_arg, embedding_fn=emb_fn)
            scores_all.append(sc)
        results["composition"] = {str(c): float(np.mean(np.concatenate([s[c] for s in scores_all]))) for c in args.cycles}
        print("composition:", json.dumps(results["composition"], indent=2))

    if "effectiveness" in args.metrics:
        eff: Dict[str, Any] = {}
        loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        for do_parent in args.effectiveness_parents:
            scores = {a: [] for a in attribute_size}
            for batch in tqdm(loader, desc=f"effectiveness do({do_parent})"):
                cf = produce_counterfactuals(
                    batch,
                    adapter,
                    do_parent,
                    train_set,
                    force_change=True,
                    possible_values=test_set.possible_values,
                    device=str(device),
                    bins=getattr(test_set, "bins", None),
                )
                e = M["effectiveness"](cf, M["unnormalize_celeba"], predictors, "celeba")
                for a in attribute_size:
                    scores[a].append(e[a])
            eff[do_parent] = {a: float(np.mean(scores[a])) for a in attribute_size if scores[a]}
            print(f"effectiveness do({do_parent}):", eff[do_parent])
        results["effectiveness"] = eff

    if "fid" in args.metrics:
        real_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        cfs = []
        for batch in tqdm(test_loader, desc="FID CF"):
            do_parent = random.choice(args.intervention_attributes)
            cf = produce_counterfactuals(
                batch,
                adapter,
                do_parent,
                intervention_source=train_set,
                force_change=True,
                possible_values=test_set.possible_values,
                bins=getattr(train_set, "bins", None),
                device=str(device),
            )
            cfs.append(cf["image"])
        fid = M["fid"](real_loader, cfs)
        results["fid"] = {"mean": float(np.mean(fid)), "std": float(np.std(fid))}
        print("fid:", results["fid"])

    if "minimality" in args.metrics:
        factuals: List[Any] = []
        counterfactuals: List[Any] = []
        interventions: List[str] = []
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        for batch in tqdm(test_loader, desc="minimality"):
            do_parent = random.choice(args.intervention_attributes)
            cf = produce_counterfactuals(
                batch,
                adapter,
                do_parent,
                intervention_source=train_set,
                force_change=True,
                possible_values=test_set.possible_values,
                bins=getattr(train_set, "bins", None),
                device=str(device),
            )
            ff = emb_fn(batch["image"].to(device), None)
            cf_f = emb_fn(cf["image"], None)
            factuals += list(zip(*(ff, batch[do_parent].detach().cpu().numpy())))
            counterfactuals += list(zip(*(cf_f, cf[do_parent].detach().cpu().numpy())))
            interventions += [do_parent] * batch["image"].shape[0]
        ms, p1, p2 = M["minimality"](
            real=factuals,
            generated=counterfactuals,
            interventions=interventions,
            bins=getattr(train_set, "bins", None) or {},
            embedding=emb_arg or "",
        )
        results["minimality"] = {
            "mean": float(np.mean(ms)),
            "std": float(np.std(ms)),
            "prob1_mean": float(np.mean(p1)),
            "prob2_mean": float(np.mean(p2)),
        }
        print("minimality:", results["minimality"])

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", outp)


if __name__ == "__main__":
    main()
