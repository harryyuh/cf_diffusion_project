"""
NeurIPS counterfactual-benchmark metrics on CelebA **complex** (Young, Male, No_Beard, Bald),
using **pretrained** ``CelebaComplexClassifier`` checkpoints from the benchmark repo.

Your cf-diffusion-celeba model is assumed to support **Male-only** counterfactuals in image space
(``parent_keys: [Male]``). By default we only randomize / ``do`` **Male** for FID and minimality,
and only report effectiveness for ``do(Male)``. Set ``--intervention-attributes`` / ``--effectiveness-parents``
to include other parents only if your adapter implements them (otherwise the script keeps the
factual image and overwrites attribute tensors — not comparable to the paper).

Requirements
------------
- Install counterfactual-benchmark ``requirements.txt`` (torchmetrics, …) in the same conda env.
- CelebA root must match torchvision layout (``img_align_celeba``, ``list_attr_celeba.txt``,
  ``list_eval_partition.txt``).
- Download benchmark **complex** classifier checkpoints under
  ``counterfactual_benchmark/methods/deepscm/checkpoints/celeba/complex/trained_classifiers/``.

Example
-------
  cd /home/yu1331/cf-diffusion-celeba
  python evaluation/run_celeba_complex_benchmark_metrics.py \\
    --cf-config configs/diffusion_celeba.yaml \\
    --celeba-root /scratch/gilbreth/yu1331/datasets/celebA \\
    --benchmark-root /path/to/counterfactual-benchmark/counterfactual_benchmark \\
    --classifier-json /path/to/.../configs/celeba/complex/classifier.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import yaml
from json import load as json_load
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(_REPO)
    if root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def _benchmark_only_on_path(benchmark_pkg: Path) -> Iterator[None]:
    """Import benchmark code without this repo's ``models`` package shadowing ``models.classifiers``."""
    bench = str(benchmark_pkg.resolve())
    repo = str(_REPO)
    saved = sys.path.copy()
    sys.path = [bench] + [p for p in sys.path if p not in (bench, repo)]
    try:
        yield
    finally:
        sys.path[:] = saved


def _import_benchmark_symbol(benchmark_pkg: Path, module_rel: str, symbol: str) -> Any:
    path = benchmark_pkg / f"{module_rel.replace('.', '/')}.py"
    if not path.is_file():
        raise ImportError(f"Benchmark module not found: {path}")
    mod_name = f"_bench_{module_rel.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)

COMPLEX_ATTRS: List[str] = ["Young", "Male", "No_Beard", "Bald"]
rng = np.random.default_rng()


def _expand_env_in_cfg(cfg: Dict[str, Any]) -> None:
    for k, v in list(cfg.items()):
        if isinstance(v, str) and "$" in v:
            cfg[k] = os.path.expandvars(v)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg:
        _expand_env_in_cfg(cfg)
    return cfg


def different_value(possible_values, value, bins, attribute):
    if bins is not None and attribute in bins:
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    return possible_values != value


def produce_counterfactuals(
    factual_batch: Dict[str, torch.Tensor],
    scm: nn.Module,
    do_parent: str,
    intervention_source,
    force_change: bool,
    possible_values,
    device: str = "cuda",
    bins=None,
):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    if force_change:
        pv_t = possible_values[do_parent]
        pv = pv_t.detach().cpu().numpy().reshape(-1)
        values = factual_batch[do_parent].detach().cpu().numpy().reshape(-1)
        if do_parent in ("digit", "apoE", "slice"):
            raise NotImplementedError("digit/apoE/slice sampling not used for CelebA complex")
        chosen: List[float] = []
        for value in values:
            mask = different_value(pv, value, bins, do_parent)
            cand = pv[mask] if np.any(mask) else pv
            chosen.append(float(rng.choice(cand)))
        interventions = {
            do_parent: torch.tensor(chosen, device=device, dtype=factual_batch[do_parent].dtype).view(-1, 1)
        }
    else:
        bsz = factual_batch["image"].shape[0]
        idxs = torch.randperm(len(intervention_source))[:bsz]
        interventions = {
            do_parent: torch.cat(
                [intervention_source[int(i)][do_parent] for i in idxs],
                dim=0,
            ).to(device)
        }
    noise = scm.encode(**factual_batch)
    return scm.decode(interventions, **noise)


def _import_benchmark(benchmark_pkg: Path):
    with _benchmark_only_on_path(benchmark_pkg):
        from datasets.celeba.dataset import Celeba, unnormalize as unnormalize_celeba  # type: ignore
        from datasets.transforms import ReturnDictTransform  # type: ignore
        from evaluation.metrics.composition import composition as composition_metric  # type: ignore
        from evaluation.metrics.effectiveness import effectiveness as effectiveness_metric  # type: ignore
        from evaluation.metrics.fid import fid as fid_metric  # type: ignore
        from evaluation.metrics.minimality import minimality as minimality_metric  # type: ignore

    return {
        "Celeba": Celeba,
        "unnormalize_celeba": unnormalize_celeba,
        "ReturnDictTransform": ReturnDictTransform,
        "composition": composition_metric,
        "effectiveness": effectiveness_metric,
        "fid": fid_metric,
        "minimality": minimality_metric,
    }


def _parent_1d(batch: Dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    t = batch[key].to(device).float()
    if t.dim() == 1:
        t = t.unsqueeze(1)
    return t


def _diffusion_ckpt_subdir(cfg: Dict[str, Any]) -> str:
    """Match ``training/train_diffusion.py`` checkpoint layout."""
    if cfg.get("checkpoint_subdir"):
        return str(cfg["checkpoint_subdir"])
    use_vae = bool(cfg.get("use_vae_condition", True))
    if not use_vae:
        return "parent_only"
    if bool(cfg.get("condition_include_z_rest", True)):
        return "with_vae_condition"
    return "with_vae_parent_cond_only"


@torch.no_grad()
def _ddim_cf(
    vae: Optional[nn.Module],
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    x: torch.Tensor,
    f_obs: torch.Tensor,
    f_tgt: torch.Tensor,
    n_parent_dims: int,
    start_t: int,
    device: torch.device,
    *,
    include_z_rest: bool,
) -> torch.Tensor:
    if f_obs.dim() == 1:
        f_obs = f_obs.unsqueeze(1)
    if f_tgt.dim() == 1:
        f_tgt = f_tgt.unsqueeze(1)
    if include_z_rest:
        assert vae is not None
        mu, _ = vae.encode(x)
        z_rest = mu[:, n_parent_dims:]
        cond_enc = torch.cat([z_rest, f_obs], dim=1)
        cond_dec = torch.cat([z_rest, f_tgt], dim=1)
    else:
        cond_enc = f_obs
        cond_dec = f_tgt
    x_t = diffusion.ddim_encode_to_xt(unet, x, start_t, cond_enc, device)
    out = diffusion.p_sample_loop_from_xt(unet, x_t, start_t, cond_dec, device)
    return out.clamp(0.0, 1.0)


class CelebACfDiffusionAdapter(nn.Module):
    """
    Benchmark ``SCM``-like ``encode`` / ``decode`` for cf-diffusion-celeba (Male-only image CF).

    ``decode`` with ``repl`` containing only ``Male`` runs DDIM encode/decode with flipped Male.
    Other keys in ``repl`` (or unsupported parents): factual image is kept and only tensors updated.
    """

    def __init__(
        self,
        vae: Optional[nn.Module],
        unet: nn.Module,
        diffusion: GaussianDiffusion,
        n_parent_dims: int,
        device: torch.device,
        start_t: int = 300,
        supported_do: Optional[Set[str]] = None,
        *,
        include_z_rest: bool = True,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.diffusion = diffusion
        self.n_parent_dims = n_parent_dims
        self._device = device
        self.start_t = int(start_t)
        self.supported_do = supported_do or {"Male"}
        self.include_z_rest = bool(include_z_rest)

    def encode(self, **factual_batch: torch.Tensor) -> Dict[str, Any]:
        cache = {k: v for k, v in factual_batch.items()}
        return {"_cf": cache}

    def decode(self, repl: Optional[Dict[str, torch.Tensor]] = None, **kwargs: Any) -> Dict[str, torch.Tensor]:
        factual = kwargs["_cf"]
        device = next(self.unet.parameters()).device
        x = factual["image"].to(device)
        f_obs = _parent_1d(factual, "Male", device)

        if repl is None or "Male" not in repl:
            f_tgt = f_obs
        else:
            f_tgt = repl["Male"].to(device=device, dtype=f_obs.dtype)
            if f_tgt.dim() == 1:
                f_tgt = f_tgt.unsqueeze(1)

        if repl is not None and set(repl.keys()) <= self.supported_do and "Male" in repl:
            x_out = _ddim_cf(self.vae, self.unet, self.diffusion, x, f_obs, f_tgt, self.n_parent_dims, self.start_t, device, include_z_rest=self.include_z_rest)
        elif repl is None or len(repl) == 0:
            x_out = _ddim_cf(self.vae, self.unet, self.diffusion, x, f_obs, f_obs, self.n_parent_dims, self.start_t, device, include_z_rest=self.include_z_rest)
        else:
            x_out = x.clone()

        out: Dict[str, torch.Tensor] = {}
        for k, v in factual.items():
            out[k] = v.to(device).clone() if isinstance(v, torch.Tensor) else v
        out["image"] = x_out
        if repl is not None:
            for k, v in repl.items():
                out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return out


def _load_cf_stack(cfg: Dict[str, Any], device: torch.device) -> tuple:
    _ensure_repo_on_path()
    from models.diffusion_unet_factory import build_conditional_diffusion_unet  # noqa: E402
    from models.vae_factory import build_vae_from_train_cfg  # noqa: E402
    from utils.checkpoint import load_checkpoint  # noqa: E402
    from utils.diffusion_utils import DiffusionConfig, GaussianDiffusion  # noqa: E402

    use_vae = bool(cfg.get("use_vae_condition", True))
    condition_include_z_rest = bool(cfg.get("condition_include_z_rest", True)) if use_vae else False
    vae = None
    n_parent_dims = 0
    rest_dim = 0
    if use_vae:
        vpath = _REPO / cfg.get("vae_config", "configs/vae_celeba.yaml")
        vcfg = _load_yaml(vpath)
        n_parent_dims = int(vcfg.get("n_parent_dims", 0))
        rest_dim = int(vcfg["latent_dim"]) - n_parent_dims
        vae = build_vae_from_train_cfg(vcfg)
        load_checkpoint(Path(cfg["vae_checkpoint"]), model=vae, map_location="cpu")
        vae = vae.to(device).eval()
        for p in vae.parameters():
            p.requires_grad = False

    parent_keys: List[str] = list(cfg.get("parent_keys") or ["Male"])
    if set(parent_keys) != {"Male"}:
        print(
            "WARN: benchmark adapter is verified for parent_keys=[Male]; got {parent_keys}.",
            file=sys.stderr,
        )
    parent_dim = len(parent_keys)
    if use_vae and condition_include_z_rest:
        cond_dim = rest_dim + parent_dim
    else:
        cond_dim = parent_dim
    unet = build_conditional_diffusion_unet(cfg, cond_dim).to(device)
    subdir = _diffusion_ckpt_subdir(cfg)
    ckpt_path = Path(cfg["output_dir"]) / subdir / "checkpoints" / "diffusion_best.pt"
    if not ckpt_path.is_file():
        alt = ckpt_path.parent / "diffusion_last.pt"
        ckpt_path = alt if alt.is_file() else ckpt_path
    ck = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ck["unet_state_dict"])
    unet.eval()

    diff = GaussianDiffusion(
        DiffusionConfig(
            timesteps=int(cfg.get("timesteps", 1000)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 0.02)),
            ddim_eta=float(cfg.get("ddim_eta", 0.0)),
        )
    )
    for a in ("betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
        setattr(diff, a, getattr(diff, a).to(device))
    return vae, unet, diff, n_parent_dims, condition_include_z_rest


def _pixel_embedding_fn(unnormalize_fn: Any):
    def fn(x: torch.Tensor, _cond: Any = None) -> np.ndarray:
        return unnormalize_fn(x, "image").cpu().numpy()

    return fn


def _resolve_embedding_helpers(
    benchmark_pkg: Path,
    embedding_arg: Optional[str],
    clf_path: str,
    unnormalize_fn: Any,
) -> tuple[Any, Any]:
    if not embedding_arg:
        return None, _pixel_embedding_fn(unnormalize_fn)
    with _benchmark_only_on_path(benchmark_pkg):
        from evaluation.embeddings.embeddings import get_embedding_fn, get_embedding_model  # type: ignore

    model = get_embedding_model(embedding_arg, pretrained_vgg=True, classifier_config=clf_path)
    fn = get_embedding_fn(embedding_arg, unnormalize_fn, model)
    return model, fn


def _ensure_classifier_json(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for k in list(cfg.get("attribute_size", {}).keys()):
        cfg.setdefault(f"{k}_num_out", int(cfg["attribute_size"][k]))
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CelebA complex benchmark metrics for cf-diffusion-celeba.")
    p.add_argument("--cf-config", type=str, default="configs/diffusion_celeba.yaml")
    p.add_argument(
        "--benchmark-root",
        type=str,
        default=str(_REPO.parent / "counterfactual-benchmark" / "counterfactual_benchmark"),
    )
    p.add_argument(
        "--classifier-json",
        type=str,
        default="",
        help="Path to counterfactual_benchmark/.../configs/celeba/complex/classifier.json",
    )
    p.add_argument("--celeba-root", type=str, required=True, help="CelebA root (img_align_celeba + list files).")
    p.add_argument(
        "--benchmark-data-dir",
        type=str,
        default="",
        help="Directory passed to benchmark ``Celeba`` (often .../counterfactual_benchmark/datasets/celeba/data). "
        "If empty, uses ``--celeba-root`` for both.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--start-t", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=["composition", "effectiveness", "fid", "minimality"],
        default=["composition", "effectiveness", "fid", "minimality"],
    )
    p.add_argument("--cycles", nargs="+", type=int, default=[1, 10])
    p.add_argument("--embeddings", type=str, default="", choices=["", "vgg", "clfs", "vae", "lpips", "clip"])
    p.add_argument(
        "--intervention-attributes",
        nargs="+",
        default=["Male"],
        help="Random parent pool for FID/minimality (paper uses all four for complex).",
    )
    p.add_argument(
        "--effectiveness-parents",
        nargs="+",
        default=["Male"],
        help="Which do() parents to run for effectiveness.",
    )
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_repo_on_path()
    from utils.seed import set_seed  # noqa: E402

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bench_root = Path(args.benchmark_root).resolve()
    if not bench_root.is_dir():
        sys.exit(f"benchmark-root not found: {bench_root}")

    clf_path = Path(args.classifier_json).resolve() if args.classifier_json.strip() else (
        bench_root.parent / "counterfactual_benchmark" / "methods" / "deepscm" / "configs" / "celeba" / "complex" / "classifier.json"
    )
    if not clf_path.is_file():
        clf_path = bench_root / "methods" / "deepscm" / "configs" / "celeba" / "complex" / "classifier.json"
    if not clf_path.is_file():
        sys.exit(f"classifier-json not found: {clf_path}")

    M = _import_benchmark(bench_root)

    cf_cfg_path = (_REPO / args.cf_config).resolve()
    cf_cfg = _load_yaml(cf_cfg_path)

    device = torch.device(cf_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    vae, unet, diffusion, n_parent_dims, include_z_rest = _load_cf_stack(cf_cfg, device)
    adapter = CelebACfDiffusionAdapter(
        vae,
        unet,
        diffusion,
        n_parent_dims,
        device,
        start_t=args.start_t,
        supported_do={"Male"},
        include_z_rest=include_z_rest,
    ).to(device)

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
    for atr in attribute_size.keys():
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
        mod.to(device)
        mod.eval()

    emb_arg: Optional[str] = args.embeddings if args.embeddings else None
    emb_model, emb_fn = _resolve_embedding_helpers(bench_root, emb_arg, str(clf_path), M["unnormalize_celeba"])

    bs = int(args.batch_size)
    results: Dict[str, Any] = {}
    produce_cf = produce_counterfactuals

    if "composition" in args.metrics:
        scores_all = []
        loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        for batch in tqdm(loader, desc="composition"):
            b = {k: v.to(device) for k, v in batch.items()}
            sc, _ = M["composition"](b, M["unnormalize_celeba"], adapter, cycles=args.cycles, embedding=emb_arg, embedding_fn=emb_fn)
            scores_all.append(sc)
        results["composition"] = {
            str(c): float(np.mean(np.concatenate([s[c] for s in scores_all]))) for c in args.cycles
        }
        print("composition:", json.dumps(results["composition"], indent=2))

    if "effectiveness" in args.metrics:
        eff: Dict[str, Any] = {}
        loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        for do_parent in args.effectiveness_parents:
            scores = {a: [] for a in attribute_size}
            for batch in tqdm(loader, desc=f"effectiveness do({do_parent})"):
                cf = produce_cf(
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
            eff[do_parent] = {a: float(np.mean(scores[a])) for a in attribute_size if len(scores[a])}
            print(f"effectiveness do({do_parent}):", eff[do_parent])
        results["effectiveness"] = eff

    if "fid" in args.metrics:
        real_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        cfs = []
        for batch in tqdm(test_loader, desc="FID CF"):
            do_parent = random.choice(args.intervention_attributes)
            cf = produce_cf(
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
            cf = produce_cf(
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
