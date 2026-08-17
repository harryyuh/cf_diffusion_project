"""Evaluate MiniSD LoRA label-global-modulation counterfactuals on CelebA do(Male)."""
from __future__ import annotations

import argparse, importlib.util, json, os, sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
COMPLEX_ATTRS = ["Young", "Male", "No_Beard", "Bald"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/latent_lora_celeba_ca_pipeline_label_film.yaml")
    p.add_argument("--lora-dir", default="")
    p.add_argument("--selection-json", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--benchmark-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark")
    p.add_argument("--scm-checkpoint", default="/scratch/gilbreth/yu1331/ckpts/celeba/full_label_scm/causal_adapter_graph/checkpoints/full_label_scm_best.pt")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    from data.celeba_dataset import expand_env_in_cfg
    expand_env_in_cfg(cfg)
    return cfg


class FilenameSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.filenames = getattr(dataset, "_filenames", None)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx: int):
        src = self.indices[idx]
        item = dict(self.dataset[src])
        item["source_index"] = torch.tensor(src, dtype=torch.long)
        if self.filenames is not None:
            item["filename"] = self.filenames[src]
        return item


class CausalLabelEmbedding(nn.Module):
    def __init__(self, label_dim: int, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(label_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        first = self.net[0]
        return self.net(labels.to(device=first.weight.device, dtype=first.weight.dtype))


def _unet_time_embed_dim(unet) -> int:
    dim = getattr(getattr(unet, "time_embedding", None), "linear_2", None)
    if dim is not None and hasattr(dim, "out_features"):
        return int(dim.out_features)
    cfg = getattr(unet, "config", None)
    block_channels = getattr(cfg, "block_out_channels", None) if cfg is not None else None
    if block_channels:
        return int(block_channels[0]) * 4
    raise ValueError("Cannot infer UNet time embedding dim")


def _pipeline_class(model_family: str):
    if model_family.lower() in ("sd", "sd15", "sd2"):
        from diffusers import StableDiffusionPipeline
        return StableDiffusionPipeline
    raise ValueError("This evaluator currently supports SD/MiniSD pipeline only")


def _load_indices(path: str, n: int) -> List[int]:
    with open(path) as f:
        raw = json.load(f)
    idxs = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
    return idxs[:n]


def _labels_pm1(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.stack([batch[k].to(device).float().view(-1) for k in COMPLEX_ATTRS], dim=1)


def _to_01_labels(labels_pm1: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: (labels_pm1[:, i:i+1] > 0).float() for i, k in enumerate(COMPLEX_ATTRS)}


def _binary_pred_labels(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    if np.nanmin(pred) < 0.0 or np.nanmax(pred) > 1.0:
        return pred > 0.0
    return pred > 0.5


def _binary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    p = _binary_pred_labels(pred).reshape(-1).astype(bool)
    t = (np.asarray(target).reshape(-1) > 0.5)
    tp = np.logical_and(p, t).sum(dtype=np.float64)
    fp = np.logical_and(p, ~t).sum(dtype=np.float64)
    fn = np.logical_and(~p, t).sum(dtype=np.float64)
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0


def _rate(values: np.ndarray, pred: bool) -> float:
    return float((_binary_pred_labels(values) if pred else (np.asarray(values).reshape(-1) > 0.5)).mean())


@contextmanager
def _benchmark_only_on_path(benchmark_pkg: Path) -> Iterator[None]:
    bench = str(benchmark_pkg.resolve())
    repo = str(REPO.resolve())
    cwd = str(Path.cwd().resolve())
    saved_path = sys.path.copy()
    saved_models = {name: mod for name, mod in sys.modules.items() if name == "models" or name.startswith("models.")}
    filtered = []
    for entry in sys.path:
        resolved = cwd if entry == "" else str(Path(entry).resolve()) if entry else entry
        if resolved in (bench, repo, cwd):
            continue
        filtered.append(entry)
    sys.path = [bench] + filtered
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        sys.modules.update(saved_models)
        sys.path[:] = saved_path


def _load_metric(bench_root: Path, module_name: str, attr_name: str):
    path = bench_root / "evaluation" / "metrics" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"bench_{module_name}", path)
    mod = importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(mod)
    return getattr(mod, attr_name)


def _build_predictors(bench_root: Path, device: torch.device):
    cfg_path = bench_root / "methods" / "deepscm" / "configs" / "celeba" / "complex" / "classifier.json"
    with open(cfg_path) as f: cfg = json.load(f)
    ckpt_dir = Path(cfg["ckpt_path"])
    if not ckpt_dir.is_absolute():
        rel = Path(str(ckpt_dir).lstrip("./"))
        candidates = [bench_root / "methods" / "deepscm" / rel, cfg_path.parent / rel]
        ckpt_dir = next((p for p in candidates if p.exists()), candidates[0])
    with _benchmark_only_on_path(bench_root):
        from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier  # type: ignore
    predictors = {}
    for atr in COMPLEX_ATTRS:
        mod = CelebaComplexClassifier(attr=atr, context_dim=len(list(cfg["anticausal_graph"][atr])), num_outputs=int(cfg["attribute_size"][atr]), lr=float(cfg["lr"]), version=str(cfg["version"]))
        fname = next(fn for fn in os.listdir(ckpt_dir) if fn.startswith(atr))
        sd = torch.load(ckpt_dir / fname, map_location=device)
        mod.load_state_dict(sd["state_dict"])
        predictors[atr] = mod.to(device).eval()
    return predictors


def _prepare_real64(data_root: Path, filenames, device: torch.device) -> torch.Tensor:
    tfm = transforms.Compose([transforms.CenterCrop(150), transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
    imgs = []
    for fn in filenames:
        imgs.append(tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB")))
    return torch.stack(imgs).to(device)


class LabelFilmEditor:
    def __init__(self, cfg: Dict[str, Any], lora_dir: Path, device: torch.device, steps: int, guidance_scale: float):
        from diffusers import DDIMScheduler
        self.cfg = cfg; self.device = device; self.steps = int(steps); self.guidance_scale = float(guidance_scale)
        dtype = torch.bfloat16 if str(cfg.get("mixed_precision", "bf16")).lower() == "bf16" else torch.float16
        pipe_cls = _pipeline_class(str(cfg.get("model_family", "sd")))
        self.pipe = pipe_cls.from_pretrained(str(cfg["pretrained_model_name_or_path"]), torch_dtype=dtype).to(device)
        label_cols = list(cfg.get("label_condition_cols") or cfg.get("celeba_attr_cols"))
        self.label_cols = label_cols
        self.pipe.unet.class_embedding = CausalLabelEmbedding(len(label_cols), _unet_time_embed_dim(self.pipe.unet), int(cfg.get("label_condition_hidden_dim", 256))).to(device=device, dtype=dtype)
        self.pipe.load_lora_weights(str(lora_dir))
        state = torch.load(lora_dir / "training_state.pt", map_location=device)
        if "class_embedding_state_dict" in state:
            self.pipe.unet.class_embedding.load_state_dict(state["class_embedding_state_dict"])
        self.scheduler = DDIMScheduler.from_pretrained(str(cfg["pretrained_model_name_or_path"]), subfolder="scheduler")
        self.scheduler.set_timesteps(self.steps, device=device)
        self.pipe.vae.eval(); self.pipe.unet.eval(); self.pipe.text_encoder.eval()
        self.prompt = str(cfg.get("fixed_prompt", "a human face"))
        self.neg_prompt = ""
        self.image_tfm = transforms.Compose([transforms.CenterCrop(150), transforms.Resize((int(cfg.get("image_size", 256)), int(cfg.get("image_size", 256))), interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def _prompt_embeds(self, bsz: int):
        tok = self.pipe.tokenizer([self.prompt] * bsz, padding="max_length", truncation=True, max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
        neg = self.pipe.tokenizer([self.neg_prompt] * bsz, padding="max_length", truncation=True, max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            pe = self.pipe.text_encoder(tok.input_ids.to(self.device))[0]
            ne = self.pipe.text_encoder(neg.input_ids.to(self.device))[0]
        return pe, ne

    def _noise_pred(self, latents, t, cond_pm1, prompt_embeds, neg_embeds):
        if self.guidance_scale == 1.0:
            return self.pipe.unet(latents, t, encoder_hidden_states=prompt_embeds, class_labels=cond_pm1).sample
        x = torch.cat([latents, latents], dim=0)
        cc = torch.cat([cond_pm1, cond_pm1], dim=0)
        ee = torch.cat([neg_embeds, prompt_embeds], dim=0)
        pred = self.pipe.unet(x, t, encoder_hidden_states=ee, class_labels=cc).sample
        uncond, text = pred.chunk(2)
        return uncond + self.guidance_scale * (text - uncond)

    @torch.no_grad()
    def encode_images(self, data_root: Path, filenames) -> torch.Tensor:
        xs = [self.image_tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB")) for fn in filenames]
        x = torch.stack(xs).to(self.device, dtype=self.pipe.vae.dtype)
        lat = self.pipe.vae.encode(x).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        return lat

    @torch.no_grad()
    def invert(self, latents, cond_pm1, prompt_embeds, neg_embeds):
        # DDIM inversion from x_0 to x_T using the observed condition.
        sample = latents
        timesteps = list(self.scheduler.timesteps)
        for i, t in enumerate(reversed(timesteps)):
            next_t = timesteps[len(timesteps) - i - 2] if i < len(timesteps) - 1 else timesteps[0]
            a_t = self.scheduler.alphas_cumprod[t].to(sample.device, sample.dtype)
            a_next = self.scheduler.alphas_cumprod[next_t].to(sample.device, sample.dtype)
            eps = self._noise_pred(sample, t, cond_pm1, prompt_embeds, neg_embeds)
            pred_x0 = (sample - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            sample = a_next.sqrt() * pred_x0 + (1 - a_next).sqrt() * eps
        return sample

    @torch.no_grad()
    def sample(self, latents_t, cond_pm1, prompt_embeds, neg_embeds):
        sample = latents_t
        for t in self.scheduler.timesteps:
            eps = self._noise_pred(sample, t, cond_pm1, prompt_embeds, neg_embeds)
            sample = self.scheduler.step(eps, t, sample).prev_sample
        return sample

    @torch.no_grad()
    def decode64(self, latents):
        img = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return F.interpolate(img.float(), size=(64,64), mode="bicubic", align_corners=False).clamp(0,1)

    @torch.no_grad()
    def edit(self, data_root: Path, filenames, c_orig_pm1, c_cf_pm1):
        lat0 = self.encode_images(data_root, filenames)
        pe, ne = self._prompt_embeds(lat0.shape[0])
        lat_t = self.invert(lat0, c_orig_pm1.to(self.device), pe, ne)
        lat_cf = self.sample(lat_t, c_cf_pm1.to(self.device), pe, ne)
        return self.decode64(lat_cf)


def _fid(real_batches, fake_batches, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric = FrechetInceptionDistance(normalize=True, reset_real_features=False).set_dtype(torch.float32).to(device)
    for x in real_batches: metric.update(x.to(device), real=True)
    for x in fake_batches: metric.update(x.to(device), real=False)
    return float(metric.compute().detach().cpu())


def main():
    args = parse_args()
    sys.path.insert(0, str(REPO))
    from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset
    from models.full_label_scm import FullLabelSCM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(args.config); cfg_path = cfg_path if cfg_path.is_absolute() else REPO / cfg_path
    cfg = _load_yaml(cfg_path)
    run_root = Path(cfg["output_dir"]) / str(cfg.get("run_name"))
    lora_dir = Path(args.lora_dir) if args.lora_dir else run_root / "lora"
    data_root = Path(cfg["data_root"])
    indices = _load_indices(args.selection_json, args.max_samples)
    ds = CelebADataset(root=cfg["data_root"], split=args.split, factor_cols=list(CELEBA_ATTR_ORDER), image_size=64)
    loader = DataLoader(FilenameSubset(ds, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    bench_root = Path(args.benchmark_root)
    effectiveness = _load_metric(bench_root, "effectiveness_SD", "effectiveness")
    with _benchmark_only_on_path(bench_root):
        from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba  # type: ignore
    predictors = _build_predictors(bench_root, device)
    scm = FullLabelSCM().to(device).eval()
    ck = torch.load(args.scm_checkpoint, map_location=device)
    scm.load_state_dict(ck.get("model_state_dict", ck))
    male_idx = COMPLEX_ATTRS.index("Male")
    editor = LabelFilmEditor(cfg, lora_dir, device, args.steps, args.guidance_scale)

    preds = {a: [] for a in COMPLEX_ATTRS}; tgts = {a: [] for a in COMPLEX_ATTRS}
    real_batches = []; fake_batches = []; mse_vals = []
    for batch in tqdm(loader, desc="label-film eval"):
        filenames = batch["filename"]
        c_orig = _labels_pm1(batch, device)
        target_male = -c_orig[:, male_idx]
        c_cf = torch.cat([
            scm.intervene_pm1(c_orig[i:i+1], male_idx, target_male[i].item())
            for i in range(c_orig.shape[0])
        ], dim=0)
        fake = editor.edit(data_root, filenames, c_orig, c_cf)
        real = _prepare_real64(data_root, filenames, device)
        cf = _to_01_labels(c_cf)
        cf["image"] = fake
        _, raw = effectiveness(cf, unnormalize_celeba, predictors, "celeba")
        for a in COMPLEX_ATTRS:
            preds[a].append(np.asarray(raw["predictions"][a])); tgts[a].append(np.asarray(raw["targets"][a]))
        real_batches.append(real.detach()); fake_batches.append(fake.detach())
        mse_vals.extend(((real - fake) ** 2).mean(dim=(1,2,3)).detach().cpu().numpy().tolist())

    result = {
        "args": vars(args), "num_samples": len(indices), "indices": indices,
        "eval_protocol": {"preprocess": "CenterCrop(150)->Resize(256), eval resize 64", "target": "full_label_scm.intervene_pm1 do(Male flip)", "condition": "inversion c_orig, sampling c_cf"},
        "method": "minisd_ca_pipeline_label_film",
        "effectiveness": {a: _binary_f1(np.concatenate(preds[a],0), np.concatenate(tgts[a],0)) for a in COMPLEX_ATTRS},
        "effectiveness_debug": {a: {"target_pos_rate": _rate(np.concatenate(tgts[a],0), False), "pred_pos_rate": _rate(np.concatenate(preds[a],0), True)} for a in COMPLEX_ATTRS},
        "fid": _fid(real_batches, fake_batches, device),
        "pixel_mse_minimality": {"mean": float(np.mean(mse_vals)), "std": float(np.std(mse_vals))},
    }
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
