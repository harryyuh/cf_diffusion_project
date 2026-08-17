"""Evaluate SD3/MMDiT LoRA full-causal-prompt counterfactuals on CelebA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset, expand_env_in_cfg
from evaluation.evaluate_latent_prompt_lora_celeba import (
    COMPLEX_ATTRS,
    FilenameSubset,
    _benchmark_only_on_path,
    _binary_f1,
    _build_predictors,
    _causal_adapter_intervene_pm1,
    _fid,
    _labels_pm1,
    _load_causal_adapter_scm,
    _load_metric,
    _prepare_real64,
    _rate,
    _to_01_labels,
)
from utils.visualization import save_image_grid

REPO = Path(__file__).resolve().parents[1]

PHRASE_MAP = {
    "Young": ("young", "older"),
    "Male": ("man", "woman"),
    "No_Beard": ("no beard", "beard"),
    "Bald": ("bald", "with hair"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/latent_dit_lora_celeba_full_causal_four_label.yaml")
    p.add_argument("--lora-dir", default="")
    p.add_argument("--selection-json", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--strength", type=float, default=0.45)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--lora-scale", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--intervention-attr", choices=COMPLEX_ATTRS, required=True)
    p.add_argument("--benchmark-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark")
    p.add_argument("--causal-adapter-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter")
    p.add_argument("--causal-adapter-scm", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt")
    p.add_argument("--output-json", required=True)
    p.add_argument("--grid-dir", default="")
    p.add_argument("--grid-num", type=int, default=32)
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def _load_indices(path: str, n: int) -> List[int]:
    with open(path) as f:
        raw = json.load(f)
    idxs = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
    return idxs[:n]


def _prompts_from_pm1(labels_pm1: torch.Tensor, cfg: Dict[str, Any]) -> List[str]:
    prefix = str(cfg.get("caption_prefix", "a portrait photo of a"))
    label_cols = list(cfg.get("caption_attr_cols") or COMPLEX_ATTRS)
    prompts: List[str] = []
    for i in range(labels_pm1.shape[0]):
        parts = [prefix]
        for key in label_cols:
            if key not in COMPLEX_ATTRS:
                continue
            j = COMPLEX_ATTRS.index(key)
            pos, neg = PHRASE_MAP[key]
            parts.append(pos if float(labels_pm1[i, j]) > 0 else neg)
        prompts.append(", ".join(parts))
    return prompts


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    return data.view(size, size, 3).permute(2, 0, 1).float() / 255.0


class SD3PromptLoraEditor:
    def __init__(self, cfg: Dict[str, Any], lora_dir: Path, device: torch.device, args: argparse.Namespace):
        from diffusers import StableDiffusion3Pipeline

        precision = str(cfg.get("mixed_precision", "bf16")).lower()
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision in ("fp16", "float16") else torch.float32
        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if cfg.get("variant"):
            kwargs["variant"] = cfg["variant"]
        self.pipe = StableDiffusion3Pipeline.from_pretrained(str(cfg["pretrained_model_name_or_path"]), **kwargs).to(device)
        self.pipe.load_lora_weights(str(lora_dir))
        self.pipe.vae.eval()
        self.pipe.transformer.eval()
        self.cfg = cfg
        self.device = device
        self.image_size = int(cfg.get("image_size", 512))
        self.steps = int(args.steps)
        self.strength = float(args.strength)
        self.guidance_scale = float(args.guidance_scale)
        self.lora_scale = float(args.lora_scale)
        self.negative_prompt = str(cfg.get("negative_prompt", "blurry, low resolution, jpeg artifacts, distorted face"))
        self.generator = torch.Generator(device=device).manual_seed(int(args.seed))
        self.image_tfm = transforms.Compose(
            [
                transforms.CenterCrop(150),
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    @torch.no_grad()
    def _encode_prompts(self, prompts: List[str]):
        neg = [self.negative_prompt] * len(prompts)
        max_len = int(self.cfg.get("max_sequence_length", 256))
        pe, ne, pooled, neg_pooled = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            negative_prompt=neg,
            negative_prompt_2=neg,
            negative_prompt_3=neg,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            max_sequence_length=max_len,
        )
        dtype = self.pipe.transformer.dtype
        return pe.to(dtype=dtype), ne.to(dtype=dtype), pooled.to(dtype=dtype), neg_pooled.to(dtype=dtype)

    @torch.no_grad()
    def encode_images(self, data_root: Path, filenames) -> torch.Tensor:
        xs = [self.image_tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB")) for fn in filenames]
        x = torch.stack(xs).to(self.device, dtype=self.pipe.vae.dtype) * 2.0 - 1.0
        latents = self.pipe.vae.encode(x).latent_dist.sample()
        return (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

    def _active_schedule(self):
        self.pipe.scheduler.set_timesteps(self.steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        sigmas = self.pipe.scheduler.sigmas.to(device=self.device, dtype=self.pipe.transformer.dtype)
        n_active = max(1, min(self.steps, int(round(self.steps * max(0.0, min(1.0, self.strength))))))
        start = self.steps - n_active
        return timesteps[start:], sigmas[start : start + n_active + 1]

    def _noise_pred(self, latents, t, prompt_embeds, neg_embeds, pooled, neg_pooled):
        timestep = t.expand(latents.shape[0])
        if self.guidance_scale <= 1.0:
            kwargs = {
                "hidden_states": latents,
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled,
                "return_dict": False,
            }
            try:
                return self.pipe.transformer(**kwargs, joint_attention_kwargs={"scale": self.lora_scale})[0]
            except TypeError:
                return self.pipe.transformer(**kwargs)[0]
        x = torch.cat([latents, latents], dim=0)
        tt = t.expand(x.shape[0])
        ee = torch.cat([neg_embeds, prompt_embeds], dim=0)
        pp = torch.cat([neg_pooled, pooled], dim=0)
        kwargs = {
            "hidden_states": x,
            "timestep": tt,
            "encoder_hidden_states": ee,
            "pooled_projections": pp,
            "return_dict": False,
        }
        try:
            pred = self.pipe.transformer(**kwargs, joint_attention_kwargs={"scale": self.lora_scale})[0]
        except TypeError:
            pred = self.pipe.transformer(**kwargs)[0]
        uncond, cond = pred.chunk(2)
        return uncond + self.guidance_scale * (cond - uncond)

    @torch.no_grad()
    def invert(self, latents0, prompt_embeds, neg_embeds, pooled, neg_pooled, timesteps, sigmas):
        sample = latents0.to(dtype=self.pipe.transformer.dtype)
        # FlowMatch Euler denoising integrates from high sigma to low sigma.
        # Inversion follows the same ODE in the reverse direction under source condition.
        inv_timesteps = list(reversed(timesteps))
        inv_sigmas = list(reversed(sigmas))
        for i, t in enumerate(inv_timesteps):
            sigma_low = inv_sigmas[i]
            sigma_high = inv_sigmas[i + 1]
            pred = self._noise_pred(sample, t, prompt_embeds, neg_embeds, pooled, neg_pooled)
            sample = sample + (sigma_high - sigma_low) * pred
        return sample

    @torch.no_grad()
    def sample(self, latents_t, prompt_embeds, neg_embeds, pooled, neg_pooled, timesteps, sigmas):
        sample = latents_t.to(dtype=self.pipe.transformer.dtype)
        for i, t in enumerate(timesteps):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            pred = self._noise_pred(sample, t, prompt_embeds, neg_embeds, pooled, neg_pooled)
            sample = sample + (sigma_next - sigma_curr) * pred
        return sample

    @torch.no_grad()
    def decode64(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        img = self.pipe.vae.decode(latents.to(dtype=self.pipe.vae.dtype), return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)
        return F.interpolate(img.float(), size=(64, 64), mode="bicubic", align_corners=False).clamp(0, 1)

    @torch.no_grad()
    def edit(self, data_root: Path, filenames, c_orig_pm1: torch.Tensor, c_cf_pm1: torch.Tensor) -> torch.Tensor:
        orig_prompts = _prompts_from_pm1(c_orig_pm1, self.cfg)
        cf_prompts = _prompts_from_pm1(c_cf_pm1, self.cfg)
        pe_orig, ne_orig, pooled_orig, neg_pooled_orig = self._encode_prompts(orig_prompts)
        pe_cf, ne_cf, pooled_cf, neg_pooled_cf = self._encode_prompts(cf_prompts)
        lat0 = self.encode_images(data_root, filenames)
        timesteps, sigmas = self._active_schedule()
        lat_t = self.invert(lat0, pe_orig, ne_orig, pooled_orig, neg_pooled_orig, timesteps, sigmas)
        lat_cf = self.sample(lat_t, pe_cf, ne_cf, pooled_cf, neg_pooled_cf, timesteps, sigmas)
        return self.decode64(lat_cf)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else REPO / cfg_path
    cfg = _load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(cfg["data_root"])
    lora_dir = Path(args.lora_dir) if args.lora_dir else Path(cfg["output_dir"]) / str(cfg["run_name"]) / "lora"
    indices = _load_indices(args.selection_json, args.max_samples)
    ds = CelebADataset(root=cfg["data_root"], split=args.split, factor_cols=list(CELEBA_ATTR_ORDER), image_size=64)
    loader = DataLoader(FilenameSubset(ds, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    bench_root = Path(args.benchmark_root)
    effectiveness = _load_metric(bench_root, "effectiveness_SD", "effectiveness")
    with _benchmark_only_on_path(bench_root):
        from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba  # type: ignore
    predictors = _build_predictors(bench_root, device)
    scm = _load_causal_adapter_scm(Path(args.causal_adapter_root), Path(args.causal_adapter_scm), device)
    intervention_idx = COMPLEX_ATTRS.index(args.intervention_attr)
    editor = SD3PromptLoraEditor(cfg, lora_dir, device, args)

    preds = {a: [] for a in COMPLEX_ATTRS}
    tgts = {a: [] for a in COMPLEX_ATTRS}
    real_batches = []
    fake_batches = []
    grid_tiles = []
    orig_tfm = transforms.Compose(
        [
            transforms.CenterCrop(150),
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    seen = 0
    for batch in tqdm(loader, desc=f"sd3 full causal eval do({args.intervention_attr})"):
        filenames = batch["filename"]
        c_orig = _labels_pm1(batch, device)
        c_cf = _causal_adapter_intervene_pm1(scm, c_orig, intervention_idx)
        fake = editor.edit(data_root, filenames, c_orig, c_cf)
        real = _prepare_real64(data_root, filenames, device)
        cf = _to_01_labels(c_cf)
        cf["image"] = fake
        _, raw = effectiveness(cf, unnormalize_celeba, predictors, "celeba")
        for a in COMPLEX_ATTRS:
            preds[a].append(np.asarray(raw["predictions"][a]))
            tgts[a].append(np.asarray(raw["targets"][a]))
        real_batches.append(real.detach())
        fake_batches.append(fake.detach())
        if args.grid_dir and seen < args.grid_num:
            for j, fn in enumerate(filenames):
                if seen >= args.grid_num:
                    break
                orig = orig_tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB"))
                grid_tiles.extend([orig.cpu(), fake[j].detach().cpu()])
                seen += 1

    result = {
        "args": vars(args),
        "num_samples": len(indices),
        "indices": indices,
        "method": "sd3_full_causal_prompt_lora_flow_inversion",
        "eval_protocol": {
            "preprocess": "CenterCrop(150)->Resize(512) for SD3 VAE encode/decode, eval resize 64",
            "target": f"Causal-Adapter SCM do({args.intervention_attr} flip)",
            "condition": "FlowMatch inversion with factual four-label prompt, denoising with CA SCM counterfactual four-label prompt",
            "inversion": {
                "type": "deterministic FlowMatch Euler ODE inversion",
                "strength": args.strength,
                "active_steps": max(1, min(args.steps, int(round(args.steps * max(0.0, min(1.0, args.strength)))))),
                "guidance_scale": args.guidance_scale,
            },
        },
        "effectiveness": {a: _binary_f1(np.concatenate(preds[a], 0), np.concatenate(tgts[a], 0)) for a in COMPLEX_ATTRS},
        "effectiveness_debug": {
            a: {
                "target_pos_rate": _rate(np.concatenate(tgts[a], 0), False),
                "pred_pos_rate": _rate(np.concatenate(preds[a], 0), True),
            }
            for a in COMPLEX_ATTRS
        },
        "fid": _fid(real_batches, fake_batches, device),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    if args.grid_dir and grid_tiles:
        grid_dir = Path(args.grid_dir) / f"grids_do_{args.intervention_attr}"
        save_image_grid(torch.stack(grid_tiles), nrow=2, path=grid_dir / "grid_32_orig_cf.png", cmap=None, dpi=100.0)
        with open(grid_dir / "grid_32_metadata.json", "w") as f:
            json.dump({"output_json": str(out), "intervention_attr": args.intervention_attr, "indices": indices[: args.grid_num]}, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
