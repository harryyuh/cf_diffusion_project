"""Evaluate token-only causal cross-attention counterfactuals on CelebA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset
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
    _load_yaml,
    _prepare_real64,
    _rate,
    _to_01_labels,
)
from models.causal_token_mapper import CausalTokenMapper

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/token_only_celeba_ca_pipeline_four_label.yaml")
    p.add_argument("--mapper-dir", required=True)
    p.add_argument("--lora-dir", default="")
    p.add_argument("--selection-json", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--editing-mode", choices=["standard", "p2p_like"], default="standard")
    p.add_argument("--p2p-cross-replace-steps", type=float, default=0.2)
    p.add_argument("--p2p-self-replace-steps", type=float, default=0.2)
    p.add_argument("--p2p-late-target-scale", type=float, default=0.35)
    p.add_argument("--p2p-source-latent-blend", type=float, default=0.25)
    p.add_argument("--intervention-attr", choices=COMPLEX_ATTRS, required=True)
    p.add_argument("--benchmark-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark")
    p.add_argument("--causal-adapter-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter")
    p.add_argument("--causal-adapter-scm", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _load_indices(path: str, n: int) -> List[int]:
    with open(path) as f:
        raw = json.load(f)
    idxs = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
    return idxs[:n]


class TokenOnlyEditor:
    def __init__(
        self,
        cfg: Dict[str, Any],
        mapper_dir: Path,
        device: torch.device,
        steps: int,
        guidance_scale: float,
        lora_dir: Path | None = None,
        editing_mode: str = "standard",
        p2p_cross_replace_steps: float = 0.2,
        p2p_self_replace_steps: float = 0.2,
        p2p_late_target_scale: float = 0.35,
        p2p_source_latent_blend: float = 0.25,
    ):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        self.cfg = cfg
        self.device = device
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.editing_mode = str(editing_mode)
        self.p2p_cross_replace_steps = float(p2p_cross_replace_steps)
        self.p2p_self_replace_steps = float(p2p_self_replace_steps)
        self.p2p_late_target_scale = float(p2p_late_target_scale)
        self.p2p_source_latent_blend = float(p2p_source_latent_blend)
        self.prompt = str(cfg.get("natural_prompt", "a human face, realistic portrait photo"))
        dtype = torch.bfloat16 if str(cfg.get("mixed_precision", "bf16")).lower() == "bf16" else torch.float16
        self.pipe = StableDiffusionPipeline.from_pretrained(str(cfg["pretrained_model_name_or_path"]), torch_dtype=dtype).to(device)
        if lora_dir is not None:
            self.pipe.load_lora_weights(str(lora_dir))
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()
        ckpt = torch.load(mapper_dir / "causal_token_mapper.pt", map_location=device)
        meta = ckpt.get("mapper", {})
        self.attr_names = list(meta.get("attr_names") or cfg["celeba_attr_cols"])
        self.mapper = CausalTokenMapper(
            self.attr_names,
            int(meta.get("hidden_dim", self.pipe.text_encoder.config.hidden_size)),
            max_token_norm=float(meta.get("max_token_norm", cfg.get("max_token_norm", 0.0))),
        ).to(device)
        self.mapper.load_state_dict(ckpt["model_state_dict"])
        self.mapper.eval()
        self.scheduler = DDIMScheduler.from_pretrained(str(cfg["pretrained_model_name_or_path"]), subfolder="scheduler")
        self.scheduler.set_timesteps(self.steps, device=device)
        self.neg_prompt = ""
        self.image_tfm = transforms.Compose(
            [
                transforms.CenterCrop(150),
                transforms.Resize((int(cfg.get("image_size", 256)), int(cfg.get("image_size", 256))), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @torch.no_grad()
    def _text_embeds(self, n: int, prompt: str) -> torch.Tensor:
        tok = self.pipe.tokenizer(
            [prompt] * n,
            padding="max_length",
            truncation=True,
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.pipe.text_encoder(tok.input_ids.to(self.device))[0]

    @torch.no_grad()
    def _condition(self, labels_pm1: torch.Tensor) -> torch.Tensor:
        prompt = self._text_embeds(labels_pm1.shape[0], self.prompt)
        causal = self.mapper(labels_pm1.float()).to(device=self.device, dtype=prompt.dtype)
        return torch.cat([prompt, causal], dim=1)

    @torch.no_grad()
    def _negative_condition(self, n: int) -> torch.Tensor:
        prompt = self._text_embeds(n, self.neg_prompt)
        zeros = torch.zeros(n, len(self.attr_names), prompt.shape[-1], device=self.device, dtype=prompt.dtype)
        return torch.cat([prompt, zeros], dim=1)

    def _noise_pred(self, latents, t, cond, uncond):
        if self.guidance_scale == 1.0:
            return self.pipe.unet(latents, t, encoder_hidden_states=cond).sample
        x = torch.cat([latents, latents], dim=0)
        ee = torch.cat([uncond, cond], dim=0)
        pred = self.pipe.unet(x, t, encoder_hidden_states=ee).sample
        u, c = pred.chunk(2)
        return u + self.guidance_scale * (c - u)

    @torch.no_grad()
    def encode_images(self, data_root: Path, filenames) -> torch.Tensor:
        xs = [self.image_tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB")) for fn in filenames]
        x = torch.stack(xs).to(self.device, dtype=self.pipe.vae.dtype)
        return self.pipe.vae.encode(x).latent_dist.sample() * self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def invert(self, latents, cond, uncond):
        sample = latents
        timesteps = list(self.scheduler.timesteps)
        for i, t in enumerate(reversed(timesteps)):
            next_t = timesteps[len(timesteps) - i - 2] if i < len(timesteps) - 1 else timesteps[0]
            a_t = self.scheduler.alphas_cumprod[t].to(sample.device, sample.dtype)
            a_next = self.scheduler.alphas_cumprod[next_t].to(sample.device, sample.dtype)
            eps = self._noise_pred(sample, t, cond, uncond)
            pred_x0 = (sample - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            sample = a_next.sqrt() * pred_x0 + (1 - a_next).sqrt() * eps
        return sample

    @torch.no_grad()
    def sample(self, latents_t, cond, uncond):
        sample = latents_t
        for t in self.scheduler.timesteps:
            eps = self._noise_pred(sample, t, cond, uncond)
            sample = self.scheduler.step(eps, t, sample).prev_sample
        return sample

    @torch.no_grad()
    def sample_with_trajectory(self, latents_t, cond, uncond):
        sample = latents_t
        trajectory = []
        for t in self.scheduler.timesteps:
            eps = self._noise_pred(sample, t, cond, uncond)
            sample = self.scheduler.step(eps, t, sample).prev_sample
            trajectory.append(sample.detach())
        return sample, trajectory

    def _target_mix(self, step_idx: int) -> float:
        if self.steps <= 1:
            return 1.0
        frac = step_idx / float(self.steps - 1)
        cutoff = max(0.0, min(1.0, self.p2p_cross_replace_steps))
        late = max(0.0, min(1.0, self.p2p_late_target_scale))
        if frac <= cutoff or cutoff >= 1.0:
            return 1.0
        progress = (frac - cutoff) / max(1.0 - cutoff, 1e-6)
        return 1.0 + (late - 1.0) * progress

    def _source_latent_mix(self, step_idx: int) -> float:
        if self.steps <= 1:
            return 0.0
        frac = step_idx / float(self.steps - 1)
        cutoff = max(0.0, min(1.0, self.p2p_self_replace_steps))
        if frac > cutoff:
            return 0.0
        return max(0.0, min(1.0, self.p2p_source_latent_blend))

    @torch.no_grad()
    def sample_p2p_like(self, latents_t, cond_orig, cond_cf, uncond):
        _, source_trajectory = self.sample_with_trajectory(latents_t, cond_orig, uncond)
        sample = latents_t
        for step_idx, t in enumerate(self.scheduler.timesteps):
            eps_orig = self._noise_pred(sample, t, cond_orig, uncond)
            eps_cf = self._noise_pred(sample, t, cond_cf, uncond)
            target_mix = self._target_mix(step_idx)
            eps = eps_orig + target_mix * (eps_cf - eps_orig)
            sample = self.scheduler.step(eps, t, sample).prev_sample
            source_mix = self._source_latent_mix(step_idx)
            if source_mix > 0.0:
                sample = (1.0 - source_mix) * sample + source_mix * source_trajectory[step_idx].to(sample)
        return sample

    @torch.no_grad()
    def decode64(self, latents):
        img = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return F.interpolate(img.float(), size=(64, 64), mode="bicubic", align_corners=False).clamp(0, 1)

    @torch.no_grad()
    def edit(self, data_root: Path, filenames, c_orig_pm1, c_cf_pm1):
        lat0 = self.encode_images(data_root, filenames)
        cond_orig = self._condition(c_orig_pm1)
        cond_cf = self._condition(c_cf_pm1)
        uncond = self._negative_condition(c_orig_pm1.shape[0])
        lat_t = self.invert(lat0, cond_orig, uncond)
        if self.editing_mode == "p2p_like":
            lat_cf = self.sample_p2p_like(lat_t, cond_orig, cond_cf, uncond)
        else:
            lat_cf = self.sample(lat_t, cond_cf, uncond)
        return self.decode64(lat_cf)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else REPO / cfg_path
    cfg = _load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(cfg["data_root"])
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
    lora_dir = Path(args.lora_dir) if args.lora_dir else None
    editor = TokenOnlyEditor(
        cfg,
        Path(args.mapper_dir),
        device,
        args.steps,
        args.guidance_scale,
        lora_dir=lora_dir,
        editing_mode=args.editing_mode,
        p2p_cross_replace_steps=args.p2p_cross_replace_steps,
        p2p_self_replace_steps=args.p2p_self_replace_steps,
        p2p_late_target_scale=args.p2p_late_target_scale,
        p2p_source_latent_blend=args.p2p_source_latent_blend,
    )

    preds = {a: [] for a in COMPLEX_ATTRS}
    tgts = {a: [] for a in COMPLEX_ATTRS}
    real_batches = []
    fake_batches = []
    mse_vals = []
    for batch in tqdm(loader, desc=f"token-only eval do({args.intervention_attr})"):
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
        mse_vals.extend(((real - fake) ** 2).mean(dim=(1, 2, 3)).detach().cpu().numpy().tolist())

    result = {
        "args": vars(args),
        "num_samples": len(indices),
        "indices": indices,
        "eval_protocol": {
            "preprocess": "CenterCrop(150)->Resize(256), eval resize 64",
            "target": f"Causal-Adapter controlnet_cond_embedding.inference do({args.intervention_attr} flip)",
            "condition": "DDIM inversion with factual prompt+causal tokens, sampling with SCM counterfactual prompt+causal tokens",
            "editing_mode": args.editing_mode,
            "p2p_like": {
                "description": "source/target denoising-path blend plus early source latent trajectory blend; CA-inspired but not the original CA attention controller",
                "cross_replace_steps": args.p2p_cross_replace_steps,
                "self_replace_steps": args.p2p_self_replace_steps,
                "late_target_scale": args.p2p_late_target_scale,
                "source_latent_blend": args.p2p_source_latent_blend,
            } if args.editing_mode == "p2p_like" else None,
        },
        "method": "minisd_token_lora_causal_cross_attention" if args.lora_dir else "minisd_token_only_causal_cross_attention",
        "effectiveness": {a: _binary_f1(np.concatenate(preds[a], 0), np.concatenate(tgts[a], 0)) for a in COMPLEX_ATTRS},
        "effectiveness_debug": {
            a: {
                "target_pos_rate": _rate(np.concatenate(tgts[a], 0), False),
                "pred_pos_rate": _rate(np.concatenate(preds[a], 0), True),
            }
            for a in COMPLEX_ATTRS
        },
        "fid": _fid(real_batches, fake_batches, device),
        "pixel_mse_minimality": {"mean": float(np.mean(mse_vals)), "std": float(np.std(mse_vals))},
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
