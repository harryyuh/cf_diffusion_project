"""Fine-tune a pretrained latent diffusion model on CelebA with UNet LoRA.

This is intentionally separate from ``training.train_diffusion``: the existing code
trains a pixel-space epsilon model for causal counterfactual experiments, while this
script adapts a large pretrained text-to-image latent model for high visual quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.celeba_dataset import CelebADataset, CELEBA_ATTR_ORDER, expand_env_in_cfg
from utils.logger import get_logger
from utils.seed import set_seed


def _require_diffusers() -> None:
    missing: List[str] = []
    for name in ("diffusers", "peft", "transformers", "accelerate"):
        try:
            __import__(name)
        except Exception as exc:  # pragma: no cover - environment dependent
            missing.append(f"{name} ({exc})")
    if missing:
        raise RuntimeError(
            "Latent LoRA fine-tuning requires extra packages. Install/update them first, e.g.\n"
            "  pip install -U 'diffusers>=0.35.0' 'peft>=0.13.0' 'transformers>=4.44.0' accelerate safetensors\n"
            f"Missing or broken imports: {', '.join(missing)}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SDXL/SD-style latent diffusion on CelebA with LoRA.")
    p.add_argument("--config", type=str, required=True, help="Path to latent LoRA YAML config.")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


_ATTR_PHRASES = {
    "Young": ("young", "mature"),
    "Smiling": ("smiling", "serious expression"),
    "Attractive": ("attractive", "natural-looking"),
    "Heavy_Makeup": ("wearing heavy makeup", "minimal makeup"),
    "Wearing_Lipstick": ("wearing lipstick", "no lipstick"),
    "Mouth_Slightly_Open": ("mouth slightly open", "closed mouth"),
    "Eyeglasses": ("wearing eyeglasses", "without eyeglasses"),
    "Black_Hair": ("black hair", None),
    "Blond_Hair": ("blond hair", None),
    "Brown_Hair": ("brown hair", None),
    "Gray_Hair": ("gray hair", None),
    "Bald": ("bald", None),
    "Bangs": ("bangs", None),
    "Wavy_Hair": ("wavy hair", None),
    "Straight_Hair": ("straight hair", None),
    "Wearing_Hat": ("wearing a hat", None),
    "Wearing_Earrings": ("wearing earrings", None),
    "No_Beard": ("clean-shaven", "beard"),
    "Mustache": ("mustache", None),
    "Goatee": ("goatee", None),
}


def _batch_values(batch: Dict[str, Any], key: str) -> torch.Tensor:
    v = batch[key]
    if isinstance(v, torch.Tensor):
        return v.float().view(-1)
    return torch.as_tensor(v, dtype=torch.float32).view(-1)


class CausalLabelEmbedding(nn.Module):
    """Map SCM/full-label conditions to the UNet global time-conditioning space."""

    def __init__(self, label_dim: int, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        first = self.net[0]
        labels = labels.to(device=first.weight.device, dtype=first.weight.dtype)
        return self.net(labels)


def build_label_condition(batch: Dict[str, Any], label_cols: Sequence[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [_batch_values(batch, key) for key in label_cols]
    labels = torch.stack(values, dim=1).to(device=device, dtype=dtype)
    return labels


def build_balanced_sampler(dataset: CelebADataset, attr_cols: Sequence[str], cfg: Dict[str, Any]) -> WeightedRandomSampler:
    """Oversample rare binary attribute states for prompt-conditioning LoRA."""
    if not attr_cols:
        raise ValueError("balanced_attr_cols must contain at least one attribute.")
    factors = dataset.factor_matrix(attr_cols)
    weights = torch.zeros(factors.shape[0], dtype=torch.double)
    for j, attr in enumerate(attr_cols):
        values = factors[:, j]
        unique, counts = torch.unique(torch.as_tensor(values, dtype=torch.float32), return_counts=True)
        if unique.numel() < 2:
            raise ValueError(f"Cannot balance {attr}: only one value appears in this split.")
        count_by_value = {float(v.item()): float(c.item()) for v, c in zip(unique, counts)}
        attr_weights = torch.as_tensor([1.0 / count_by_value[float(v)] for v in values], dtype=torch.double)
        attr_weights = attr_weights / attr_weights.mean()
        weights += attr_weights
    weights /= float(len(attr_cols))
    power = float(cfg.get("balanced_sampler_power", 1.0))
    if power != 1.0:
        weights = weights.pow(power)
    weights = weights / weights.mean()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_captions(batch: Dict[str, Any], attr_cols: Sequence[str], cfg: Dict[str, Any]) -> List[str]:
    """Convert CelebA attributes into simple portrait captions."""
    prefix = str(cfg.get("caption_prefix", "a high quality portrait photo"))
    suffix = str(cfg.get("caption_suffix", "sharp focus, realistic skin texture"))
    include_negative = bool(cfg.get("caption_include_negative_attrs", False))
    max_attrs = int(cfg.get("caption_max_attrs", 8))

    n = int(batch["image"].shape[0])
    captions: List[str] = []
    male = _batch_values(batch, "Male") if "Male" in batch else torch.zeros(n)
    for i in range(n):
        subject = "man" if float(male[i]) > 0 else "woman"
        phrases: List[str] = [f"{prefix} of a {subject}"]
        used = 0
        for key in attr_cols:
            if key == "Male" or key not in batch or key not in _ATTR_PHRASES:
                continue
            pos, neg = _ATTR_PHRASES[key]
            value = float(_batch_values(batch, key)[i])
            phrase = pos if value > 0 else (neg if include_negative else None)
            if phrase:
                phrases.append(phrase)
                used += 1
            if used >= max_attrs:
                break
        if suffix:
            phrases.append(suffix)
        captions.append(", ".join(phrases))
    return captions


def build_full_label_prompts(batch: Dict[str, Any], label_cols: Sequence[str], cfg: Dict[str, Any]) -> List[str]:
    """Build explicit text-semantic prompts from binary labels."""
    prefix = str(cfg.get("full_label_prompt_prefix", "a human face"))
    suffix = str(cfg.get("full_label_prompt_suffix", ""))
    phrase_map = {
        "Young": ("young", "older"),
        "Male": ("male", "female"),
        "No_Beard": ("no beard", "beard"),
        "Bald": ("bald", "with hair"),
    }
    n = int(batch["image"].shape[0])
    prompts: List[str] = []
    for i in range(n):
        phrases: List[str] = [prefix]
        for key in label_cols:
            if key not in batch or key not in phrase_map:
                continue
            pos, neg = phrase_map[key]
            phrases.append(pos if float(_batch_values(batch, key)[i]) > 0 else neg)
        if suffix:
            phrases.append(suffix)
        prompts.append(", ".join([x for x in phrases if x]))
    return prompts


def _pipeline_class(model_family: str):
    family = model_family.lower()
    if family == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        return StableDiffusionXLPipeline
    if family in ("sd", "sd15", "sd2"):
        from diffusers import StableDiffusionPipeline

        return StableDiffusionPipeline
    raise ValueError("model_family must be 'sdxl' or 'sd'")


def _load_pipeline(cfg: Dict[str, Any], dtype: torch.dtype):
    pipe_cls = _pipeline_class(str(cfg.get("model_family", "sdxl")))
    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    return pipe_cls.from_pretrained(str(cfg["pretrained_model_name_or_path"]), **kwargs)


def _encode_sdxl_prompts(pipe, prompts: List[str], device: torch.device, dtype: torch.dtype):
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)


def _sdxl_time_ids(pipe, batch_size: int, image_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    original_size = (image_size, image_size)
    target_size = (image_size, image_size)
    crop = (0, 0)
    projection_dim = int(getattr(pipe.text_encoder_2.config, "projection_dim", 1280))
    try:
        add_time_ids = pipe._get_add_time_ids(  # diffusers SDXL helper; signature changed across releases.
            original_size,
            crop,
            target_size,
            dtype=dtype,
            text_encoder_projection_dim=projection_dim,
        )
    except TypeError:
        add_time_ids = pipe._get_add_time_ids(original_size, crop, target_size, dtype=dtype)
    return add_time_ids.to(device=device, dtype=dtype).repeat(batch_size, 1)


def _encode_sd_prompts(pipe, prompts: List[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    return pipe.text_encoder(input_ids)[0].to(dtype=dtype)


def _unet_time_embed_dim(unet: torch.nn.Module) -> int:
    dim = getattr(getattr(unet, "time_embedding", None), "linear_2", None)
    if dim is not None and hasattr(dim, "out_features"):
        return int(dim.out_features)
    cfg = getattr(unet, "config", None)
    if cfg is not None and getattr(cfg, "time_embedding_dim", None):
        return int(cfg.time_embedding_dim)
    block_channels = getattr(cfg, "block_out_channels", None) if cfg is not None else None
    if block_channels:
        return int(block_channels[0]) * 4
    raise ValueError("Cannot infer UNet time embedding dimension for label conditioning.")


def _save_lora(pipe_cls, unet, save_dir: Path, metadata: Dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict

    save_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(unet)
    pipe_cls.save_lora_weights(str(save_dir), unet_lora_layers=state)
    extra_state: Dict[str, Any] = {}
    if getattr(unet, "class_embedding", None) is not None:
        extra_state["class_embedding_state_dict"] = unet.class_embedding.state_dict()
    torch.save({"unet_lora_state_dict": state, **extra_state, "metadata": metadata}, save_dir / "training_state.pt")
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _trainable_parameters(module: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad)


def main() -> None:
    args = parse_args()
    _require_diffusers()
    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    precision = str(cfg.get("mixed_precision", "bf16")).lower()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision in ("fp16", "float16") else torch.float32
    model_family = str(cfg.get("model_family", "sdxl")).lower()
    image_size = int(cfg.get("image_size", 512))
    use_label_conditioning = bool(cfg.get("use_label_conditioning", False))

    output_dir = Path(cfg["output_dir"]) / str(cfg.get("run_name", "sdxl_lora_celeba"))
    log_dir = output_dir / "logs"
    lora_dir = output_dir / "lora"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train_latent_lora", log_dir=log_dir)
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    pipe = _load_pipeline(cfg, dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(str(cfg["pretrained_model_name_or_path"]), subfolder="scheduler")
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)

    rank = int(cfg.get("lora_rank", 16))
    alpha = int(cfg.get("lora_alpha", rank))
    target_modules = list(cfg.get("lora_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]))
    pipe.unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    label_cols = list(cfg.get("label_condition_cols") or cfg.get("celeba_attr_cols") or [])
    if use_label_conditioning:
        if not label_cols:
            raise ValueError("use_label_conditioning=true requires label_condition_cols or celeba_attr_cols.")
        embed_dim = _unet_time_embed_dim(pipe.unet)
        hidden_dim = int(cfg.get("label_condition_hidden_dim", 256))
        pipe.unet.class_embedding = CausalLabelEmbedding(len(label_cols), embed_dim, hidden_dim)
    pipe.to(device)
    pipe.unet.train()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.eval()

    if bool(cfg.get("gradient_checkpointing", True)):
        pipe.unet.enable_gradient_checkpointing()

    attr_cols = list(CELEBA_ATTR_ORDER) if cfg.get("use_all_celeba_attrs") else list(cfg["celeba_attr_cols"])
    for key in label_cols:
        if key not in attr_cols:
            attr_cols.append(key)
    dataset = CelebADataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        factor_cols=attr_cols,
        image_size=image_size,
        center_crop_size=int(cfg.get("center_crop_size", 0) or 0),
    )
    balanced_attr_cols = list(cfg.get("balanced_attr_cols") or [])
    sampler = build_balanced_sampler(dataset, balanced_attr_cols, cfg) if balanced_attr_cols else None
    if balanced_attr_cols:
        logger.info(f"Using balanced sampler over attributes: {balanced_attr_cols}")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        _trainable_parameters(pipe.unet),
        lr=float(cfg.get("lr", 1e-4)),
        betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    max_train_steps = int(cfg.get("max_train_steps", 10000))
    steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
    epochs = int(math.ceil(max_train_steps / steps_per_epoch))
    scheduler = get_scheduler(
        str(cfg.get("lr_scheduler", "constant")),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.get("lr_warmup_steps", 0)),
        num_training_steps=max_train_steps,
    )

    pipe.vae.to(device=device, dtype=dtype)
    pipe.unet.to(device=device, dtype=dtype)
    pipe.text_encoder.to(device=device, dtype=dtype)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device=device, dtype=dtype)

    global_step = 0
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    logger.info(f"Training {model_family} LoRA for up to {max_train_steps} optimizer steps; output={output_dir}")

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["image"].to(device=device, dtype=dtype) * 2.0 - 1.0
            if bool(cfg.get("prompt_from_full_labels", False)):
                prompt_cols = list(cfg.get("prompt_label_cols") or label_cols or attr_cols)
                captions = build_full_label_prompts(batch, prompt_cols, cfg)
            elif use_label_conditioning and bool(cfg.get("fixed_prompt_for_label_conditioning", True)):
                captions = [str(cfg.get("fixed_prompt", "a human face"))] * int(pixel_values.shape[0])
            else:
                captions = build_captions(batch, attr_cols, cfg)
            label_condition = None
            if use_label_conditioning:
                label_condition = build_label_condition(batch, label_cols, device, dtype)

            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if model_family == "sdxl":
                    prompt_embeds, pooled_prompt_embeds = _encode_sdxl_prompts(pipe, captions, device, dtype)
                    added_cond_kwargs = {
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": _sdxl_time_ids(pipe, bsz, image_size, device, dtype),
                    }
                else:
                    prompt_embeds = _encode_sd_prompts(pipe, captions, device, dtype)
                    added_cond_kwargs = None

            if model_family == "sdxl":
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    class_labels=label_condition,
                ).sample
            else:
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=label_condition,
                ).sample

            target = noise
            if getattr(noise_scheduler.config, "prediction_type", None) == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") / grad_accum
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Non-finite training loss at global_step={global_step}, batch_idx={batch_idx}: {float(loss.detach().cpu())}")
            loss.backward()
            running += float(loss.detach().cpu())

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(list(_trainable_parameters(pipe.unet)), float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix({"step": global_step, "loss": f"{running:.4f}"})
                if global_step % int(cfg.get("log_every", 25)) == 0:
                    logger.info(f"step={global_step} loss={running:.6f} lr={scheduler.get_last_lr()[0]:.3e}")
                running = 0.0

                if global_step % int(cfg.get("save_every", 1000)) == 0:
                    _save_lora(
                        _pipeline_class(model_family),
                        pipe.unet,
                        output_dir / f"lora_step_{global_step:06d}",
                        {"global_step": global_step, "config": cfg},
                    )
                if global_step >= max_train_steps:
                    break
        if global_step >= max_train_steps:
            break

    _save_lora(_pipeline_class(model_family), pipe.unet, lora_dir, {"global_step": global_step, "config": cfg})
    logger.info(f"Finished latent LoRA fine-tuning at step={global_step}; saved {lora_dir}")


if __name__ == "__main__":
    main()
