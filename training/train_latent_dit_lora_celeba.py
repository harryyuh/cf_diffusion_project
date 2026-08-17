"""Fine-tune a Stable Diffusion 3 / MMDiT transformer on CelebA with gender-only LoRA.

This pipeline is meant to be comparable to the repo's parent-only diffusion model:
the only semantic condition used in captions is Male -> man / woman. All downloads,
logs, and checkpoints should live on scratch via the YAML/Slurm environment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.celeba_dataset import CelebADataset, CELEBA_ATTR_ORDER, expand_env_in_cfg
from utils.logger import get_logger
from utils.seed import set_seed


def _require_sd3_deps() -> None:
    missing: List[str] = []
    for name in ("diffusers", "peft", "transformers", "accelerate"):
        try:
            __import__(name)
        except Exception as exc:
            missing.append(f"{name} ({exc})")
    if missing:
        raise RuntimeError(
            "SD3 DiT LoRA fine-tuning requires diffusers/peft/transformers/accelerate. "
            f"Missing or broken imports: {', '.join(missing)}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SD3/MMDiT LoRA on CelebA with gender-only captions.")
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def _batch_values(batch: Dict[str, Any], key: str) -> torch.Tensor:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.float().view(-1)
    return torch.as_tensor(value, dtype=torch.float32).view(-1)


ATTR_PHRASES: Dict[str, tuple[str, str]] = {
    "Young": ("young", "older"),
    "Eyeglasses": ("wearing eyeglasses", "no eyeglasses"),
    "Smiling": ("smiling", "not smiling"),
    "Mouth_Slightly_Open": ("mouth slightly open", "mouth closed"),
    "Black_Hair": ("black hair", "not black hair"),
    "Blond_Hair": ("blond hair", "not blond hair"),
    "Brown_Hair": ("brown hair", "not brown hair"),
    "Gray_Hair": ("gray hair", "not gray hair"),
    "Bangs": ("bangs", "no bangs"),
    "Wavy_Hair": ("wavy hair", "not wavy hair"),
    "Straight_Hair": ("straight hair", "not straight hair"),
    "Wearing_Hat": ("wearing a hat", "no hat"),
    'No_Beard': ('no beard', 'beard'),
    'Mustache': ('mustache', 'no mustache'),
    'Goatee': ('goatee', 'no goatee'),
    'Sideburns': ('sideburns', 'no sideburns'),
    'Heavy_Makeup': ('heavy makeup', 'no makeup'),
    'Wearing_Lipstick': ('wearing lipstick', 'no lipstick'),
}


def _phrase_for_attr(key: str, value: float) -> str:
    if key == "Male":
        return "man" if float(value) > 0 else "woman"
    label = key.replace("_", " ").lower()
    pos, neg = ATTR_PHRASES.get(key, (label, f"not {label}"))
    return pos if float(value) > 0 else neg


def build_gender_captions(batch: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    prefix = str(cfg.get("caption_prefix", "a portrait photo of a"))
    gender_key = str(cfg.get("gender_key", "Male"))
    attr_cols = list(cfg.get("caption_attr_cols") or cfg.get("prompt_attr_cols") or [gender_key])
    if gender_key not in attr_cols:
        attr_cols = [gender_key, *attr_cols]
    batch_size = int(_batch_values(batch, gender_key).numel())
    captions: List[str] = []
    for i in range(batch_size):
        phrases = [_phrase_for_attr(key, float(_batch_values(batch, key)[i])) for key in attr_cols]
        captions.append(", ".join([prefix, *phrases]))
    return captions


def _load_pipeline(cfg: Dict[str, Any], dtype: torch.dtype):
    from diffusers import StableDiffusion3Pipeline

    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    return StableDiffusion3Pipeline.from_pretrained(str(cfg["pretrained_model_name_or_path"]), **kwargs)


def _encode_prompts(pipe, prompts: List[str], device: torch.device, dtype: torch.dtype, max_sequence_length: int):
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        max_sequence_length=max_sequence_length,
    )
    del negative_prompt_embeds, negative_pooled_prompt_embeds
    return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)


def _trainable_parameters(module: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad)


def _save_lora(pipe_cls, transformer, save_dir: Path, metadata: Dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict

    save_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(transformer)
    pipe_cls.save_lora_weights(str(save_dir), transformer_lora_layers=state)
    torch.save({"transformer_lora_state_dict": state, "metadata": metadata}, save_dir / "training_state.pt")
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _sigma_for_timesteps(noise_scheduler, timesteps: torch.Tensor, ndim: int, dtype: torch.dtype) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device=timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def main() -> None:
    args = parse_args()
    _require_sd3_deps()
    from diffusers import StableDiffusion3Pipeline
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    precision = str(cfg.get("mixed_precision", "bf16")).lower()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision in ("fp16", "float16") else torch.float32
    image_size = int(cfg.get("image_size", 512))
    max_sequence_length = int(cfg.get("max_sequence_length", 256))

    output_dir = Path(cfg["output_dir"]) / str(cfg.get("run_name", "sd3_gender_lora_celeba"))
    log_dir = output_dir / "logs"
    lora_dir = output_dir / "lora"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train_latent_dit_lora", log_dir=log_dir)
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    pipe = _load_pipeline(cfg, dtype=dtype)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    for enc_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, enc_name, None)
        if enc is not None:
            enc.requires_grad_(False)

    rank = int(cfg.get("lora_rank", 16))
    alpha = int(cfg.get("lora_alpha", rank))
    target_modules = list(cfg.get("lora_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]))
    pipe.transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    if bool(cfg.get("gradient_checkpointing", True)):
        pipe.transformer.enable_gradient_checkpointing()

    pipe.to(device)
    pipe.vae.eval()
    pipe.transformer.train()
    for enc_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, enc_name, None)
        if enc is not None:
            enc.eval()

    use_caption_attrs = bool(cfg.get("caption_attr_cols") or cfg.get("prompt_attr_cols"))
    if not use_caption_attrs:
        with torch.no_grad():
            man_prompt = f"{cfg.get('caption_prefix', 'a portrait photo of a')} man"
            woman_prompt = f"{cfg.get('caption_prefix', 'a portrait photo of a')} woman"
            man_embeds = _encode_prompts(pipe, [man_prompt], device, dtype, max_sequence_length)
            woman_embeds = _encode_prompts(pipe, [woman_prompt], device, dtype, max_sequence_length)
        for enc_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            enc = getattr(pipe, enc_name, None)
            if enc is not None:
                enc.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if cfg.get("use_all_celeba_attrs"):
        factor_cols = list(CELEBA_ATTR_ORDER)
    else:
        factor_cols = list(cfg.get("celeba_attr_cols", [cfg.get("gender_key", "Male")]))
    gender_key = str(cfg.get("gender_key", "Male"))
    if gender_key not in factor_cols:
        factor_cols.append(gender_key)

    dataset = CelebADataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        factor_cols=factor_cols,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        _trainable_parameters(pipe.transformer),
        lr=float(cfg.get("lr", 1e-4)),
        betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    max_train_steps = int(cfg.get("max_train_steps", 1000))
    steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
    epochs = int(math.ceil(max_train_steps / steps_per_epoch))
    lr_scheduler = get_scheduler(
        str(cfg.get("lr_scheduler", "constant")),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.get("lr_warmup_steps", 0)),
        num_training_steps=max_train_steps,
    )

    noise_scheduler = pipe.scheduler
    noise_scheduler.set_timesteps(int(cfg.get("num_train_timesteps", 1000)), device=device)
    pipe.vae.to(device=device, dtype=dtype)
    pipe.transformer.to(device=device, dtype=dtype)
    for enc_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, enc_name, None)
        if enc is not None:
            enc.to(device=device, dtype=dtype)

    global_step = 0
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    logger.info(f"Training SD3/MMDiT conditional LoRA for {max_train_steps} optimizer steps; output={output_dir}")

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["image"].to(device=device, dtype=dtype) * 2.0 - 1.0
            male_values = _batch_values(batch, gender_key).to(device)

            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                indices = torch.randint(0, len(noise_scheduler.timesteps), (bsz,), device=device)
                timesteps = noise_scheduler.timesteps[indices].to(device=device)
                sigmas = _sigma_for_timesteps(noise_scheduler, timesteps, latents.ndim, dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                if use_caption_attrs:
                    captions = build_gender_captions(batch, cfg)
                    prompt_embeds, pooled_prompt_embeds = _encode_prompts(pipe, captions, device, dtype, max_sequence_length)
                else:
                    prompt_parts = [man_embeds if float(v) > 0 else woman_embeds for v in male_values]
                    prompt_embeds = torch.cat([parts[0] for parts in prompt_parts], dim=0)
                    pooled_prompt_embeds = torch.cat([parts[1] for parts in prompt_parts], dim=0)

            model_pred = pipe.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            target = noise - latents
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") / grad_accum
            loss.backward()
            running += float(loss.detach().cpu())

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(list(_trainable_parameters(pipe.transformer)), float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix({"step": global_step, "loss": f"{running:.4f}"})
                if global_step % int(cfg.get("log_every", 25)) == 0:
                    logger.info(f"step={global_step} loss={running:.6f} lr={lr_scheduler.get_last_lr()[0]:.3e}")
                running = 0.0
                if global_step % int(cfg.get("save_every", 500)) == 0:
                    _save_lora(
                        StableDiffusion3Pipeline,
                        pipe.transformer,
                        output_dir / f"lora_step_{global_step:06d}",
                        {"global_step": global_step, "config": cfg},
                    )
                if global_step >= max_train_steps:
                    break
        if global_step >= max_train_steps:
            break

    _save_lora(StableDiffusion3Pipeline, pipe.transformer, lora_dir, {"global_step": global_step, "config": cfg})
    logger.info(f"Finished SD3/MMDiT conditional LoRA at step={global_step}; saved {lora_dir}")


if __name__ == "__main__":
    main()
