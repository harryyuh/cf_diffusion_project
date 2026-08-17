"""Train causal-token-conditioned UNet LoRA on CelebA."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset, expand_env_in_cfg
from models.causal_token_mapper import CausalTokenMapper
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def _labels_pm1(batch: Dict[str, Any], attr_cols: Sequence[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = [batch[k].to(device=device, dtype=dtype).view(-1) for k in attr_cols]
    return torch.stack(vals, dim=1)


def build_balanced_sampler(dataset: CelebADataset, attr_cols: Sequence[str], cfg: Dict[str, Any]) -> WeightedRandomSampler:
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


class TokenOnlyConditioner:
    def __init__(self, pipe, mapper: CausalTokenMapper, prompt: str, device: torch.device, dtype: torch.dtype):
        self.pipe = pipe
        self.mapper = mapper
        self.prompt = prompt
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def prompt_embeds(self, batch_size: int) -> torch.Tensor:
        tok = self.pipe.tokenizer(
            [self.prompt] * batch_size,
            padding="max_length",
            truncation=True,
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.pipe.text_encoder(tok.input_ids.to(self.device))[0]

    def encoder_hidden_states(self, labels_pm1: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt_embeds(labels_pm1.shape[0]).to(dtype=self.dtype)
        causal = self.causal_tokens(labels_pm1)
        return torch.cat([prompt, causal], dim=1)

    def causal_tokens(self, labels_pm1: torch.Tensor) -> torch.Tensor:
        return self.mapper(labels_pm1.float()).to(device=self.device, dtype=self.dtype)


def save_checkpoint(mapper: CausalTokenMapper, output_dir: Path, global_step: int, cfg: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "model_state_dict": mapper.state_dict(),
        "mapper": mapper.metadata(),
        "config": cfg,
    }
    torch.save(payload, output_dir / "causal_token_mapper.pt")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({k: v for k, v in payload.items() if k != "model_state_dict"}, f, indent=2)


def _trainable_parameters(module: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad)


def causal_token_contrastive_loss(
    tokens: torch.Tensor,
    labels_pm1: torch.Tensor,
    temperature: float = 0.2,
    mode: str = "attr_value",
) -> torch.Tensor:
    """CA-style InfoNCE over learned causal tokens.

    CA applies PromptCL to pseudo-token embeddings. Our tokens are label-conditioned,
    so the default treats tokens with the same attribute and same binary value as
    positives, and all other causal tokens in the batch as negatives.
    """
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B,K,D], got {tuple(tokens.shape)}")
    batch, num_attrs, hidden = tokens.shape
    feats = F.normalize(tokens.float().reshape(batch * num_attrs, hidden), dim=-1)
    sim = feats @ feats.T
    sim = sim / max(float(temperature), 1e-8)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -9e15)

    attr_ids = torch.arange(num_attrs, device=tokens.device).view(1, num_attrs).expand(batch, num_attrs).reshape(-1)
    labels = labels_pm1.float().reshape(-1)
    if mode == "attr":
        positive = attr_ids[:, None].eq(attr_ids[None, :])
    elif mode == "attr_value":
        positive = attr_ids[:, None].eq(attr_ids[None, :]) & labels[:, None].eq(labels[None, :])
    else:
        raise ValueError(f"Unknown ctc_mode={mode!r}; expected 'attr' or 'attr_value'.")
    positive = positive & ~eye
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return tokens.new_zeros(())
    log_den = torch.logsumexp(sim[valid], dim=-1)
    pos_log_num = torch.logsumexp(sim[valid].masked_fill(~positive[valid], -9e15), dim=-1)
    return -(pos_log_num - log_den).mean()


def save_hybrid_checkpoint(pipe, mapper: CausalTokenMapper, output_dir: Path, global_step: int, cfg: Dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict

    output_dir.mkdir(parents=True, exist_ok=True)
    lora_state = get_peft_model_state_dict(pipe.unet)
    pipe.save_lora_weights(str(output_dir / "lora"), unet_lora_layers=lora_state)
    save_checkpoint(mapper, output_dir / "mapper", global_step, cfg)
    torch.save(
        {
            "global_step": int(global_step),
            "unet_lora_state_dict": lora_state,
            "mapper_state_dict": mapper.state_dict(),
            "mapper": mapper.metadata(),
            "config": cfg,
        },
        output_dir / "training_state.pt",
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("train_token_only_celeba")
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if str(cfg.get("mixed_precision", "bf16")).lower() == "bf16" else torch.float16

    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig

    output_dir = Path(cfg["output_dir"]) / str(cfg["run_name"])
    ckpt_dir = output_dir / "checkpoints"
    final_dir = output_dir / "final"
    image_size = int(cfg.get("image_size", 256))
    attr_cols = list(cfg["celeba_attr_cols"])
    prompt = str(cfg.get("natural_prompt", "a human face, realistic portrait photo"))

    dataset = CelebADataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        factor_cols=attr_cols,
        image_size=image_size,
        center_crop_size=int(cfg.get("center_crop_size", 0) or 0),
    )
    balanced_attr_cols = list(cfg.get("balanced_attr_cols") or [])
    sampler = build_balanced_sampler(dataset, balanced_attr_cols, cfg) if balanced_attr_cols else None
    if sampler is not None:
        logger.info(f"Using balanced sampler over attributes: {balanced_attr_cols}")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    pipe = StableDiffusionPipeline.from_pretrained(str(cfg["pretrained_model_name_or_path"]), torch_dtype=dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(str(cfg["pretrained_model_name_or_path"]), subfolder="scheduler")
    pipe.vae.eval()
    pipe.unet.eval()
    pipe.text_encoder.eval()
    for module in (pipe.vae, pipe.unet, pipe.text_encoder):
        for p in module.parameters():
            p.requires_grad_(False)
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
    pipe.unet.train()

    with torch.no_grad():
        sample_prompt = pipe.tokenizer([prompt], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        hidden_dim = int(pipe.text_encoder(sample_prompt.input_ids.to(device))[0].shape[-1])
    mapper = CausalTokenMapper(
        attr_names=attr_cols,
        hidden_dim=hidden_dim,
        init_std=float(cfg.get("token_init_std", 0.02)),
        max_token_norm=float(cfg.get("max_token_norm", 0.0)),
    ).to(device)
    conditioner = TokenOnlyConditioner(pipe, mapper, prompt, device, dtype)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(mapper.parameters()), "lr": float(cfg.get("mapper_lr", cfg.get("lr", 1e-3)))},
            {"params": list(_trainable_parameters(pipe.unet)), "lr": float(cfg.get("lora_lr", 5e-5))},
        ],
        betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    max_train_steps = int(cfg.get("max_train_steps", 5000))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    scheduler = get_scheduler(
        str(cfg.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.get("lr_warmup_steps", 100)),
        num_training_steps=max_train_steps,
    )
    epochs = int(math.ceil(max_train_steps / max(1, math.ceil(len(loader) / grad_accum))))
    logger.info(f"Training causal token + LoRA hybrid for {max_train_steps} steps; output={output_dir}")

    global_step = 0
    lambda_ctc = float(cfg.get("lambda_ctc", cfg.get("infonce_scale", 0.0)))
    ctc_temperature = float(cfg.get("ctc_temperature", cfg.get("infonce_temperature", 0.2)))
    ctc_mode = str(cfg.get("ctc_mode", "attr_value"))
    if lambda_ctc > 0:
        logger.info(f"Using causal-token CTC: lambda={lambda_ctc} temperature={ctc_temperature} mode={ctc_mode}")

    running = 0.0
    running_mse = 0.0
    running_ctc = 0.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["image"].to(device=device, dtype=dtype) * 2.0 - 1.0
            labels = _labels_pm1(batch, attr_cols, device, dtype=torch.float32)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            causal_tokens = conditioner.causal_tokens(labels)
            prompt_states = conditioner.prompt_embeds(labels.shape[0]).to(dtype=dtype)
            encoder_hidden_states = torch.cat([prompt_states, causal_tokens], dim=1)
            pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            target = noise
            if getattr(noise_scheduler.config, "prediction_type", None) == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
            ctc_loss = causal_token_contrastive_loss(causal_tokens, labels, ctc_temperature, ctc_mode) if lambda_ctc > 0 else mse_loss.new_zeros(())
            loss = (mse_loss + lambda_ctc * ctc_loss) / grad_accum
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Non-finite loss at step={global_step}")
            loss.backward()
            running += float(loss.detach().cpu())
            running_mse += float((mse_loss.detach() / grad_accum).cpu())
            running_ctc += float((ctc_loss.detach() / grad_accum).cpu())
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(mapper.parameters(), float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix({"step": global_step, "loss": f"{running:.4f}", "mse": f"{running_mse:.4f}", "ctc": f"{running_ctc:.4f}"})
                if global_step % int(cfg.get("log_every", 25)) == 0:
                    logger.info(
                        f"step={global_step} loss={running:.6f} mse={running_mse:.6f} "
                        f"ctc={running_ctc:.6f} lr={scheduler.get_last_lr()[0]:.3e}"
                    )
                running = 0.0
                running_mse = 0.0
                running_ctc = 0.0
                if global_step % int(cfg.get("save_every", 500)) == 0:
                    save_hybrid_checkpoint(pipe, mapper, ckpt_dir / f"step_{global_step:06d}", global_step, cfg)
                if global_step >= max_train_steps:
                    break
        if global_step >= max_train_steps:
            break

    save_hybrid_checkpoint(pipe, mapper, final_dir, global_step, cfg)
    logger.info(f"Finished token-only training at step={global_step}; saved {final_dir}")


if __name__ == "__main__":
    main()
