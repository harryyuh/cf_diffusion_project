"""Train PixArt DiT LoRA on the four CelebA causal attributes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.celeba_dataset import CelebADataset
from utils.seed import set_seed

ATTRS = ("Young", "Male", "No_Beard", "Bald")


def captions(batch):
    phrases = {
        "Young": ("young", "older"),
        "Male": ("man", "woman"),
        "No_Beard": ("no beard", "beard"),
        "Bald": ("bald", "not bald"),
    }
    result = []
    for i in range(len(batch[ATTRS[0]])):
        words = [phrases[key][0 if float(batch[key][i]) > 0 else 1] for key in ATTRS]
        result.append("a realistic portrait photograph, " + ", ".join(words))
    return result


def save_lora(pipe, out: Path, step: int, cfg):
    from peft import get_peft_model_state_dict

    out.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(pipe.transformer)
    pipe.save_lora_weights(str(out), transformer_lora_layers=state)
    (out / "metadata.json").write_text(json.dumps({"global_step": step, "config": cfg}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg.get("seed", 42)))

    from diffusers import DDIMScheduler, PixArtAlphaPipeline
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    dtype = torch.bfloat16
    pipe = PixArtAlphaPipeline.from_pretrained(cfg["model"], torch_dtype=dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False, set_alpha_to_one=False)
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    # PixArtTransformer2DModel in some Diffusers releases does not inherit
    # PeftAdapterMixin, so its otherwise valid LoRA path has no add_adapter().
    # PEFT's generic wrapper is version-independent and preserves the forward
    # signature used below.
    pipe.transformer = get_peft_model(pipe.transformer, LoraConfig(
        r=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        init_lora_weights="gaussian",
        target_modules=list(cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])),
    ))
    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.train()

    dataset = CelebADataset(
        root=cfg["data_root"], split="train", factor_cols=list(ATTRS),
        image_size=int(pipe.transformer.config.sample_size) * int(pipe.vae_scale_factor),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 1)), shuffle=True,
                        num_workers=int(cfg.get("num_workers", 4)), pin_memory=True, drop_last=True)
    params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)), weight_decay=0.01)
    max_steps = int(cfg.get("max_train_steps", 20000))
    accum = int(cfg.get("gradient_accumulation_steps", 4))
    schedule = get_scheduler("cosine", optimizer, num_warmup_steps=200, num_training_steps=max_steps)
    epochs = math.ceil(max_steps / max(1, math.ceil(len(loader) / accum)))
    out = Path(cfg["output_dir"]) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(bar):
            pixels = batch["image"].to(device=device, dtype=dtype) * 2 - 1
            with torch.no_grad():
                latents = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
                encoded = pipe.encode_prompt(
                    prompt=captions(batch), do_classifier_free_guidance=False,
                    num_images_per_prompt=1, device=device, max_sequence_length=120,
                )
                prompt_embeds, prompt_mask = encoded[0], encoded[1]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=device).long()
                noisy = pipe.scheduler.add_noise(latents, noise, timesteps)
            pred = pipe.transformer(
                noisy, encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask,
                timestep=timesteps, added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                return_dict=False,
            )[0]
            if pred.shape[1] == noise.shape[1] * 2:
                pred = pred.chunk(2, dim=1)[0]
            loss = F.mse_loss(pred.float(), noise.float()) / accum
            loss.backward()
            if (batch_idx + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                bar.set_postfix(step=global_step, loss=f"{float(loss) * accum:.4f}")
                if global_step in {5000, 20000}:
                    save_lora(pipe, out / f"lora_step_{global_step:06d}", global_step, cfg)
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break
    save_lora(pipe, out / "lora", global_step, cfg)


if __name__ == "__main__":
    main()
