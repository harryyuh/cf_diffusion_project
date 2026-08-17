"""PixArt-alpha DDIM compatibility and reconstruction smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="PixArt-alpha/PixArt-XL-2-256x256")
    p.add_argument("--image", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    return p.parse_args()


def model_eps(pipe, latents, timestep, prompt_embeds, prompt_mask):
    t = timestep.expand(latents.shape[0])
    out = pipe.transformer(
        latents,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask,
        timestep=t,
        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        return_dict=False,
    )[0]
    # PixArt checkpoints may jointly predict noise and variance. DDIM needs eps.
    if out.shape[1] == latents.shape[1] * 2:
        out = out.chunk(2, dim=1)[0]
    return out


@torch.no_grad()
def invert(pipe, latent0, prompt_embeds, prompt_mask):
    scheduler = pipe.scheduler
    sample = latent0
    step_size = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    for t in reversed(scheduler.timesteps):
        next_t = int(t.item())
        current_t = min(next_t - step_size, scheduler.config.num_train_timesteps - 1)
        eps = model_eps(pipe, sample, t, prompt_embeds, prompt_mask)
        alpha_t = scheduler.alphas_cumprod[current_t].to(sample)
        alpha_next = scheduler.alphas_cumprod[next_t].to(sample)
        x0 = (sample - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        sample = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * eps
    return sample


@torch.no_grad()
def sample(pipe, latent_t, prompt_embeds, prompt_mask):
    sample = latent_t
    for t in pipe.scheduler.timesteps:
        eps = model_eps(pipe, sample, t, prompt_embeds, prompt_mask)
        sample = pipe.scheduler.step(eps, t, sample).prev_sample
    return sample


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers import DDIMScheduler, PixArtAlphaPipeline

    dtype = torch.bfloat16
    pipe = PixArtAlphaPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, clip_sample=False, set_alpha_to_one=False
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.steps, device="cuda")

    prompt = "a realistic portrait photograph of a human face"
    encoded = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=False,
        num_images_per_prompt=1,
        device=torch.device("cuda"),
        max_sequence_length=120,
    )
    prompt_embeds, prompt_mask = encoded[0], encoded[1]

    image = Image.open(args.image).convert("RGB")
    tfm = transforms.Compose([
        transforms.CenterCrop(150),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    x = tfm(image).unsqueeze(0).to("cuda", dtype=pipe.vae.dtype)
    latent0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    latent_t = invert(pipe, latent0, prompt_embeds, prompt_mask)
    reconstructed = sample(pipe, latent_t, prompt_embeds, prompt_mask)
    decoded = pipe.vae.decode(reconstructed / pipe.vae.config.scaling_factor).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    transforms.ToPILImage()(decoded[0].float().cpu()).save(out / "reconstruction.png")
    transforms.ToPILImage()((x[0].float().cpu() / 2 + 0.5).clamp(0, 1)).save(out / "source.png")

    metrics = {
        "model": args.model,
        "architecture": pipe.transformer.__class__.__name__,
        "scheduler": pipe.scheduler.__class__.__name__,
        "prediction_type": pipe.scheduler.config.prediction_type,
        "steps": args.steps,
        "latent_mae": float((reconstructed - latent0).abs().mean().cpu()),
        "pixel_mae": float((decoded - (x / 2 + 0.5)).abs().mean().cpu()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
