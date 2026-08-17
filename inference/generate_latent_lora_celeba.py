"""Generate CelebA-style samples from a latent LoRA fine-tuned model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from data.celeba_dataset import expand_env_in_cfg


def _pipeline_class(model_family: str):
    family = model_family.lower()
    if family == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        return StableDiffusionXLPipeline
    if family in ("sd", "sd15", "sd2"):
        from diffusers import StableDiffusionPipeline

        return StableDiffusionPipeline
    raise ValueError("model_family must be 'sdxl' or 'sd'")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate samples from a CelebA latent LoRA checkpoint.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--lora-dir", type=str, default="", help="Defaults to output_dir/run_name/lora from config.")
    p.add_argument("--prompt", action="append", default=[], help="Prompt to generate; repeat for multiple prompts.")
    p.add_argument("--out-dir", type=str, default="", help="Defaults to output_dir/run_name/samples.")
    p.add_argument("--num-images-per-prompt", type=int, default=2)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    model_family = str(cfg.get("model_family", "sdxl"))
    dtype = torch.bfloat16 if str(cfg.get("mixed_precision", "bf16")).lower() == "bf16" else torch.float16
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    run_root = Path(cfg["output_dir"]) / str(cfg.get("run_name", "sdxl_lora_celeba"))
    lora_dir = Path(args.lora_dir) if args.lora_dir else run_root / "lora"
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe_cls = _pipeline_class(model_family)
    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    pipe = pipe_cls.from_pretrained(str(cfg["pretrained_model_name_or_path"]), **kwargs).to(device)
    pipe.load_lora_weights(str(lora_dir))

    prompts: List[str] = args.prompt or [
        "a high quality portrait photo of a smiling woman, realistic skin texture, sharp focus",
        "a high quality portrait photo of a smiling man, realistic skin texture, sharp focus",
    ]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    idx = 0
    for prompt in prompts:
        images = pipe(
            prompt=prompt,
            num_images_per_prompt=args.num_images_per_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images
        safe = "".join(ch if ch.isalnum() else "_" for ch in prompt[:48]).strip("_") or "sample"
        for image in images:
            path = out_dir / f"{idx:03d}_{safe}.png"
            image.save(path)
            print(path)
            idx += 1


if __name__ == "__main__":
    main()
