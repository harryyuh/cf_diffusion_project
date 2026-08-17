"""CelebA Male<->Female counterfactual grids with SD3/MMDiT img2img LoRA.

The output schema mirrors ``inference.generate_celeba_counterfactual_male``:
``grid_male_to_female.png``, ``grid_female_to_male.png``, ``selection.json``, and
``meta.json``. The only semantic condition is gender text (man/woman).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from PIL import Image

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset, expand_env_in_cfg
from inference.generate_celeba_counterfactual_male import (
    _default_male_female_indices,
    _resolve_data_root,
    _selection_record,
    _validate_indices_for_male_female,
)
from utils.visualization import save_image_grid



def _load_selection_indices_lenient(path: Path, ds: CelebADataset, parent_key: str) -> tuple[List[int], List[int], Dict[str, Any]]:
    """Reuse old benchmark indices even if this SD3 config uses a different resize/column set."""
    with open(path, "r") as f:
        raw = json.load(f)
    if "male_indices" not in raw or "female_indices" not in raw:
        raise ValueError(f"{path}: JSON must contain male_indices and female_indices")
    male_indices = [int(x) for x in raw["male_indices"]]
    female_indices = [int(x) for x in raw["female_indices"]]
    _validate_indices_for_male_female(ds, parent_key, male_indices, female_indices)
    if raw.get("male_filenames"):
        for i, expected in zip(male_indices, raw["male_filenames"]):
            if ds._filenames[i] != expected:
                raise ValueError(f"male index {i}: expected filename {expected}, dataset has {ds._filenames[i]}")
    if raw.get("female_filenames"):
        for i, expected in zip(female_indices, raw["female_filenames"]):
            if ds._filenames[i] != expected:
                raise ValueError(f"female index {i}: expected filename {expected}, dataset has {ds._filenames[i]}")
    return male_indices, female_indices, raw

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SD3/MMDiT gender-only CelebA counterfactual grids.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--lora-dir", type=str, default="")
    p.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    p.add_argument("--n-each", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--strength", type=float, default=0.45)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--lora-scale", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--selection-json", type=str, default="")
    return p.parse_args()


def _pipeline(cfg: Dict[str, Any], dtype: torch.dtype, device: torch.device):
    from diffusers import StableDiffusion3Img2ImgPipeline

    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(str(cfg["pretrained_model_name_or_path"]), **kwargs)
    return pipe.to(device)


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    data = data.view(size, size, 3).permute(2, 0, 1).float() / 255.0
    return data


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


def _target_override_value(key: str, gender_value: float, cfg: Dict[str, Any]) -> float | None:
    overrides = cfg.get("target_attr_by_gender") or {}
    target = "male" if float(gender_value) > 0 else "female"
    values = overrides.get(target) or overrides.get(target.capitalize()) or {}
    if key not in values:
        return None
    return float(values[key])


def _prompt_for_batch(batch: Dict[str, Any], row: int, gender_value: float, cfg: Dict[str, Any]) -> str:
    prefix = str(cfg.get("caption_prefix", "a portrait photo of a"))
    gender_key = str(cfg.get("gender_key", "Male"))
    attr_cols = list(cfg.get("caption_attr_cols") or cfg.get("prompt_attr_cols") or [gender_key])
    if gender_key not in attr_cols:
        attr_cols = [gender_key, *attr_cols]
    phrases: List[str] = []
    for key in attr_cols:
        if key == gender_key:
            value = gender_value
        else:
            override = _target_override_value(key, gender_value, cfg)
            value = override if override is not None else float(batch[key][row].view(-1)[0])
        phrases.append(_phrase_for_attr(key, value))
    return ", ".join([prefix, *phrases])


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[1] / cfg_path
    cfg = _load_yaml(cfg_path)

    precision = str(cfg.get("mixed_precision", "bf16")).lower()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision in ("fp16", "float16") else torch.float32
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(cfg.get("image_size", 512))
    parent_keys: List[str] = list(cfg.get("parent_keys") or [cfg.get("gender_key", "Male")])
    gender_key = parent_keys[0]

    factor_cols = list(CELEBA_ATTR_ORDER) if cfg.get("use_all_celeba_attrs") else list(cfg.get("celeba_attr_cols", [gender_key]))
    if gender_key not in factor_cols:
        factor_cols.append(gender_key)
    ds = CelebADataset(root=cfg["data_root"], split=args.split, factor_cols=factor_cols, image_size=image_size)

    if args.selection_json.strip():
        male_idx, female_idx, old_selection = _load_selection_indices_lenient(Path(args.selection_json), ds, gender_key)
        selection = _selection_record(
            data_root=_resolve_data_root(cfg),
            split=str(args.split),
            image_size=image_size,
            factor_cols=factor_cols,
            parent_keys=parent_keys,
            n_each=len(male_idx),
            male_indices=male_idx,
            female_indices=female_idx,
            ds=ds,
        )
        selection["source_selection_json"] = str(Path(args.selection_json).resolve())
        selection["source_selection_schema_version"] = old_selection.get("schema_version")
    else:
        male_idx, female_idx = _default_male_female_indices(ds, gender_key, int(args.n_each))
        _validate_indices_for_male_female(ds, gender_key, male_idx, female_idx)
        selection = _selection_record(
            data_root=_resolve_data_root(cfg),
            split=str(args.split),
            image_size=image_size,
            factor_cols=factor_cols,
            parent_keys=parent_keys,
            n_each=int(args.n_each),
            male_indices=male_idx,
            female_indices=female_idx,
            ds=ds,
        )

    pipe = _pipeline(cfg, dtype, device)
    run_root = Path(cfg["output_dir"]) / str(cfg.get("run_name", "sd3_gender_lora_celeba"))
    lora_dir = Path(args.lora_dir) if args.lora_dir else run_root / "lora"
    pipe.load_lora_weights(str(lora_dir))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    negative_prompt = str(cfg.get("negative_prompt", "blurry, low resolution, jpeg artifacts, distorted face"))

    def run_grid(name: str, indices: List[int], flip_to: float) -> None:
        subset = Subset(ds, indices)
        loader = DataLoader(subset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
        tiles: List[torch.Tensor] = []
        for batch in loader:
            x = batch["image"]
            prompt = [_prompt_for_batch(batch, i, flip_to, cfg) for i in range(x.size(0))]
            pil_images = [_tensor_to_pil(x[i]) for i in range(x.size(0))]
            result = pipe(
                prompt=prompt,
                image=pil_images,
                strength=float(args.strength),
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance_scale),
                negative_prompt=[negative_prompt] * len(prompt),
                generator=generator,
                joint_attention_kwargs={"scale": float(args.lora_scale)},
                height=image_size,
                width=image_size,
            ).images
            for bi, img in enumerate(result):
                tiles.append(x[bi])
                tiles.append(_pil_to_tensor(img, image_size))
        if tiles:
            save_image_grid(torch.stack(tiles, dim=0), nrow=2, path=out_dir / f"grid_{name}.png", cmap=None, dpi=100.0)

    run_grid("male_to_female", male_idx, -1.0)
    run_grid("female_to_male", female_idx, 1.0)

    with open(out_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    meta = {
        "schema_version": 1,
        "script": "generate_celeba_counterfactual_male_dit_img2img.py",
        "selection": selection,
        "selection_json": str(Path(args.selection_json).resolve()) if args.selection_json.strip() else "",
        "generation": {
            "method": "sd3_img2img_gender_only_prompt",
            "strength": float(args.strength),
            "steps": int(args.steps),
            "guidance_scale": float(args.guidance_scale),
            "lora_scale": float(args.lora_scale),
            "grid_files": ["grid_male_to_female.png", "grid_female_to_male.png"],
        },
        "model": {
            "config": str(cfg_path),
            "pretrained_model_name_or_path": str(cfg["pretrained_model_name_or_path"]),
            "lora_dir": str(lora_dir),
            "condition": "gender-only text prompt from Male attribute",
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote SD3/MMDiT gender-only grids under {out_dir}")


if __name__ == "__main__":
    main()
