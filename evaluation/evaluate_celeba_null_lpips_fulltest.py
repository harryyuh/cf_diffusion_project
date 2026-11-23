"""One-cycle null-intervention reconstruction on a fixed CelebA test shard.

Uses the same released CA and Label-only checkpoints and DDIM settings as the
completed full-test, but sets the prediction condition equal to the observed
condition. The reported LPIPS is composition, not source-to-edited LPIPS.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset
from evaluation import compare_causal_adapter_parent_only_celeba_n2048 as ca_eval
from evaluation import evaluate_latent_prompt_lora_celeba_n2048 as label_eval


ATTRS = ("Young", "Male", "No_Beard", "Bald")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("ca", "label_only"), required=True)
    p.add_argument("--selection-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--ca-root", type=Path, required=True)
    p.add_argument("--ca-run", type=Path, required=True)
    p.add_argument("--label-config", type=Path, required=True)
    p.add_argument("--label-checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Smoke test only; 0 uses the whole shard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for null-edit evaluation")

    selection = json.loads(args.selection_json.read_text())
    indices = [int(i) for key in ("male_indices", "female_indices") for i in selection[key]]
    if args.limit:
        indices = indices[: args.limit]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Selection must be nonempty and contain unique indices")

    dataset = CelebADataset(
        root=str(args.data_root), split="test", factor_cols=list(CELEBA_ATTR_ORDER), image_size=64
    )
    loader = DataLoader(
        ca_eval.FilenameSubset(dataset, indices),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    # The released benchmark implementation selects net_type='alex'.
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    lpips.eval()

    ca_generator = None
    label_editor = None
    if args.method == "ca":
        ca_eval._add_ca_paths(args.ca_root)
        ca_eval._patch_safety_checker_compat()
        ca_args = SimpleNamespace(
            ca_base_model_path="lambdalabs/miniSD-diffusers",
            ca_controlnet_path=str(args.ca_run / "controlnet-steps-200000.safetensors"),
            ca_text_embedding_path=str(args.ca_run / "learned_embeds-steps-200000-token-hash-alias.safetensors"),
            ca_scm_path=str(args.ca_run / "best_model.pt"),
        )
        ca_generator = ca_eval.CausalAdapterGenerator(ca_args, device, args.data_root)
    else:
        with args.label_config.open() as f:
            cfg = yaml.safe_load(f)
        label_editor = label_eval.PromptLoraEditor(
            cfg,
            args.label_checkpoint / "lora",
            device, 100, 3.0,
            invert_guidance_scale=1.0,
            negative_prompt="ugly, blurry, low res, unrealistic",
            scheduler_mode="ca",
            vae_latent_mode="mean",
            condition_mode="joint_label",
            pai_checkpoint=args.label_checkpoint / "mapper" / "pai_mapper.pt",
        )

    n = 0
    mae_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            filenames = batch["filename"]
            # CA's original transformation is the common comparison reference:
            # CenterCrop(150) -> bilinear 256 -> bicubic 64 -> [0,1].
            if ca_generator is not None:
                real = ca_generator.official_real64(batch)
                image_t = torch.cat(
                    [ca_generator._official_source(str(name)) for name in filenames], dim=0
                )
                labels = torch.cat(
                    [(batch[k].to(device).float().view(-1, 1) > 0).float() for k in ATTRS],
                    dim=1,
                )
                images, _, returned_labels, _ = ca_generator.ddim(
                    ca_generator.assets.pipe, image_t, labels,
                    ca_generator.assets.presudo_token_ids,
                    [ca_generator.assets.prompt] * len(filenames),
                    num_steps=100, invert_guidance_scale=1.0,
                    set_guidance_scale=3.0,
                    intervention_indx=None, intervention_values=None,
                    return_PIL=True, DSCM_labels=None,
                )
                if not torch.equal(returned_labels.squeeze(-1), labels):
                    raise RuntimeError("CA null intervention modified the conditioning labels")
                fake = torch.stack(
                    [ca_generator._pil_to_tensor64(im) for im in images], dim=0
                ).clamp(0, 1)
            else:
                assert label_editor is not None
                # Use the same factual image reference as CA, without loading
                # the CA model into the Label-only process.
                tfm = ca_eval.transforms.Compose([
                    ca_eval.transforms.CenterCrop(150),
                    ca_eval.transforms.Resize(
                        (256, 256), interpolation=ca_eval.transforms.InterpolationMode.BILINEAR
                    ),
                    ca_eval.transforms.Resize(
                        (64, 64), interpolation=ca_eval.transforms.InterpolationMode.BICUBIC
                    ),
                    ca_eval.transforms.ToTensor(),
                ])
                real = torch.stack([
                    tfm(ca_eval.Image.open(args.data_root / "img_align_celeba" / str(name)).convert("RGB"))
                    for name in filenames
                ]).to(device)
                labels_pm1 = torch.stack(
                    [batch[k].to(device).float().view(-1) for k in ATTRS], dim=1
                )
                fake = label_editor.edit(
                    args.data_root, filenames, labels_pm1, labels_pm1,
                    editing="standard", inversion="ddim",
                ).clamp(0, 1)
            if real.shape != fake.shape or real.shape[1:] != (3, 64, 64):
                raise RuntimeError(f"Image shape mismatch: real {real.shape}, null {fake.shape}")
            lpips.update(fake.float(), real.float())
            mae_sum += float((fake.float() - real.float()).abs().mean((1, 2, 3)).sum().cpu())
            n += fake.shape[0]

    if n != len(indices):
        raise RuntimeError(f"Expected {len(indices)} examples, generated {n}")
    result = {
        "method": args.method,
        "n": n,
        "source_indices": indices,
        "null_edit": True,
        "cycles": 1,
        "ddim_steps": 100,
        "inversion_cfg": 1.0,
        "sampling_cfg": 3.0,
        "lpips_net": "alex",
        "lpips_composition": float(lpips.compute().cpu()),
        "mae_01": mae_sum / n,
        "smoke_only": bool(args.limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_indices"}, indent=2))


if __name__ == "__main__":
    main()
