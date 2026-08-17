from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image


ATTRS = ("Young", "Male", "No_Beard", "Bald")
PRESETS = {
    "Young": ((0.3, 0.3), 0.1, 0.1),
    "Male": ((0.3, 0.5), 0.2, 0.2),
    "No_Beard": ((0.3, 0.5), 0.0, 0.0),
    "Bald": ((0.5, 0.3), 0.0, 0.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attr", choices=ATTRS, required=True)
    p.add_argument("--selection-json", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--ca-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--controlnet-path", required=True)
    p.add_argument("--text-embedding-path", required=True)
    p.add_argument("--scm-path", required=True)
    p.add_argument("--base-model", default="lambdalabs/miniSD-diffusers")
    p.add_argument("--num-each", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--mask-steps", type=int, nargs="+", default=(0, 4, 9, 24, 49))
    return p.parse_args()


def add_ca_paths(ca_root: Path):
    bench = ca_root / "counterfactual-benchmark" / "counterfactual_benchmark"
    for path in (ca_root / "diffusers" / "src", ca_root / "notebook_benchmarks", ca_root, bench):
        sys.path.insert(0, str(path))


def read_attrs(root: Path):
    lines = (root / "list_attr_celeba.txt").read_text().strip().splitlines()
    cols = lines[1].split()
    return {
        parts[0]: {key: int(value) for key, value in zip(cols, parts[1:])}
        for line in lines[2:]
        if len(parts := line.split()) == len(cols) + 1
    }


def main():
    args = parse_args()
    ca_root = Path(args.ca_root)
    data_root = Path(args.data_root)
    add_ca_paths(ca_root)

    from inference_utils import build_transforms, load_causal_adapter
    from causal_modules.ddim_modules import P2P_editing
    from causal_modules.p2p_edits.attention_control import LocalBlend

    requested_steps = set(args.mask_steps)
    captured = {step: [] for step in requested_steps}
    original_init = LocalBlend.__init__
    original_call = LocalBlend.__call__

    def capture_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        original_get_mask = self.get_mask

        def capture_get_mask(maps, alpha, use_pool):
            mask = original_get_mask(maps, alpha, use_pool)
            if use_pool:
                self._capture_primary = mask.detach()
            else:
                self._capture_substruct = mask.detach()
            return mask

        self.get_mask = capture_get_mask

    def capture_call(self, x_t, attention_store):
        self._capture_primary = None
        self._capture_substruct = None
        out = original_call(self, x_t, attention_store)
        step = self.counter - 1
        if step in requested_steps and self._capture_primary is not None:
            mask = self._capture_primary.bool()
            if self.substruct_layers is not None and self._capture_substruct is not None:
                mask = mask & (~self._capture_substruct.bool())
            batch_size = mask.shape[0] // 2
            captured[step].append(mask[batch_size:].float().cpu())
        return out

    LocalBlend.__init__ = capture_init
    LocalBlend.__call__ = capture_call

    device = torch.device("cuda")
    assets = load_causal_adapter(
        "celeA_complex",
        base_model_path=args.base_model,
        controlnet_path=args.controlnet_path,
        text_embedding_path=args.text_embedding_path,
        scm_path=args.scm_path,
        device=device,
        torch_dtype=torch.float32,
    )
    image_tfm, _, _ = build_transforms("celeA_complex", size=256)
    attrs = read_attrs(data_root)
    selection = json.loads(Path(args.selection_json).read_text())
    filenames = (
        selection["male_filenames"][: args.num_each]
        + selection["female_filenames"][: args.num_each]
    )
    attr_idx = ATTRS.index(args.attr)
    thresholds, cross_steps, self_steps = PRESETS[args.attr]

    for filename in filenames:
        pil = Image.open(data_root / "img_align_celeba" / filename).convert("RGB")
        image = image_tfm(pil).unsqueeze(0).to(device=device, dtype=torch.float32)
        row = attrs[filename]
        label = torch.tensor(
            [[1.0 if row[name] > 0 else 0.0 for name in ATTRS]],
            device=device,
            dtype=torch.float32,
        )
        intervention_value = 1.0 - label[:, attr_idx]
        P2P_editing(
            assets.pipe,
            image,
            label,
            assets.presudo_token_ids,
            assets.prompt,
            assets.presudo_list,
            num_steps=args.steps,
            invert_guidance_scale=1.0,
            set_guidance_scale=3.0,
            intervention_indx=attr_idx,
            intervention_values=intervention_value.squeeze(0),
            return_PIL=True,
            blend_word=True,
            blend_params={"start_blend": 0.0, "th": thresholds},
            disentangle=False,
            cross_replace_steps=cross_steps,
            self_replace_steps=self_steps,
            DSCM_labels=None,
        )

    out_dir = Path(args.out_root) / f"masks_do_{args.attr}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Causal Adapter official P2P_editing",
        "attr": args.attr,
        "num_images": len(filenames),
        "steps": args.steps,
        "thresholds": thresholds,
        "cross_replace_steps": cross_steps,
        "self_replace_steps": self_steps,
        "mask_area_ratio": {},
    }
    for step in sorted(captured):
        masks = torch.cat(captured[step], dim=0)
        per_image = masks.mean(dim=(1, 2, 3))
        summary["mask_area_ratio"][str(step)] = {
            "mean": float(per_image.mean()),
            "std": float(per_image.std(unbiased=False)),
            "min": float(per_image.min()),
            "max": float(per_image.max()),
            "per_image": [float(value) for value in per_image],
        }
        masks64 = F.interpolate(masks, size=(64, 64), mode="nearest")
        save_image(masks64, out_dir / f"mask_step_{step:02d}.png", nrow=8, padding=0)
        torch.save(masks, out_dir / f"mask_step_{step:02d}.pt")
    (out_dir / "mask_area_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
