"""Paired Pendulum evaluation for released Causal-Adapter and our PAI+LoRA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

from data.pendulum_dataset import (
    PENDULUM_ATTRS,
    PENDULUM_MINMAX,
    PendulumDataset,
    denormalize_pendulum,
    normalize_pendulum,
)
from training.train_pendulum_regressor import PendulumRegressor

CA_MEAN_SPREAD = torch.tensor([[2, 42], [104, 44], [7.5, 4.5], [11, 8]], dtype=torch.float32)
CA_GRAPH = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['ca', 'pai'], required=True)
    p.add_argument('--data-root', default='/scratch/gilbreth/yu1331/datasets/pendulum')
    p.add_argument('--ca-root', default='/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter')
    p.add_argument('--base-model', default='lambdalabs/miniSD-diffusers')
    p.add_argument('--ca-controlnet', default='/scratch/gilbreth/yu1331/models/Causal-Adapter/pendulum/controlnet/controlnet-steps-20000.safetensors')
    p.add_argument('--ca-embedding', default='/scratch/gilbreth/yu1331/models/Causal-Adapter/pendulum/controlnet/learned_embeds-steps-20000.safetensors')
    p.add_argument('--ca-scm', default='/scratch/gilbreth/yu1331/models/Causal-Adapter/pendulum/scm/best_model.pt')
    p.add_argument('--pai-config', default='configs/pai_lora_pendulum.yaml')
    p.add_argument('--pai-root', default='/scratch/gilbreth/yu1331/ckpts/pendulum/pai_lora/minisd_continuous_pai_lora/final')
    p.add_argument('--regressor', default='/scratch/gilbreth/yu1331/ckpts/pendulum/regressor/best.pt')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--grid-samples', type=int, default=8)
    return p.parse_args()


def raw_to_ca(raw: torch.Tensor) -> torch.Tensor:
    scale = CA_MEAN_SPREAD.to(raw)
    return (raw - scale[:, 0]) / scale[:, 1]


def ca_to_raw(value: torch.Tensor) -> torch.Tensor:
    scale = CA_MEAN_SPREAD.to(value)
    return value * scale[:, 1] + scale[:, 0]


def load_scm(ca_root: str, checkpoint: str, device: torch.device):
    if ca_root not in sys.path:
        sys.path.insert(0, ca_root)
    from causal_modules.control_heads import ControlNetConditioningEmbedding
    head = ControlNetConditioningEmbedding(in_dim=4, hidden_dims=16, dataset_name='pendulum').to(device)
    state = torch.load(checkpoint, map_location=device)
    head.load_state_dict(state)
    head.update_mask(CA_GRAPH.to(device))
    head.eval()
    return head


@torch.no_grad()
def scm_target(head, source_raw: torch.Tensor, intervention_idx: int, intervention_raw: torch.Tensor):
    source_ca = raw_to_ca(source_raw)
    value_ca = raw_to_ca(intervention_raw)[:, intervention_idx]
    out, _ = head.inference(source_ca, intervention_indx=intervention_idx, intervention_values=value_ca)
    while out.ndim > 2:
        out = out.squeeze(-1)
    target_raw = ca_to_raw(out.float())
    target_raw[:, intervention_idx] = intervention_raw[:, intervention_idx]
    return out.float(), target_raw, normalize_pendulum(target_raw)


def load_regressor(path: str, device: torch.device):
    payload = torch.load(path, map_location=device)
    model = PendulumRegressor().to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    return model


def pil_to_tensor(images, size=128):
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return torch.stack([tfm(x.convert('RGB')) for x in images])


def tensor_to_pil(images: torch.Tensor):
    return [transforms.ToPILImage()(x.detach().cpu().clamp(0, 1)) for x in images]


class PAIEditor:
    def __init__(self, config_path: str, root: str, device: torch.device, steps: int):
        from evaluation.evaluate_latent_prompt_lora_celeba import PromptLoraEditor
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.inner = PromptLoraEditor(
            cfg, Path(root) / 'lora', device, steps, guidance_scale=1.0,
            invert_guidance_scale=1.0, negative_prompt='', scheduler_mode='ca',
            vae_latent_mode='mean', condition_mode='pai',
            pai_checkpoint=Path(root) / 'mapper' / 'pai_mapper.pt',
        )
        self.inner.image_tfm = transforms.Compose([
            transforms.Resize((int(cfg.get('image_size', 256)),) * 2),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])

    @torch.no_grad()
    def edit(self, paths, source_unit, target_unit):
        # Reuse the tested MiniSD DDIM inversion/sampling path, overriding only
        # CelebA's path convention and crop.
        pipe = self.inner.pipe
        x = torch.stack([self.inner.image_tfm(Image.open(p).convert('RGB')) for p in paths]).to(
            self.inner.device, dtype=pipe.vae.dtype)
        dist = pipe.vae.encode(x).latent_dist
        lat0 = dist.mean * pipe.vae.config.scaling_factor
        pe_src, ne = self.inner.condition_embeds(source_unit)
        pe_tgt, _ = self.inner.condition_embeds(target_unit)
        lat_t = self.inner.invert(lat0, pe_src, ne)
        return tensor_to_pil(self.inner.decode64(self.inner.sample(lat_t, pe_tgt, ne)))


class CAEditor:
    def __init__(self, args, device):
        if args.ca_root not in sys.path:
            sys.path.insert(0, args.ca_root)
        nb = str(Path(args.ca_root) / 'notebook_benchmarks')
        if nb not in sys.path:
            sys.path.insert(0, nb)
        from inference_utils import build_transforms, load_causal_adapter
        self.sample_tfm, _, _ = build_transforms('pendulum', size=256)
        self.assets = load_causal_adapter(
            'pendulum', base_model_path=args.base_model,
            controlnet_path=args.ca_controlnet, text_embedding_path=args.ca_embedding,
            scm_path=args.ca_scm,
        )
        self.steps = args.steps

    @torch.no_grad()
    def edit(self, paths, source_raw, intervention_idx, intervention_raw):
        from causal_modules.ddim_modules import ddim_editing, sample
        outputs = []
        for path, raw, do_raw in zip(paths, source_raw, intervention_raw):
            image = Image.open(path).convert('RGB')
            image_t = self.sample_tfm(image).unsqueeze(0)
            label = raw_to_ca(raw.unsqueeze(0)).to(self.assets.device)
            _, trajectory, _, uncond = ddim_editing(
                self.assets.pipe, image_t, label.clone(), self.assets.presudo_token_ids,
                self.assets.prompt, num_steps=self.steps, invert_guidance_scale=1.0,
                set_guidance_scale=1.0, intervention_indx=None,
                intervention_values=None, return_PIL=True,
            )
            value = raw_to_ca(do_raw.unsqueeze(0))[0, intervention_idx].item()
            images, _ = sample(
                self.assets.pipe, self.assets.prompt, self.assets.presudo_token_ids,
                start_step=0, start_latents=trajectory[-1].clone(), guidance_scale=1.0,
                num_inference_steps=self.steps, num_images_per_prompt=1,
                negative_prompt=None, device=self.assets.device, controlnet_image=None,
                intervention_indx=intervention_idx, intervention_values=value,
                label=label.clone(), return_PIL=True, disentangle=False,
                uncond_embeddings=uncond,
            )
            outputs.append(images[0].convert('RGB'))
        return outputs


def make_pair_grid(records, output: Path, title: str):
    if not records:
        return
    cell = 192
    canvas = Image.new('RGB', (cell * 2, cell * len(records) + 28), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title + ' — source | counterfactual', fill='black')
    for row, (src, cf) in enumerate(records):
        canvas.paste(src.resize((cell, cell)), (0, 28 + row * cell))
        canvas.paste(cf.resize((cell, cell)), (cell, 28 + row * cell))
    canvas.save(output)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda')
    output = Path(args.output_dir) / args.method
    image_root = output / 'images'
    grid_root = output / 'grids'
    image_root.mkdir(parents=True, exist_ok=True)
    grid_root.mkdir(parents=True, exist_ok=True)

    test = PendulumDataset(args.data_root, 'test', image_size=256)
    train = PendulumDataset(args.data_root, 'train', image_size=256)
    indices = list(range(min(args.max_samples, len(test))))
    # The benchmark samples intervention values from the train distribution.
    intervention_sources = {
        j: rng.integers(0, len(train), size=len(indices)).tolist() for j in range(4)
    }
    (output / 'selection.json').write_text(json.dumps({
        'seed': args.seed, 'test_indices': indices,
        'intervention_train_indices': intervention_sources,
    }, indent=2))

    scm = load_scm(args.ca_root, args.ca_scm, device)
    regressor = load_regressor(args.regressor, device)
    editor = CAEditor(args, device) if args.method == 'ca' else PAIEditor(args.pai_config, args.pai_root, device, args.steps)
    results = {'method': args.method, 'n': len(indices), 'steps': args.steps, 'interventions': {}}

    for do_idx, do_name in enumerate(PENDULUM_ATTRS):
        sums = torch.zeros(4)
        pixel_mae = 0.0
        seen = 0
        grid_records = []
        do_dir = image_root / do_name
        do_dir.mkdir(parents=True, exist_ok=True)
        for start in tqdm(range(0, len(indices), args.batch_size), desc=f'{args.method}:{do_name}'):
            batch_ids = indices[start:start + args.batch_size]
            src_items = [test[i] for i in batch_ids]
            paths = [x['path'] for x in src_items]
            source_raw = torch.stack([x['raw_factors'] for x in src_items]).to(device)
            source_unit = torch.stack([x['factors'] for x in src_items]).to(device)
            train_ids = intervention_sources[do_idx][start:start + len(batch_ids)]
            intervention_raw = source_raw.clone()
            intervention_raw[:, do_idx] = torch.stack([train[i]['raw_factors'][do_idx] for i in train_ids]).to(device)
            _, target_raw, target_unit = scm_target(scm, source_raw, do_idx, intervention_raw)
            if args.method == 'ca':
                cf = editor.edit(paths, source_raw, do_idx, intervention_raw)
            else:
                cf = editor.edit(paths, source_unit, target_unit)
            cf_tensor = pil_to_tensor(cf).to(device)
            pred_unit = regressor(cf_tensor)
            pred_raw = denormalize_pendulum(pred_unit).to(device)
            sums += (pred_raw - target_raw).abs().sum(0).cpu()
            src_tensor = pil_to_tensor([Image.open(p).convert('RGB') for p in paths]).to(device)
            pixel_mae += float((cf_tensor - src_tensor).abs().mean((1, 2, 3)).sum().cpu())
            for local, image in enumerate(cf):
                image.save(do_dir / f'{batch_ids[local]:05d}.png')
                if len(grid_records) < args.grid_samples:
                    grid_records.append((Image.open(paths[local]).convert('RGB'), image))
            seen += len(batch_ids)
        mae = sums / seen
        make_pair_grid(grid_records, grid_root / f'{do_name}.png', f'{args.method.upper()} do({do_name})')
        results['interventions'][do_name] = {
            'target_mae_raw': {name: float(mae[i]) for i, name in enumerate(PENDULUM_ATTRS)},
            'mean_target_mae_raw': float(mae.mean()),
            'pixel_mae_01': pixel_mae / seen,
        }
    per_do = [x['mean_target_mae_raw'] for x in results['interventions'].values()]
    results['mean_over_interventions'] = float(np.mean(per_do))
    (output / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
