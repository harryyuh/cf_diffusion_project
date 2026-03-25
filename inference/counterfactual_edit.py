"""
Counterfactual image editing: given x and target parent f_target,

- use_vae_condition=True: x_cf ~ p_theta(x | z_rest, f_target) with frozen VAE for z_rest.
- use_vae_condition=False: x_cf ~ p_theta(x | f_target) (parent A only; no z_rest).
"""
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.morphomnist_dataset import MorphoMNISTDataset
from models.diffusion_unet import ConditionedUNet, UNetConfig
from models.vae import ConvVAE, VAEConfig
from utils.checkpoint import load_checkpoint
from utils.diffusion_utils import DiffusionConfig, GaussianDiffusion
from utils.seed import set_seed
from utils.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual thickness editing.")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_models(cfg: Dict[str, Any], device: torch.device) -> Tuple[Optional[ConvVAE], torch.nn.Module, GaussianDiffusion, int, bool]:
    """Load conditional diffusion; optionally load frozen VAE if use_vae_condition."""
    ckpt = torch.load(cfg["diffusion_checkpoint"], map_location="cpu")
    use_vae_condition = bool(cfg.get("use_vae_condition", True))
    if "use_vae_condition" in ckpt:
        use_vae_condition = bool(ckpt["use_vae_condition"])

    vae: Optional[ConvVAE] = None
    n_parent_dims = 0
    rest_dim = 0

    if use_vae_condition:
        vae_cfg_path = cfg.get("vae_config", "configs/vae.yaml")
        with open(vae_cfg_path, "r") as f:
            vae_cfg = yaml.safe_load(f)
        latent_dim = vae_cfg["latent_dim"]
        n_parent_dims = vae_cfg.get("n_parent_dims", 0)
        rest_dim = latent_dim - n_parent_dims

        vae_config = VAEConfig(
            in_channels=vae_cfg.get("in_channels", 1),
            latent_dim=latent_dim,
            hidden_dims=tuple(vae_cfg.get("hidden_dims", [32, 64, 128])),
            image_size=vae_cfg.get("image_size", 28),
            n_parent_dims=n_parent_dims,
            parent_pred_hidden=vae_cfg.get("parent_pred_hidden", 32),
            use_adversary=vae_cfg.get("use_adversary", False),
            adv_hidden=vae_cfg.get("adv_hidden", 32),
        )
        vae = ConvVAE(vae_config)
        load_checkpoint(Path(cfg["vae_checkpoint"]), model=vae, map_location="cpu")
        vae = vae.to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    cond_dim = rest_dim + 1 if use_vae_condition else 1

    unet_config = UNetConfig(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("unet_base_channels", 32),
        channel_mults=tuple(cfg.get("unet_channel_mults", [1, 2, 4])),
        time_emb_dim=cfg.get("time_emb_dim", 128),
        cond_dim=cond_dim,
    )
    unet = ConditionedUNet(unet_config)
    diffusion_model = unet

    if "unet_state_dict" in ckpt:
        unet.load_state_dict(ckpt["unet_state_dict"])
    else:
        diffusion_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()

    diff_config = DiffusionConfig(
        timesteps=cfg.get("timesteps", 1000),
        beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02),
        ddim_eta=cfg.get("ddim_eta", 0.0),
    )
    diffusion = GaussianDiffusion(diff_config)
    for attr in (
        "betas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
    ):
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    return vae, diffusion_model, diffusion, n_parent_dims, use_vae_condition


def run_counterfactual(
    vae: Optional[torch.nn.Module],
    diffusion_model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    x: torch.Tensor,
    z_rest: Optional[torch.Tensor],
    f_target: float,
    device: torch.device,
    sampling_mode: str = "from_xt",
    start_t: int = 300,
    use_vae_condition: bool = True,
    f_batch: Optional[torch.Tensor] = None,
    use_ddim_inversion: bool = True,
    encode_decode_split: bool = False,
) -> torch.Tensor:
    """
    Generate samples from diffusion. Reverse steps use DDIM when ddim_eta=0 (see GaussianDiffusion.p_sample).

    If encode_decode_split is False (default):
      - If f_batch is (B,1), use per-sample parent for cond on both encode and decode (reconstruction).
      - Else use scalar f_target for the whole batch.

    If encode_decode_split is True (counterfactual): inversion uses each sample's own thickness f_batch (and
    z_rest); reverse sampling uses f_target for all samples (and the same z_rest). Matches (f, z_rest) -> x_t
    then (f', z_rest) -> x0.

    If use_ddim_inversion and sampling_mode is from_xt: build x_t via multi-step DDIM encode (model eps at each
    step). Otherwise use one-shot q_sample with random Gaussian noise (training-time forward).
    """
    b = x.size(0)
    dt = z_rest.dtype if z_rest is not None else x.dtype

    def _cond(f_col: torch.Tensor) -> torch.Tensor:
        if use_vae_condition:
            assert z_rest is not None
            return torch.cat([z_rest, f_col], dim=1)
        return f_col

    if encode_decode_split:
        assert f_batch is not None, "encode_decode_split requires f_batch (source thickness per sample)"
        f_enc = f_batch.to(device=device, dtype=dt)
        if f_enc.dim() == 1:
            f_enc = f_enc.unsqueeze(1)
        f_dec = torch.full((b, 1), f_target, dtype=dt, device=device)
        cond_enc = _cond(f_enc)
        cond_dec = _cond(f_dec)
    else:
        if f_batch is not None:
            f = f_batch.to(device=device, dtype=dt)
            if f.dim() == 1:
                f = f.unsqueeze(1)
        else:
            f = torch.full((b, 1), f_target, dtype=dt, device=device)
        cond_enc = cond_dec = _cond(f)

    with torch.no_grad():
        if sampling_mode == "from_noise":
            x_cf = diffusion.p_sample_loop(diffusion_model, x.shape, cond_dec, device)
        else:
            # Start from mid-step noised version of original image to preserve identity.
            start_t = max(0, min(int(start_t), diffusion.timesteps - 1))
            if use_ddim_inversion:
                x_t = diffusion.ddim_encode_to_xt(diffusion_model, x, start_t, cond_enc, device)
            else:
                t_batch = torch.full((b,), start_t, device=device, dtype=torch.long)
                noise = torch.randn_like(x)
                x_t = diffusion.q_sample(x, t_batch, noise)
            x_cf = diffusion.p_sample_loop_from_xt(diffusion_model, x_t, start_t, cond_dec, device)
    return x_cf


def _stats(name: str, x: torch.Tensor) -> str:
    x = x.detach()
    return (
        f"{name}: shape={tuple(x.shape)} "
        f"min={x.min().item():.4f} max={x.max().item():.4f} "
        f"mean={x.mean().item():.4f} std={x.std().item():.4f}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 123))

    father_key = cfg.get("father_key", "thickness")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    vae, diffusion_model, diffusion, n_parent_dims, use_vae_condition = load_models(cfg, device)

    dataset = MorphoMNISTDataset(
        root=cfg["data_root"],
        split=cfg.get("split", "test"),
        mode=cfg.get("data_mode", "images_csv"),
        data_file=cfg.get("data_file"),
        metadata_file=cfg.get("metadata_file"),
        image_key=cfg.get("image_key", "images"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    if cfg.get("thickness_targets") is not None:
        thickness_targets = list(cfg["thickness_targets"])
    else:
        thickness_targets = [0.2, 0.4, 0.6, 0.8]
    recon_st_list: List[int] = [int(t) for t in (cfg.get("reconstruction_start_t_list") or [])]
    max_images: Optional[int] = cfg.get("max_images")
    debug_stats: bool = bool(cfg.get("debug_stats", False))
    sampling_mode: str = cfg.get("sampling_mode", "from_xt")  # from_xt or from_noise
    # True: x_t from multi-step DDIM inversion (model eps each step). False: one-shot q_sample + random noise.
    use_ddim_inversion: bool = bool(cfg.get("use_ddim_inversion", True))
    start_t: int = int(cfg.get("start_t", 300))
    # Evaluate each start_t with both reconstruction and counterfactual outputs.
    # Backward-compatible: if reconstruction_start_t_list is empty, use single start_t.
    start_t_list: List[int] = recon_st_list if recon_st_list else [start_t]

    # Grouping: each output "pair" consists of:
    # - original grid containing `grid_size` samples
    # - counterfactual grid for each target thickness, also containing `grid_size` samples
    grid_size: int = int(cfg.get("grid_size", 9))
    grid_nrow: int = int(cfg.get("grid_nrow", 3))
    if grid_nrow * grid_nrow != grid_size:
        # keep it permissive; save_image_grid handles generic nrow, but user asked 9 samples (3x3)
        pass

    metadata_rows: List[Dict[str, Any]] = []
    recon_metadata_rows: List[Dict[str, Any]] = []
    pending_original: List[torch.Tensor] = []
    pending_cf: Dict[int, List[List[torch.Tensor]]] = {
        st: [[] for _ in thickness_targets] for st in start_t_list
    }  # per start_t, per target f_t
    pending_recon: Dict[int, List[torch.Tensor]] = {st: [] for st in start_t_list}
    pending_father: List[float] = []
    pending_ids: List[int] = []

    count = 0
    desc = "Inference"
    if start_t_list and thickness_targets:
        desc = "Reconstruction + counterfactual"
    elif start_t_list:
        desc = "Reconstruction"
    else:
        desc = "Counterfactual edit"
    for batch in tqdm(loader, desc=desc):
        if max_images is not None and count >= max_images:
            break

        x = batch["image"].to(device)
        father_src = batch[father_key]
        if isinstance(father_src, torch.Tensor):
            father_src = father_src.cpu().numpy()
        else:
            father_src = [float(father_src)] * x.size(0)

        z_rest: Optional[torch.Tensor] = None
        if use_vae_condition:
            assert vae is not None
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = mu
                z_rest = z[:, n_parent_dims:]

        f_src = torch.as_tensor(father_src, device=device, dtype=torch.float32).view(-1, 1)

        # Reconstruction: same parent A as input; from_xt + DDIM reverse (see ddim_eta in config)
        recon_batch: Dict[int, torch.Tensor] = {}
        if start_t_list:
            # Reconstruction always uses from_xt (noised x at start_t); reverse is DDIM when ddim_eta=0.
            with torch.no_grad():
                for st in start_t_list:
                    recon_batch[st] = run_counterfactual(
                        vae,
                        diffusion_model,
                        diffusion,
                        x,
                        z_rest,
                        0.0,
                        device,
                        sampling_mode="from_xt",
                        start_t=st,
                        use_vae_condition=use_vae_condition,
                        f_batch=f_src,
                        use_ddim_inversion=use_ddim_inversion,
                    )

        # Counterfactuals for each start_t and each target thickness.
        cf_by_t: Dict[int, List[torch.Tensor]] = {}
        for st in start_t_list:
            cf_list: List[torch.Tensor] = []
            for f_t in thickness_targets:
                x_cf = run_counterfactual(
                    vae, diffusion_model, diffusion, x, z_rest, f_t, device,
                    sampling_mode=sampling_mode, start_t=st,
                    use_vae_condition=use_vae_condition,
                    f_batch=f_src,
                    use_ddim_inversion=use_ddim_inversion,
                    encode_decode_split=True,
                )
                cf_list.append(x_cf)
            cf_by_t[st] = cf_list

        if debug_stats and count == 0:
            # Print stats for the first batch only
            print(_stats("x", x))
            if z_rest is not None:
                print(_stats("z_rest", z_rest))
            for st in start_t_list:
                x_hat = recon_batch[st]
                mse = F.mse_loss(x_hat, x).item()
                print(f"reconstruction start_t={st} batch MSE vs x: {mse:.6f}")
                print(_stats(f"x_recon(start_t={st})", x_hat))
            for st in start_t_list:
                for f_i, f_t in enumerate(thickness_targets):
                    print(_stats(f"x_cf_raw(start_t={st},f={f_t})", cf_by_t[st][f_i]))
                    print(_stats(f"x_cf_clamped(start_t={st},f={f_t})", cf_by_t[st][f_i].clamp(0.0, 1.0)))

        n = x.size(0)
        for i in range(n):
            idx = count + i
            if max_images is not None and idx >= max_images:
                break

            pending_original.append(x[i : i + 1].detach().cpu())
            pending_father.append(float(father_src[i]))
            pending_ids.append(idx)

            for st in start_t_list:
                pending_recon[st].append(recon_batch[st][i : i + 1].detach().cpu())

            for st in start_t_list:
                for f_i, x_cf in enumerate(cf_by_t[st]):
                    pending_cf[st][f_i].append(x_cf[i : i + 1].detach().cpu())

            # Flush one group when we have enough samples
            if len(pending_original) == grid_size:
                group_id = pending_ids[0] // grid_size

                original_grid = torch.cat(pending_original, dim=0)  # (grid_size, 1, 28, 28)
                original_path = grid_dir / f"original_group_{group_id:05d}.png"
                save_image_grid(original_grid, nrow=grid_nrow, path=original_path, cmap="gray")

                for st in start_t_list:
                    recon_grid = torch.cat(pending_recon[st], dim=0).clamp(0.0, 1.0)
                    recon_path = grid_dir / f"reconstruction_start{st}_group_{group_id:05d}.png"
                    save_image_grid(recon_grid, nrow=grid_nrow, path=recon_path, cmap="gray")
                    mse_g = F.mse_loss(recon_grid, original_grid).item()
                    recon_metadata_rows.append(
                        {
                            "group_id": group_id,
                            "start_t": st,
                            "sample_ids": ",".join(map(str, pending_ids)),
                            "mse_vs_original": mse_g,
                            "grid_path": str(recon_path),
                        }
                    )
                    pending_recon[st].clear()

                # Save one counterfactual grid per target thickness
                for st in start_t_list:
                    for f_i, f_t in enumerate(thickness_targets):
                        cf_grid = torch.cat(pending_cf[st][f_i], dim=0)  # (grid_size, 1, 28, 28)
                        cf_grid = cf_grid.clamp(0.0, 1.0)
                        f_tag = str(f_t).replace(".", "p")
                        cf_path = grid_dir / f"counterfactual_start{st}_f{f_tag}_group_{group_id:05d}.png"
                        save_image_grid(cf_grid, nrow=grid_nrow, path=cf_path, cmap="gray")

                        metadata_rows.append(
                            {
                                "group_id": group_id,
                                "start_t": st,
                                "sample_ids": ",".join(map(str, pending_ids)),
                                "source_father_list": ",".join(map(str, pending_father)),
                                "target_f": f_t,
                                "original_grid_path": str(original_path),
                                "counterfactual_grid_path": str(cf_path),
                            }
                        )

                # Reset pending buffers
                pending_original.clear()
                pending_cf = {st: [[] for _ in thickness_targets] for st in start_t_list}
                pending_father.clear()
                pending_ids.clear()

        count += n

        if max_images is not None and count >= max_images:
            break

    # Save metadata CSV
    if thickness_targets:
        with open(output_dir / "counterfactual_metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "group_id",
                    "start_t",
                    "sample_ids",
                    "source_father_list",
                    "target_f",
                    "original_grid_path",
                    "counterfactual_grid_path",
                ],
            )
            writer.writeheader()
            writer.writerows(metadata_rows)

    if recon_metadata_rows:
        with open(output_dir / "reconstruction_metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["group_id", "start_t", "sample_ids", "mse_vs_original", "grid_path"],
            )
            writer.writeheader()
            writer.writerows(recon_metadata_rows)

    print(f"Done. Grids under {grid_dir}.")
    if thickness_targets:
        print(f"Counterfactual metadata: {output_dir / 'counterfactual_metadata.csv'}")
    if recon_metadata_rows:
        print(f"Reconstruction metadata: {output_dir / 'reconstruction_metadata.csv'}")


if __name__ == "__main__":
    main()
