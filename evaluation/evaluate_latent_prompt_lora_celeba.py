"""Evaluate MiniSD LoRA full-label-prompt counterfactuals on CelebA interventions."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
COMPLEX_ATTRS = ["Young", "Male", "No_Beard", "Bald"]
PHRASE_MAP = {
    "Young": ("young", "older"),
    "Male": ("male", "female"),
    "No_Beard": ("no beard", "beard"),
    "Bald": ("bald", "with hair"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/latent_lora_celeba_ca_pipeline_full_label_prompt.yaml")
    p.add_argument("--lora-dir", default="")
    p.add_argument("--selection-json", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--invert-guidance-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--scheduler-mode", choices=["pretrained", "ca"], default="pretrained")
    p.add_argument("--vae-latent-mode", choices=["sample", "mean"], default="sample")
    p.add_argument("--editing", choices=["standard", "ptp"], default="standard")
    p.add_argument("--condition-mode", choices=["prompt", "pai", "joint_label"], default="prompt")
    p.add_argument("--pai-checkpoint", default="")
    p.add_argument("--inversion", choices=["ddim", "direct"], default="ddim")
    p.add_argument("--ptp-cross-replace-steps", type=float, default=0.2)
    p.add_argument("--ptp-self-replace-steps", type=float, default=0.2)
    p.add_argument("--ptp-controller", choices=["simple", "ca", "ca_refine"], default="simple")
    p.add_argument("--ptp-local-blend", action="store_true")
    p.add_argument("--ptp-blend-th", type=float, nargs=2, default=(0.3, 0.3))
    p.add_argument("--ptp-start-blend", type=float, default=0.0)
    p.add_argument("--ptp-substruct-attr-indices", type=int, nargs="*", default=())
    p.add_argument("--benchmark-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark")
    p.add_argument("--scm-checkpoint", default="/scratch/gilbreth/yu1331/ckpts/celeba/full_label_scm/causal_adapter_graph/checkpoints/full_label_scm_best.pt")
    p.add_argument("--target-scm", choices=["ours", "causal_adapter"], default="ours")
    p.add_argument("--intervention-attr", choices=COMPLEX_ATTRS, default="Male")
    p.add_argument("--causal-adapter-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter")
    p.add_argument("--causal-adapter-scm", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-grid", default="")
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    from data.celeba_dataset import expand_env_in_cfg

    expand_env_in_cfg(cfg)
    return cfg


class FilenameSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.filenames = getattr(dataset, "_filenames", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        src = self.indices[idx]
        item = dict(self.dataset[src])
        item["source_index"] = torch.tensor(src, dtype=torch.long)
        if self.filenames is not None:
            item["filename"] = self.filenames[src]
        return item


def _pipeline_class(model_family: str):
    if model_family.lower() in ("sd", "sd15", "sd2"):
        from diffusers import StableDiffusionPipeline

        return StableDiffusionPipeline
    raise ValueError("This evaluator currently supports SD/MiniSD pipeline only")


def _load_indices(path: str, n: int) -> List[int]:
    with open(path) as f:
        raw = json.load(f)
    idxs = [int(x) for x in raw.get("male_indices", [])] + [int(x) for x in raw.get("female_indices", [])]
    return idxs[:n]


def _labels_pm1(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.stack([batch[k].to(device).float().view(-1) for k in COMPLEX_ATTRS], dim=1)


def _prompts_from_pm1(labels_pm1: torch.Tensor, cfg: Dict[str, Any]) -> List[str]:
    prefix = str(cfg.get("full_label_prompt_prefix", "a human face"))
    suffix = str(cfg.get("full_label_prompt_suffix", ""))
    label_cols = list(cfg.get("prompt_label_cols") or COMPLEX_ATTRS)
    prompts: List[str] = []
    for i in range(labels_pm1.shape[0]):
        phrases: List[str] = [prefix]
        for key in label_cols:
            if key not in COMPLEX_ATTRS:
                continue
            j = COMPLEX_ATTRS.index(key)
            pos, neg = PHRASE_MAP[key]
            phrases.append(pos if float(labels_pm1[i, j].item()) > 0 else neg)
        if suffix:
            phrases.append(suffix)
        prompts.append(", ".join([x for x in phrases if x]))
    return prompts


def _to_01_labels(labels_pm1: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: (labels_pm1[:, i : i + 1] > 0).float() for i, k in enumerate(COMPLEX_ATTRS)}


def _load_causal_adapter_scm(ca_root: Path, scm_path: Path, device: torch.device):
    ca_root = ca_root.resolve()
    if str(ca_root) not in sys.path:
        sys.path.insert(0, str(ca_root))
    from causal_modules.control_heads import ControlNetConditioningEmbedding  # type: ignore

    graph = torch.tensor(
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.float32,
        device=device,
    )
    head = ControlNetConditioningEmbedding(in_dim=4, hidden_dims=16, dataset_name="celeA_complex").to(device)
    head.load_state_dict(torch.load(scm_path, map_location=device))
    head.update_mask(graph)
    head.eval()
    return head


@torch.no_grad()
def _causal_adapter_intervene_pm1(head, labels_pm1: torch.Tensor, intervention_index: int) -> torch.Tensor:
    labels01 = (labels_pm1 > 0).float()
    intervention_values = 1.0 - labels01[:, intervention_index]
    out, _ = head.inference(
        labels01,
        intervention_indx=intervention_index,
        intervention_values=intervention_values,
    )
    return out.squeeze(2).float() * 2.0 - 1.0


def _binary_pred_labels(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    if np.nanmin(pred) < 0.0 or np.nanmax(pred) > 1.0:
        return pred > 0.0
    return pred > 0.5


def _binary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    p = _binary_pred_labels(pred).reshape(-1).astype(bool)
    t = np.asarray(target).reshape(-1) > 0.5
    tp = np.logical_and(p, t).sum(dtype=np.float64)
    fp = np.logical_and(p, ~t).sum(dtype=np.float64)
    fn = np.logical_and(~p, t).sum(dtype=np.float64)
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0


def _binary_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    p = _binary_pred_labels(pred).reshape(-1).astype(bool)
    t = np.asarray(target).reshape(-1) > 0.5
    return float(np.equal(p, t).mean())


def _binary_macro_f1(pred: np.ndarray, target: np.ndarray) -> float:
    p = _binary_pred_labels(pred).reshape(-1).astype(bool)
    t = np.asarray(target).reshape(-1) > 0.5
    scores = []
    for positive in (False, True):
        pp, tt = p == positive, t == positive
        tp = np.logical_and(pp, tt).sum(dtype=np.float64)
        fp = np.logical_and(pp, ~tt).sum(dtype=np.float64)
        fn = np.logical_and(~pp, tt).sum(dtype=np.float64)
        denom = 2 * tp + fp + fn
        # Only average classes present in either target or prediction.
        if denom > 0:
            scores.append(float(2 * tp / denom))
    return float(np.mean(scores)) if scores else 1.0


def _rate(values: np.ndarray, pred: bool) -> float:
    return float((_binary_pred_labels(values) if pred else (np.asarray(values).reshape(-1) > 0.5)).mean())


@contextmanager
def _benchmark_only_on_path(benchmark_pkg: Path) -> Iterator[None]:
    bench = str(benchmark_pkg.resolve())
    repo = str(REPO.resolve())
    cwd = str(Path.cwd().resolve())
    saved_path = sys.path.copy()
    saved_models = {name: mod for name, mod in sys.modules.items() if name == "models" or name.startswith("models.")}
    filtered = []
    for entry in sys.path:
        resolved = cwd if entry == "" else str(Path(entry).resolve()) if entry else entry
        if resolved in (bench, repo, cwd):
            continue
        filtered.append(entry)
    sys.path = [bench] + filtered
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        sys.modules.update(saved_models)
        sys.path[:] = saved_path


def _load_metric(bench_root: Path, module_name: str, attr_name: str):
    path = bench_root / "evaluation" / "metrics" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"bench_{module_name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return getattr(mod, attr_name)


def _build_predictors(bench_root: Path, device: torch.device):
    cfg_path = bench_root / "methods" / "deepscm" / "configs" / "celeba" / "complex" / "classifier.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    ckpt_dir = Path(cfg["ckpt_path"])
    if not ckpt_dir.is_absolute():
        rel = Path(str(ckpt_dir).lstrip("./"))
        candidates = [bench_root / "methods" / "deepscm" / rel, cfg_path.parent / rel]
        ckpt_dir = next((p for p in candidates if p.exists()), candidates[0])
    with _benchmark_only_on_path(bench_root):
        from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier  # type: ignore

    predictors = {}
    for atr in COMPLEX_ATTRS:
        mod = CelebaComplexClassifier(
            attr=atr,
            context_dim=len(list(cfg["anticausal_graph"][atr])),
            num_outputs=int(cfg["attribute_size"][atr]),
            lr=float(cfg["lr"]),
            version=str(cfg["version"]),
        )
        fname = next(fn for fn in os.listdir(ckpt_dir) if fn.startswith(atr))
        sd = torch.load(ckpt_dir / fname, map_location=device)
        mod.load_state_dict(sd["state_dict"])
        predictors[atr] = mod.to(device).eval()
    return predictors


def _prepare_real64(data_root: Path, filenames, device: torch.device) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.CenterCrop(150),
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    imgs = []
    for fn in filenames:
        imgs.append(tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB")))
    return torch.stack(imgs).to(device)


class PromptLoraEditor:
    def __init__(
        self,
        cfg: Dict[str, Any],
        lora_dir: Path,
        device: torch.device,
        steps: int,
        guidance_scale: float,
        invert_guidance_scale: float = 1.0,
        negative_prompt: str = "",
        scheduler_mode: str = "pretrained",
        vae_latent_mode: str = "sample",
        condition_mode: str = "prompt",
        pai_checkpoint: Path | None = None,
        ptp_controller: str = "simple",
        ptp_local_blend: bool = False,
        ptp_blend_th=(0.3, 0.3),
        ptp_start_blend: float = 0.0,
        ptp_substruct_attr_indices=(),
        capture_ptp_masks: bool = False,
        ptp_mask_steps=(),
    ):
        from diffusers import DDIMScheduler

        self.cfg = cfg
        self.device = device
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.invert_guidance_scale = float(invert_guidance_scale)
        self.vae_latent_mode = str(vae_latent_mode)
        self.condition_mode = str(condition_mode)
        self.ptp_controller = str(ptp_controller)
        self.ptp_local_blend = bool(ptp_local_blend)
        self.ptp_blend_th = tuple(float(x) for x in ptp_blend_th)
        self.ptp_start_blend = float(ptp_start_blend)
        self.ptp_substruct_attr_indices = tuple(int(x) for x in ptp_substruct_attr_indices)
        self.capture_ptp_masks = bool(capture_ptp_masks)
        self.ptp_mask_steps = {int(x) for x in ptp_mask_steps}
        self.last_ptp_masks = {}
        dtype = torch.bfloat16 if str(cfg.get("mixed_precision", "bf16")).lower() == "bf16" else torch.float16
        pipe_cls = _pipeline_class(str(cfg.get("model_family", "sd")))
        self.pipe = pipe_cls.from_pretrained(
            str(cfg["pretrained_model_name_or_path"]),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        self.pipe.load_lora_weights(str(lora_dir))
        if scheduler_mode == "ca":
            self.scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                str(cfg["pretrained_model_name_or_path"]), subfolder="scheduler"
            )
        self.scheduler.set_timesteps(self.steps, device=device)
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()
        self.neg_prompt = str(negative_prompt)
        self.pai = None
        self.pai_prompt = ""
        self.pai_slot_positions = None
        if self.condition_mode in ("pai", "joint_label"):
            if pai_checkpoint is None or not pai_checkpoint.exists():
                raise FileNotFoundError(f"Missing PAI checkpoint: {pai_checkpoint}")
            from models.prompt_aligned_injection import (
                JointLabelTokenInjection,
                PromptAlignedInjection,
                find_slot_positions,
            )

            payload = torch.load(pai_checkpoint, map_location=device)
            metadata = dict(payload["mapper"])
            self.pai_prompt = str(cfg.get("pai_prompt", "a human is @ and * and & and #"))
            slot_tokens = list(cfg.get("pai_slot_tokens", ["@", "*", "&", "#"]))
            mapper_cls = JointLabelTokenInjection if self.condition_mode == "joint_label" else PromptAlignedInjection
            self.pai = mapper_cls(
                attr_names=metadata.get("attr_names", COMPLEX_ATTRS),
                hidden_dim=int(metadata["hidden_dim"]),
                projector_hidden_dim=int(metadata.get("projector_hidden_dim", 256)),
            ).to(device)
            self.pai.load_state_dict(payload["model_state_dict"])
            self.pai.eval()
            self.pai_slot_positions = find_slot_positions(
                self.pipe.tokenizer, self.pai_prompt, slot_tokens
            ).to(device)
        self.image_tfm = transforms.Compose(
            [
                transforms.CenterCrop(150),
                transforms.Resize((int(cfg.get("image_size", 256)), int(cfg.get("image_size", 256))), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _prompt_embeds(self, prompts: List[str]):
        tok = self.pipe.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        neg = self.pipe.tokenizer(
            [self.neg_prompt] * len(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            pe = self.pipe.text_encoder(tok.input_ids.to(self.device))[0]
            ne = self.pipe.text_encoder(neg.input_ids.to(self.device))[0]
        return pe, ne

    def _pai_embeds(self, labels_pm1: torch.Tensor):
        assert self.pai is not None and self.pai_slot_positions is not None
        prompts = [self.pai_prompt] * labels_pm1.shape[0]
        pe, ne = self._prompt_embeds(prompts)
        pe = self.pai(pe, labels_pm1, self.pai_slot_positions)
        return pe, ne

    def condition_embeds(self, labels_pm1: torch.Tensor):
        if self.condition_mode in ("pai", "joint_label"):
            return self._pai_embeds(labels_pm1)
        return self._prompt_embeds(_prompts_from_pm1(labels_pm1, self.cfg))

    def _noise_pred(self, latents, t, prompt_embeds, neg_embeds, guidance_scale=None):
        scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        if scale == 1.0:
            return self.pipe.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        x = torch.cat([latents, latents], dim=0)
        ee = torch.cat([neg_embeds, prompt_embeds], dim=0)
        pred = self.pipe.unet(x, t, encoder_hidden_states=ee).sample
        uncond, text = pred.chunk(2)
        return uncond + scale * (text - uncond)

    def _changed_token_positions(self, src_prompts: List[str], tgt_prompts: List[str]):
        out = []
        pad_id = self.pipe.tokenizer.pad_token_id
        eos_id = self.pipe.tokenizer.eos_token_id
        for src, tgt in zip(src_prompts, tgt_prompts):
            src_ids = self.pipe.tokenizer(
                src,
                padding="max_length",
                truncation=True,
                max_length=self.pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            tgt_ids = self.pipe.tokenizer(
                tgt,
                padding="max_length",
                truncation=True,
                max_length=self.pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            src_seq = [x for x in src_ids.tolist() if x not in (pad_id, eos_id)]
            tgt_seq = [x for x in tgt_ids.tolist() if x not in (pad_id, eos_id)]
            src_pos, tgt_pos = [], []
            for tag, i1, i2, j1, j2 in SequenceMatcher(
                a=src_seq, b=tgt_seq, autojunk=False
            ).get_opcodes():
                if tag == "equal":
                    continue
                src_pos.extend(range(i1, i2))
                tgt_pos.extend(range(j1, j2))
            if not src_pos:
                src_pos = tgt_pos[:]
            if not tgt_pos:
                tgt_pos = src_pos[:]
            out.append((src_pos, tgt_pos))
        return out

    @torch.no_grad()
    def encode_images(self, data_root: Path, filenames) -> torch.Tensor:
        xs = [self.image_tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB")) for fn in filenames]
        x = torch.stack(xs).to(self.device, dtype=self.pipe.vae.dtype)
        latent_dist = self.pipe.vae.encode(x).latent_dist
        lat = (latent_dist.mean if self.vae_latent_mode == "mean" else latent_dist.sample())
        lat = lat * self.pipe.vae.config.scaling_factor
        return lat

    @torch.no_grad()
    def invert(self, latents, prompt_embeds, neg_embeds):
        return self.invert_trajectory(latents, prompt_embeds, neg_embeds)[-1]

    @torch.no_grad()
    def invert_trajectory(self, latents, prompt_embeds, neg_embeds):
        model_dtype = self.pipe.unet.dtype
        sample = latents.to(dtype=model_dtype)
        trajectory = [sample]
        step_size = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        for t in reversed(self.scheduler.timesteps):
            next_t = int(t.item())
            current_t = min(next_t - step_size, self.scheduler.config.num_train_timesteps - 1)
            if current_t >= 0:
                a_t = self.scheduler.alphas_cumprod[current_t].to(sample.device, sample.dtype)
            else:
                a_t = self.scheduler.final_alpha_cumprod.to(sample.device, sample.dtype)
            a_next = self.scheduler.alphas_cumprod[next_t].to(sample.device, sample.dtype)
            eps = self._noise_pred(
                sample,
                t,
                prompt_embeds,
                neg_embeds,
                guidance_scale=self.invert_guidance_scale,
            )
            pred_x0 = (sample - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            sample = (a_next.sqrt() * pred_x0 + (1 - a_next).sqrt() * eps).to(dtype=model_dtype)
            trajectory.append(sample)
        return trajectory

    @torch.no_grad()
    def direct_inversion_residuals(self, trajectory, source_embeds, target_embeds, neg_embeds):
        """CA-style per-step reconstruction offsets for source and target paths."""
        latent_cur = torch.cat([trajectory[-1], trajectory[-1]], dim=0)
        cond = torch.cat([source_embeds, target_embeds], dim=0)
        uncond = torch.cat([neg_embeds, neg_embeds], dim=0)
        residuals = []
        for i, t in enumerate(self.scheduler.timesteps):
            reference_prev = torch.cat([trajectory[-i - 2], trajectory[-i - 2]], dim=0)
            if self.guidance_scale == 1.0:
                eps = self.pipe.unet(latent_cur, t, encoder_hidden_states=cond).sample
            else:
                pred = self.pipe.unet(
                    torch.cat([latent_cur, latent_cur], dim=0),
                    t,
                    encoder_hidden_states=torch.cat([uncond, cond], dim=0),
                ).sample
                eps_uncond, eps_cond = pred.chunk(2)
                eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
            reconstructed_prev = self.scheduler.step(eps, t, latent_cur).prev_sample
            residual = reference_prev - reconstructed_prev
            residuals.append(residual)
            latent_cur = reconstructed_prev + residual
        return residuals

    @torch.no_grad()
    def sample(self, latents_t, prompt_embeds, neg_embeds):
        model_dtype = self.pipe.unet.dtype
        sample = latents_t.to(dtype=model_dtype)
        for t in self.scheduler.timesteps:
            eps = self._noise_pred(sample, t, prompt_embeds, neg_embeds)
            sample = self.scheduler.step(eps, t, sample).prev_sample.to(dtype=model_dtype)
        return sample

    @torch.no_grad()
    def sample_ptp(
        self,
        latents_t,
        source_embeds,
        target_embeds,
        neg_embeds,
        token_positions,
        source_prompts=None,
        target_prompts=None,
        cross_replace_steps: float = 0.2,
        self_replace_steps: float = 0.2,
        direct_residuals=None,
    ):
        use_ca_controller = self.ptp_controller in ("ca", "ca_refine")
        model_dtype = self.pipe.unet.dtype
        if self.ptp_controller == "ca":
            from causal_modules.p2p_edits.attention_control import make_controller

            update_token_index = [list(src_pos) for src_pos, _ in token_positions]
            substruct_token_index = None
            if self.ptp_substruct_attr_indices:
                if self.pai_slot_positions is None:
                    raise ValueError("CA substructure slots currently require PAI conditioning")
                substruct_positions = [
                    int(self.pai_slot_positions[i].item()) for i in self.ptp_substruct_attr_indices
                ]
                substruct_token_index = [substruct_positions] * latents_t.shape[0]
            controller = make_controller(
                pipeline=self.pipe,
                update_token_index=update_token_index,
                is_replace_controller=True,
                cross_replace_steps={"default_": cross_replace_steps},
                self_replace_steps=self_replace_steps,
                blend_words=self.ptp_local_blend,
                blend_params={"start_blend": self.ptp_start_blend, "th": self.ptp_blend_th},
                num_ddim_steps=len(self.scheduler.timesteps),
                substruct_token_index=substruct_token_index,
                device=self.device,
            )
            _cast_ca_controller(controller, device=self.device, dtype=model_dtype)
            if self.capture_ptp_masks and getattr(controller, "local_blend", None) is not None:
                _enable_local_blend_mask_capture(controller.local_blend)
            handles = _register_unet_attention_control_ca(self.pipe.unet, controller)
            controller.reset()
        elif self.ptp_controller == "ca_refine":
            if source_prompts is None or target_prompts is None:
                raise ValueError("CA refinement controller requires source and target prompts")
            controller = _build_ca_refine_controller(
                source_prompts,
                target_prompts,
                token_positions,
                tokenizer=self.pipe.tokenizer,
                num_steps=len(self.scheduler.timesteps),
                cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps,
                local_blend=self.ptp_local_blend,
                blend_th=self.ptp_blend_th,
                start_blend=self.ptp_start_blend,
                device=self.device,
            )
            _cast_ca_controller(controller, device=self.device, dtype=model_dtype)
            if self.capture_ptp_masks and getattr(controller, "local_blend", None) is not None:
                _enable_local_blend_mask_capture(controller.local_blend)
            handles = _register_unet_attention_control_ca(self.pipe.unet, controller)
            controller.reset()
        else:
            controller = _MiniSDP2PController(
                batch_size=latents_t.shape[0],
                token_positions=token_positions,
                num_steps=len(self.scheduler.timesteps),
                cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps,
            )
            handles = _register_unet_attention_control(self.pipe.unet, controller)
        try:
            batch_size = latents_t.shape[0]
            self.last_ptp_masks = {}
            sample = torch.cat([latents_t, latents_t], dim=0).to(dtype=model_dtype)
            text_context = torch.cat([source_embeds, target_embeds], dim=0)
            neg_context = torch.cat([neg_embeds, neg_embeds], dim=0)
            for step_idx, t in enumerate(self.scheduler.timesteps):
                sample = sample.to(dtype=model_dtype)
                if self.guidance_scale == 1.0:
                    eps = self.pipe.unet(sample, t, encoder_hidden_states=text_context).sample
                else:
                    x = torch.cat([sample, sample], dim=0)
                    ee = torch.cat([neg_context, text_context], dim=0)
                    pred = self.pipe.unet(x, t, encoder_hidden_states=ee).sample
                    uncond, text = pred.chunk(2)
                    eps = uncond + self.guidance_scale * (text - uncond)
                sample = self.scheduler.step(eps, t, sample).prev_sample.to(dtype=model_dtype)
                if direct_residuals is not None:
                    residual = direct_residuals[step_idx].to(sample)
                    sample = torch.cat(
                        [
                            sample[:batch_size] + residual[:batch_size],
                            sample[batch_size:],
                        ],
                        dim=0,
                    )
                if use_ca_controller:
                    blend = getattr(controller, "local_blend", None)
                    if blend is not None:
                        blend._captured_primary_mask = None
                        blend._captured_substruct_mask = None
                    sample = controller.step_callback(sample).to(dtype=model_dtype)
                    if (
                        self.capture_ptp_masks
                        and step_idx in self.ptp_mask_steps
                        and blend is not None
                        and blend._captured_primary_mask is not None
                    ):
                        mask = blend._captured_primary_mask
                        if (
                            getattr(blend, "substruct_layers", None) is not None
                            and blend._captured_substruct_mask is not None
                        ):
                            mask = mask & (~blend._captured_substruct_mask)
                        self.last_ptp_masks[step_idx] = mask[batch_size:].float().cpu()
                else:
                    controller.step()
            return sample[batch_size:]
        finally:
            for module, forward in handles:
                module.forward = forward

    @torch.no_grad()
    def decode64(self, latents):
        img = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return F.interpolate(img.float(), size=(64, 64), mode="bicubic", align_corners=False).clamp(0, 1)

    @torch.no_grad()
    def edit(
        self,
        data_root: Path,
        filenames,
        c_orig_pm1,
        c_cf_pm1,
        editing: str = "standard",
        inversion: str = "ddim",
        ptp_cross_replace_steps: float = 0.2,
        ptp_self_replace_steps: float = 0.2,
    ):
        lat0 = self.encode_images(data_root, filenames)
        pe_orig, ne = self.condition_embeds(c_orig_pm1)
        pe_cf, _ = self.condition_embeds(c_cf_pm1)
        trajectory = self.invert_trajectory(lat0, pe_orig, ne)
        lat_t = trajectory[-1]
        direct_residuals = None
        if inversion == "direct":
            if editing != "ptp":
                raise ValueError("DirectInversion is defined here only for two-branch P2P editing")
            direct_residuals = self.direct_inversion_residuals(trajectory, pe_orig, pe_cf, ne)
        if editing == "ptp":
            orig_prompts = None
            cf_prompts = None
            if self.condition_mode in ("pai", "joint_label"):
                assert self.pai_slot_positions is not None
                token_positions = []
                for src, tgt in zip(c_orig_pm1, c_cf_pm1):
                    changed = torch.where((src - tgt).abs() > 1e-6)[0].tolist()
                    positions = [int(self.pai_slot_positions[i].item()) for i in changed]
                    token_positions.append((positions, positions))
            else:
                orig_prompts = _prompts_from_pm1(c_orig_pm1, self.cfg)
                cf_prompts = _prompts_from_pm1(c_cf_pm1, self.cfg)
                token_positions = self._changed_token_positions(orig_prompts, cf_prompts)
            lat_cf = self.sample_ptp(
                lat_t,
                pe_orig,
                pe_cf,
                ne,
                token_positions,
                source_prompts=orig_prompts,
                target_prompts=cf_prompts,
                cross_replace_steps=ptp_cross_replace_steps,
                self_replace_steps=ptp_self_replace_steps,
                direct_residuals=direct_residuals,
            )
        else:
            lat_cf = self.sample(lat_t, pe_cf, ne)
        return self.decode64(lat_cf)


class _MiniSDP2PController:
    def __init__(
        self,
        batch_size: int,
        token_positions,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
    ):
        self.batch_size = int(batch_size)
        self.token_positions = token_positions
        self.cross_until = int(float(cross_replace_steps) * int(num_steps))
        self.self_until = int(float(self_replace_steps) * int(num_steps))
        self.cur_step = 0

    def step(self):
        self.cur_step += 1

    def __call__(self, attn, is_cross: bool):
        b = self.batch_size
        total_branches = 4 if is_cross else 4
        if attn.shape[0] % (total_branches * b) != 0:
            return attn
        h = attn.shape[0] // (total_branches * b)
        view = attn.view(total_branches, b, h, *attn.shape[1:])
        if is_cross and self.cur_step < self.cross_until:
            src_cond = view[2]
            tgt_cond = view[3].clone()
            for i, (src_pos, tgt_pos) in enumerate(self.token_positions):
                if not src_pos or not tgt_pos:
                    continue
                src_idx = torch.tensor(src_pos, device=attn.device, dtype=torch.long)
                tgt_idx = torch.tensor(tgt_pos, device=attn.device, dtype=torch.long)
                src_map = src_cond[i, :, :, src_idx].mean(dim=-1, keepdim=True)
                tgt_cond[i, :, :, tgt_idx] = src_map.expand(-1, -1, len(tgt_pos))
            view[3] = tgt_cond
        elif (not is_cross) and self.cur_step < self.self_until:
            view[1] = view[0]
            view[3] = view[2]
        return view.reshape_as(attn)


def _register_unet_attention_control(unet, controller: _MiniSDP2PController):
    handles = []

    def wrap_forward(module):
        old_forward = module.forward
        to_out = module.to_out[0] if isinstance(module.to_out, torch.nn.ModuleList) else module.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
            is_cross = encoder_hidden_states is not None
            residual = hidden_states
            if getattr(module, "spatial_norm", None) is not None:
                hidden_states = module.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
            sequence_length = hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
            attention_mask_prepared = module.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            if getattr(module, "group_norm", None) is not None:
                hidden_states = module.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = module.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif getattr(module, "norm_cross", False):
                encoder_hidden_states = module.norm_encoder_hidden_states(encoder_hidden_states)
            key = module.to_k(encoder_hidden_states)
            value = module.to_v(encoder_hidden_states)
            query = module.head_to_batch_dim(query)
            key = module.head_to_batch_dim(key)
            value = module.head_to_batch_dim(value)
            attention_probs = module.get_attention_scores(query, key, attention_mask_prepared)
            attention_probs = controller(attention_probs, is_cross)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = module.batch_to_head_dim(hidden_states)
            hidden_states = to_out(hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if getattr(module, "residual_connection", False):
                hidden_states = hidden_states + residual
            return hidden_states / module.rescale_output_factor

        module.forward = forward
        handles.append((module, old_forward))

    for module in unet.modules():
        if module.__class__.__name__ == "Attention" and hasattr(module, "to_q") and hasattr(module, "get_attention_scores"):
            wrap_forward(module)
    return handles


def _register_unet_attention_control_ca(unet, controller):
    handles = []

    def wrap_forward(module, place_in_unet):
        old_forward = module.forward
        to_out = module.to_out[0] if isinstance(module.to_out, torch.nn.ModuleList) else module.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
            is_cross = encoder_hidden_states is not None
            residual = hidden_states
            if getattr(module, "spatial_norm", None) is not None:
                hidden_states = module.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
            sequence_length = hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
            attention_mask_prepared = module.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            if getattr(module, "group_norm", None) is not None:
                hidden_states = module.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = module.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif getattr(module, "norm_cross", False):
                encoder_hidden_states = module.norm_encoder_hidden_states(encoder_hidden_states)
            key = module.to_k(encoder_hidden_states)
            value = module.to_v(encoder_hidden_states)
            query = module.head_to_batch_dim(query)
            key = module.head_to_batch_dim(key)
            value = module.head_to_batch_dim(value)
            attention_probs = module.get_attention_scores(query, key, attention_mask_prepared)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = module.batch_to_head_dim(hidden_states)
            hidden_states = to_out(hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if getattr(module, "residual_connection", False):
                hidden_states = hidden_states + residual
            return hidden_states / module.rescale_output_factor

        module.forward = forward
        handles.append((module, old_forward))

    count = 0
    for block_name, block in unet.named_children():
        if "down" in block_name:
            place = "down"
        elif "up" in block_name:
            place = "up"
        elif "mid" in block_name:
            place = "mid"
        else:
            continue
        for module in block.modules():
            if module.__class__.__name__ == "Attention" and hasattr(module, "to_q"):
                wrap_forward(module, place)
                count += 1
    controller.num_att_layers = count
    return handles


def _cast_ca_controller(controller, device, dtype):
    for name in ("mapper", "alphas", "cross_replace_alpha"):
        value = getattr(controller, name, None)
        if torch.is_tensor(value):
            if name == "mapper" and not torch.is_floating_point(value):
                setattr(controller, name, value.to(device=device))
            else:
                setattr(controller, name, value.to(device=device, dtype=dtype))
    blend = getattr(controller, "local_blend", None)
    if blend is not None:
        for name in ("alpha_layers", "substruct_layers"):
            value = getattr(blend, name, None)
            if torch.is_tensor(value):
                setattr(blend, name, value.to(device=device, dtype=dtype))


def _enable_local_blend_mask_capture(blend):
    if getattr(blend, "_mask_capture_enabled", False):
        return
    original_get_mask = blend.get_mask

    def get_mask(maps, alpha, use_pool):
        mask = original_get_mask(maps, alpha, use_pool)
        if use_pool:
            blend._captured_primary_mask = mask.detach()
        else:
            blend._captured_substruct_mask = mask.detach()
        return mask

    blend.get_mask = get_mask
    blend._mask_capture_enabled = True


def _build_ca_refine_controller(
    source_prompts,
    target_prompts,
    token_positions,
    tokenizer,
    num_steps,
    cross_replace_steps,
    self_replace_steps,
    local_blend,
    blend_th,
    start_blend,
    device,
):
    from causal_modules.p2p_edits import seq_aligner
    from causal_modules.p2p_edits.attention_control import AttentionControlEdit, LocalBlend

    update_token_index = [
        sorted(set(src_pos).union(tgt_pos)) for src_pos, tgt_pos in token_positions
    ]
    blend = None
    if local_blend:
        blend = LocalBlend(
            update_token_index,
            start_blend=float(start_blend),
            th=tuple(float(x) for x in blend_th),
            tokenizer=tokenizer,
            device=device,
            num_ddim_steps=num_steps,
        )

    class BatchedAttentionRefine(AttentionControlEdit):
        def __init__(self):
            super().__init__(
                update_token_index=update_token_index,
                num_steps=num_steps,
                cross_replace_steps={"default_": cross_replace_steps},
                self_replace_steps=self_replace_steps,
                local_blend=blend,
                tokenizer=tokenizer,
                device=device,
            )
            pairs = [
                seq_aligner.get_mapper(src, tgt, tokenizer)
                for src, tgt in zip(source_prompts, target_prompts)
            ]
            self.mapper = torch.stack([pair[0] for pair in pairs]).clamp_min(0).to(device)
            self.alphas = torch.stack([pair[1] for pair in pairs]).to(device).reshape(
                len(pairs), 1, 1, -1
            )

        def replace_cross_attention(self, attn_base, attn_replace):
            batch, heads, queries, words = attn_base.shape
            gather_index = self.mapper[:, None, None, :].expand(batch, heads, queries, words)
            mapped = torch.gather(attn_base, dim=3, index=gather_index)
            alphas = self.alphas.to(dtype=attn_base.dtype)
            return mapped * alphas + attn_replace * (1.0 - alphas)

    return BatchedAttentionRefine()


def _fid(real_batches, fake_batches, device):
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(normalize=True, reset_real_features=False).set_dtype(torch.float32).to(device)
    for x in real_batches:
        metric.update(x.to(device), real=True)
    for x in fake_batches:
        metric.update(x.to(device), real=False)
    return float(metric.compute().detach().cpu())


def _save_source_cf_grid(real_batches, fake_batches, output_path, limit=32):
    """Save interleaved source/counterfactual pairs for quick visual QA."""
    from torchvision.utils import save_image

    real = torch.cat(real_batches, dim=0)[:limit].detach().cpu()
    fake = torch.cat(fake_batches, dim=0)[:limit].detach().cpu()
    paired = torch.stack((real, fake), dim=1).flatten(0, 1)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(paired, out, nrow=8, padding=2)


def main():
    args = parse_args()
    sys.path.insert(0, str(REPO))
    from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset
    from models.full_label_scm import FullLabelSCM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(args.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else REPO / cfg_path
    cfg = _load_yaml(cfg_path)
    run_root = Path(cfg["output_dir"]) / str(cfg.get("run_name"))
    lora_dir = Path(args.lora_dir) if args.lora_dir else run_root / "lora"
    data_root = Path(cfg["data_root"])
    indices = _load_indices(args.selection_json, args.max_samples)
    ds = CelebADataset(root=cfg["data_root"], split=args.split, factor_cols=list(CELEBA_ATTR_ORDER), image_size=64)
    loader = DataLoader(FilenameSubset(ds, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    bench_root = Path(args.benchmark_root)
    effectiveness = _load_metric(bench_root, "effectiveness_SD", "effectiveness")
    with _benchmark_only_on_path(bench_root):
        from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba  # type: ignore

    predictors = _build_predictors(bench_root, device)
    intervention_idx = COMPLEX_ATTRS.index(args.intervention_attr)
    intervention_name = args.intervention_attr
    if args.target_scm == "causal_adapter":
        scm = _load_causal_adapter_scm(Path(args.causal_adapter_root), Path(args.causal_adapter_scm), device)
        target_desc = f"Causal-Adapter controlnet_cond_embedding.inference do({intervention_name} flip)"
    else:
        scm = FullLabelSCM().to(device).eval()
        ck = torch.load(args.scm_checkpoint, map_location=device)
        scm.load_state_dict(ck.get("model_state_dict", ck))
        target_desc = f"full_label_scm.intervene_pm1 do({intervention_name} flip)"
    pai_checkpoint = Path(args.pai_checkpoint) if args.pai_checkpoint else None
    editor = PromptLoraEditor(
        cfg,
        lora_dir,
        device,
        args.steps,
        args.guidance_scale,
        invert_guidance_scale=args.invert_guidance_scale,
        negative_prompt=args.negative_prompt,
        scheduler_mode=args.scheduler_mode,
        vae_latent_mode=args.vae_latent_mode,
        condition_mode=args.condition_mode,
        pai_checkpoint=pai_checkpoint,
        ptp_controller=args.ptp_controller,
        ptp_local_blend=args.ptp_local_blend,
        ptp_blend_th=args.ptp_blend_th,
        ptp_start_blend=args.ptp_start_blend,
        ptp_substruct_attr_indices=args.ptp_substruct_attr_indices,
    )

    preds = {a: [] for a in COMPLEX_ATTRS}
    tgts = {a: [] for a in COMPLEX_ATTRS}
    real_batches = []
    fake_batches = []
    mse_vals = []
    mae_vals = []
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    for batch in tqdm(loader, desc="prompt-lora eval"):
        filenames = batch["filename"]
        c_orig = _labels_pm1(batch, device)
        if args.target_scm == "causal_adapter":
            c_cf = _causal_adapter_intervene_pm1(scm, c_orig, intervention_idx)
        else:
            target_value = -c_orig[:, intervention_idx]
            c_cf = torch.cat(
                [scm.intervene_pm1(c_orig[i : i + 1], intervention_idx, target_value[i].item()) for i in range(c_orig.shape[0])],
                dim=0,
            )
        fake = editor.edit(
            data_root,
            filenames,
            c_orig,
            c_cf,
            editing=args.editing,
            inversion=args.inversion,
            ptp_cross_replace_steps=args.ptp_cross_replace_steps,
            ptp_self_replace_steps=args.ptp_self_replace_steps,
        )
        real = _prepare_real64(data_root, filenames, device)
        cf = _to_01_labels(c_cf)
        cf["image"] = fake
        _, raw = effectiveness(cf, unnormalize_celeba, predictors, "celeba")
        for a in COMPLEX_ATTRS:
            preds[a].append(np.asarray(raw["predictions"][a]))
            tgts[a].append(np.asarray(raw["targets"][a]))
        real_batches.append(real.detach())
        fake_batches.append(fake.detach())
        mse_vals.extend(((real - fake) ** 2).mean(dim=(1, 2, 3)).detach().cpu().numpy().tolist())
        mae_vals.extend((real - fake).abs().mean(dim=(1, 2, 3)).detach().cpu().numpy().tolist())
        lpips_metric.update(fake, real)

    result = {
        "args": vars(args),
        "num_samples": len(indices),
        "indices": indices,
        "eval_protocol": {
            "preprocess": "CenterCrop(150)->Resize(256), eval resize 64",
            "target": target_desc,
            "condition": args.condition_mode,
            "inversion": args.inversion,
            "invert_guidance_scale": args.invert_guidance_scale,
            "generation_guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "scheduler_mode": args.scheduler_mode,
            "vae_latent_mode": args.vae_latent_mode,
            "editing": args.editing,
            "ptp": {
                "cross_replace_steps": args.ptp_cross_replace_steps,
                "self_replace_steps": args.ptp_self_replace_steps,
            }
            if args.editing == "ptp"
            else None,
        },
        "method": f"minisd_{args.condition_mode}_lora_{args.editing}_{args.inversion}",
        "effectiveness": {a: _binary_f1(np.concatenate(preds[a], 0), np.concatenate(tgts[a], 0)) for a in COMPLEX_ATTRS},
        "effectiveness_accuracy": {
            a: _binary_accuracy(np.concatenate(preds[a], 0), np.concatenate(tgts[a], 0))
            for a in COMPLEX_ATTRS
        },
        "effectiveness_macro_f1": {
            a: _binary_macro_f1(np.concatenate(preds[a], 0), np.concatenate(tgts[a], 0))
            for a in COMPLEX_ATTRS
        },
        "effectiveness_debug": {
            a: {
                "target_pos_rate": _rate(np.concatenate(tgts[a], 0), False),
                "pred_pos_rate": _rate(np.concatenate(preds[a], 0), True),
            }
            for a in COMPLEX_ATTRS
        },
        "fid": _fid(real_batches, fake_batches, device),
        "lpips_identity": float(lpips_metric.compute().detach().cpu()),
        "pixel_mae_identity": {"mean": float(np.mean(mae_vals)), "std": float(np.std(mae_vals))},
        "pixel_mse_minimality": {"mean": float(np.mean(mse_vals)), "std": float(np.std(mse_vals))},
        "cld": None,
        "cld_note": "Exact CA CLD requires the unconditional CelebA VAE checkpoint, which is absent from the bundled benchmark.",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    if args.output_grid:
        _save_source_cf_grid(real_batches, fake_batches, args.output_grid)
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
