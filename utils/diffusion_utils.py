from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    # DDIM stochasticity. 0.0 = deterministic DDIM.
    ddim_eta: float = 0.0


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings as in DDPM/Transformer.
    """
    half_dim = dim // 2
    exponent = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
    exponent = -torch.log(torch.tensor(10000.0, device=timesteps.device)) * exponent / (half_dim - 1)
    emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class GaussianDiffusion:
    """
    Diffusion utilities:
    - Training uses standard forward noising q(x_t|x_0) and epsilon prediction objective.
    - Sampling uses DDIM updates (deterministic when ddim_eta=0).
    """

    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.timesteps = config.timesteps
        self.ddim_eta = float(config.ddim_eta)

        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data: q(x_t | x_0).
        """
        sqrt_ac = self.sqrt_alphas_cumprod.to(x_start.device)[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_start + sqrt_om * noise

    @torch.no_grad()
    def ddim_encode_to_xt(
        self,
        model,
        x0: torch.Tensor,
        end_t: int,
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        DDIM inversion (encode): step from x_0 to x_{end_t} **only** by the recurrence
        (no one-shot from x_0 to x_t). At each step t -> t+1 use model eps_theta(x_t, t).

        Using cumulative alphas bar_alpha_t = alphas_cumprod[t]:

            x_{t+1} = sqrt(bar_alpha_{t+1}/bar_alpha_t) * x_t
                    + ( sqrt(1-bar_alpha_{t+1})
                        - sqrt( (bar_alpha_{t+1}/bar_alpha_t) * (1-bar_alpha_t) ) ) * eps_theta(x_t, t)

        This replaces random `q_sample(x0, end_t, randn)` when use_ddim_inversion is enabled.
        """
        if end_t <= 0:
            return x0
        # Need bar_alpha_{t+1} for t = end_t-1 => last step uses index end_t.
        end_t = min(int(end_t), self.timesteps - 1)
        x = x0
        ac = self.alphas_cumprod.to(device)

        for t in range(0, end_t):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            eps_theta = model(x, t_batch, cond)

            abar_t = ac[t].view(-1, 1, 1, 1)
            abar_tp1 = ac[t + 1].view(-1, 1, 1, 1)
            ratio = abar_tp1 / abar_t.clamp(min=1e-12)
            coef_x = torch.sqrt(ratio)
            inner = ratio * (1.0 - abar_t)
            inner = torch.clamp(inner, min=0.0)
            coef_eps = torch.sqrt(torch.clamp(1.0 - abar_tp1, min=0.0)) - torch.sqrt(inner)
            x = coef_x * x + coef_eps * eps_theta

        return x

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Single reverse diffusion step using DDIM.
        """
        alphas_cumprod_t = self.alphas_cumprod.to(device)[t]  # (B,)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev.to(device)[t]  # (B,)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[t]  # (B,)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t]  # (B,)

        eps_theta = model(x_t, t, cond)  # (B, C, H, W)

        # Predict x0 from epsilon prediction
        pred_x0 = (
            x_t
            - sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * eps_theta
        ) / sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)

        # DDIM sigma_t (eta=0 => deterministic)
        ratio = (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
        ratio = torch.clamp(ratio, min=0.0)
        inner = 1.0 - (alphas_cumprod_t / torch.clamp(alphas_cumprod_prev_t, min=1e-12))
        inner = torch.clamp(inner, min=0.0)
        sigma_t = self.ddim_eta * torch.sqrt(ratio) * torch.sqrt(inner)  # (B,)

        # DDIM update:
        # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0
        #         + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_theta
        #         + sigma_t * z
        coeff_eps = torch.sqrt(
            torch.clamp(1.0 - alphas_cumprod_prev_t - sigma_t**2, min=0.0)
        ).view(-1, 1, 1, 1)
        mean = (
            torch.sqrt(alphas_cumprod_prev_t).view(-1, 1, 1, 1) * pred_x0
            + coeff_eps * eps_theta
        )

        noise = torch.randn_like(x_t)
        x_prev = mean + sigma_t.view(-1, 1, 1, 1) * noise

        # At t=0, directly use x0 prediction.
        t0_mask = (t == 0).view(-1, 1, 1, 1)
        x_prev = torch.where(t0_mask, pred_x0, x_prev)
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Full reverse diffusion loop to sample from noise.
        """
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, cond, device)
        return x

    @torch.no_grad()
    def p_sample_loop_from_xt(
        self,
        model,
        x_t: torch.Tensor,
        start_t: int,
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Reverse diffusion starting from a provided x_t at timestep start_t.
        Useful for identity-preserving edits (img2img-style counterfactuals).
        """
        x = x_t
        for t in reversed(range(start_t + 1)):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, cond, device)
        return x

