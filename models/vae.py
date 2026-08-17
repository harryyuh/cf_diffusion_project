from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    """
    Forward: identity. Backward: multiply gradient by -scale.
    Used so that when loss_adv = MSE(D(GRL(mu_rest)), A) is minimized,
    the encoder receives gradient that maximizes this MSE (i.e. makes mu_rest bad for predicting A).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.scale, None


@dataclass
class VAEConfig:
    in_channels: int = 1
    latent_dim: int = 32
    hidden_dims: tuple = (32, 64, 128)
    image_size: int = 28
    # First n_parent_dims of z are used to predict parent (e.g. thickness); rest is z_rest. 0 = off.
    n_parent_dims: int = 0
    parent_pred_hidden: int = 32  # hidden size of parent predictor MLP
    # Number of parent regression targets (e.g. 1 for thickness, 2 for SCM t and i).
    parent_pred_dim: int = 1
    # Adversary: D(mu_rest) -> A on encoder means (same as HSIC blocks). If use_adversary=True, GRL on mu_rest.
    use_adversary: bool = False
    adv_hidden: int = 32
    # If True: separate Linear heads for mu/logvar of z1 block vs z_rest (shared CNN trunk).
    dual_branch_encoder: bool = False


class ConvVAE(nn.Module):
    """
    Convolutional Beta-VAE for 28x28 grayscale images.

    Methods:
        encode(x) -> mu, logvar
        reparameterize(mu, logvar) -> z
        decode(z) -> recon
        forward(x) -> dict with recon, mu, logvar, z
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        modules = []
        in_ch = config.in_channels
        for h in config.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = h
        self.encoder = nn.Sequential(*modules)

        # Compute flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.image_size, config.image_size)
            enc_out = self.encoder(dummy)
            self.flatten_dim = enc_out.numel()

        npd = getattr(config, "n_parent_dims", 0) or 0
        self.dual_branch = (
            bool(getattr(config, "dual_branch_encoder", False))
            and npd > 0
            and npd < config.latent_dim
        )
        if self.dual_branch:
            n_rest = config.latent_dim - npd
            self.fc_mu_z1 = nn.Linear(self.flatten_dim, npd)
            self.fc_logvar_z1 = nn.Linear(self.flatten_dim, npd)
            self.fc_mu_rest = nn.Linear(self.flatten_dim, n_rest)
            self.fc_logvar_rest = nn.Linear(self.flatten_dim, n_rest)
            self.fc_mu = None  # unused
            self.fc_logvar = None
        else:
            self.fc_mu = nn.Linear(self.flatten_dim, config.latent_dim)
            self.fc_logvar = nn.Linear(self.flatten_dim, config.latent_dim)
            self.fc_mu_z1 = None  # type: ignore
            self.fc_logvar_z1 = None
            self.fc_mu_rest = None
            self.fc_logvar_rest = None

        # Decoder
        self.decoder_input = nn.Linear(config.latent_dim, self.flatten_dim)

        hidden_dims = list(config.hidden_dims)
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                config.in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

        # Optional: predict parent A from first n_parent_dims of z (regression)
        self.n_parent_dims = getattr(config, "n_parent_dims", 0) or 0
        self.parent_pred_dim = int(getattr(config, "parent_pred_dim", 1) or 1)
        if self.n_parent_dims > 0 and self.n_parent_dims <= config.latent_dim:
            self.parent_predictor = nn.Sequential(
                nn.Linear(self.n_parent_dims, config.parent_pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(config.parent_pred_hidden, self.parent_pred_dim),
            )
            n_rest = config.latent_dim - self.n_parent_dims
            use_adv = getattr(config, "use_adversary", False)
            if use_adv and n_rest > 0:
                self.grl_scale = 1.0
                self.adversary = nn.Sequential(
                    nn.Linear(n_rest, getattr(config, "adv_hidden", 32)),
                    nn.ReLU(inplace=True),
                    nn.Linear(getattr(config, "adv_hidden", 32), self.parent_pred_dim),
                )
            else:
                self.adversary = None
        else:
            self.parent_predictor = None
            self.adversary = None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images into latent distribution parameters.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            (mu, logvar): Each of shape (B, latent_dim).
        """
        enc = self.encoder(x)
        enc = enc.view(x.size(0), -1)
        if self.dual_branch:
            mu1 = self.fc_mu_z1(enc)
            lv1 = self.fc_logvar_z1(enc)
            mu2 = self.fc_mu_rest(enc)
            lv2 = self.fc_logvar_rest(enc)
            mu = torch.cat([mu1, mu2], dim=1)
            logvar = torch.cat([lv1, lv2], dim=1)
        else:
            mu = self.fc_mu(enc)
            logvar = self.fc_logvar(enc)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps.

        Args:
            mu: Mean.
            logvar: Log-variance.

        Returns:
            z: Latent sample.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors into images.

        Args:
            z: Latent tensor of shape (B, latent_dim).

        Returns:
            Reconstructed images of shape (B, 1, 28, 28).
        """
        x = self.decoder_input(z)
        x = x.view(z.size(0), -1, 1, 1)
        # We need to reshape to encoder output shape
        # Determine the shape by re-encoding a dummy once (same device as z)
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.config.in_channels, self.config.image_size, self.config.image_size,
                device=z.device, dtype=z.dtype,
            )
            enc_out = self.encoder(dummy)
        ch, h, w = enc_out.shape[1:]
        x = x.view(z.size(0), ch, h, w)
        x = self.decoder(x)
        x = self.final_layer(x)
        # Decoder may not match exact image_size (e.g. 32x32 vs 28x28); resize to target
        if x.shape[2] != self.config.image_size or x.shape[3] != self.config.image_size:
            x = F.interpolate(
                x, size=(self.config.image_size, self.config.image_size),
                mode="bilinear", align_corners=False,
            )
        return x

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through VAE.

        Args:
            x: Input images.

        Returns:
            dict with keys: recon, mu, logvar, z; if n_parent_dims > 0 also parent_pred from mu_part1,
            z_rest (= sampled z tail, for compat), mu_rest, and z_rest_grl = GRL(mu_rest) for adversary.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        out = {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
        if self.parent_predictor is not None:
            mu_part1 = mu[:, : self.n_parent_dims]
            out["parent_pred"] = self.parent_predictor(mu_part1)
            mu_rest = mu[:, self.n_parent_dims :]
            out["mu_rest"] = mu_rest
            # Legacy key: stochastic tail of z (decode path); unchanged for callers expecting samples.
            out["z_rest"] = z[:, self.n_parent_dims :]
            if self.adversary is not None:
                # RGL on encoder means so objective matches HSIC(mu_z1, mu_rest).
                out["z_rest_grl"] = GradientReversalLayer.apply(mu_rest, self.grl_scale)
        return out


def beta_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_type: str = "bce",
) -> tuple[torch.Tensor, dict]:
    """
    Compute Beta-VAE loss.

    Args:
        recon_x: Reconstruction.
        x: Input.
        mu: Latent mean.
        logvar: Latent log variance.
        beta: KL weight.
        recon_loss_type: 'bce' or 'mse'.

    Returns:
        (loss, components_dict)
    """
    if recon_loss_type.lower() == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    elif recon_loss_type.lower() == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    else:
        raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")

    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon_loss + beta * kl
    return loss, {"recon_loss": recon_loss, "kl": kl}