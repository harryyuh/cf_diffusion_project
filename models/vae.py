from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    """
    Forward: identity. Backward: multiply gradient by -scale.
    Used so that when loss_adv = MSE(D(GRL(z_rest)), A) is minimized,
    the encoder receives gradient that maximizes this MSE (i.e. makes z_rest bad for predicting A).
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
    # Adversary: D(z_rest) -> A. If use_adversary=True, encoder gets reversed grad so z_rest cannot predict A.
    use_adversary: bool = False
    adv_hidden: int = 32


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

        self.fc_mu = nn.Linear(self.flatten_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, config.latent_dim)

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
        if self.n_parent_dims > 0 and self.n_parent_dims <= config.latent_dim:
            self.parent_predictor = nn.Sequential(
                nn.Linear(self.n_parent_dims, config.parent_pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(config.parent_pred_hidden, 1),
            )
            n_rest = config.latent_dim - self.n_parent_dims
            use_adv = getattr(config, "use_adversary", False)
            if use_adv and n_rest > 0:
                self.grl_scale = 1.0
                self.adversary = nn.Sequential(
                    nn.Linear(n_rest, getattr(config, "adv_hidden", 32)),
                    nn.ReLU(inplace=True),
                    nn.Linear(getattr(config, "adv_hidden", 32), 1),
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
            dict with keys: recon, mu, logvar, z; if n_parent_dims > 0 also parent_pred, z_rest.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        out = {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
        if self.parent_predictor is not None:
            z_part1 = z[:, : self.n_parent_dims]
            out["parent_pred"] = self.parent_predictor(z_part1)
            z_rest = z[:, self.n_parent_dims :]
            out["z_rest"] = z_rest
            if self.adversary is not None:
                # GRL in forward: identity; backward will reverse grad so encoder gets "maximize adv loss"
                out["z_rest_grl"] = GradientReversalLayer.apply(z_rest, self.grl_scale)
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