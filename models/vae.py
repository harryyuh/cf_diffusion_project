from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    in_channels: int = 1
    latent_dim: int = 32
    hidden_dims = (32, 64, 128)
    image_size: int = 28


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
        # Determine the shape by re-encoding a dummy once
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, self.config.image_size, self.config.image_size)
            enc_out = self.encoder(dummy)
        ch, h, w = enc_out.shape[1:]
        x = x.view(z.size(0), ch, h, w)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through VAE.

        Args:
            x: Input images.

        Returns:
            dict with keys: recon, mu, logvar, z.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}


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