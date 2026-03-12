#!/usr/bin/env bash
# Train conditional diffusion decoder (requires VAE and latent selection)
set -e
cd "$(dirname "$0")/.."
python -m training.train_diffusion --config configs/diffusion.yaml
