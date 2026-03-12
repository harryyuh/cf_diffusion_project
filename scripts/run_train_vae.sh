#!/usr/bin/env bash
# Train VAE on MorphoMNIST
set -e
cd "$(dirname "$0")/.."
python -m training.train_vae --config configs/vae.yaml
