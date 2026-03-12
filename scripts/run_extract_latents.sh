#!/usr/bin/env bash
# Extract latents from train set using frozen VAE encoder
set -e
cd "$(dirname "$0")/.."
python -m analysis.extract_latents --config configs/logistic.yaml
