#!/usr/bin/env bash
# Fit logistic regression and select father/rest latent dimensions
set -e
cd "$(dirname "$0")/.."
python -m analysis.fit_logistic_regression --config configs/logistic.yaml
python -m analysis.select_latent_dims --config configs/logistic.yaml
