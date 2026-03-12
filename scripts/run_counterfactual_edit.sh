#!/usr/bin/env bash
# Run counterfactual editing (set cond_input_dim in configs/inference.yaml to n_rest_dims+1)
set -e
cd "$(dirname "$0")/.."
python -m inference.counterfactual_edit --config configs/inference.yaml
