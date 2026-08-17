#!/bin/bash
# Prepare local paths for the official Causal-Adapter benchmark on Gilbreth.

set -euo pipefail

CA_ROOT="${CA_ROOT:-/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter}"
OFFICIAL_BENCH="${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark"
USER_BENCH="${USER_BENCH:-/home/yu1331/counterfactual-benchmark/counterfactual_benchmark}"
CA_CKPT_ROOT="${CA_CKPT_ROOT:-/scratch/gilbreth/yu1331/models/Causal-Adapter}"

mkdir -p "${OFFICIAL_BENCH}/methods/deepscm/checkpoints/celeba/complex"

if [[ -d "${USER_BENCH}/methods/deepscm/checkpoints/celeba/complex/trained_classifiers" ]]; then
  ln -sfn \
    "${USER_BENCH}/methods/deepscm/checkpoints/celeba/complex/trained_classifiers" \
    "${OFFICIAL_BENCH}/methods/deepscm/checkpoints/celeba/complex/trained_classifiers"
fi

cat <<EOF
Official benchmark root:
  ${OFFICIAL_BENCH}

Causal-Adapter CelebA checkpoint paths:
  controlnet: ${CA_CKPT_ROOT}/celeba/controlnet/controlnet-steps-200000.safetensors
  embeddings: ${CA_CKPT_ROOT}/celeba/controlnet/learned_embeds-steps-200000.safetensors
  causalnet/scm head: ${CA_CKPT_ROOT}/celeba/scm/best_model.pt

Classifier checkpoint symlink:
  ${OFFICIAL_BENCH}/methods/deepscm/checkpoints/celeba/complex/trained_classifiers

Still required for evaluate_SD_DSCM.py:
  ${OFFICIAL_BENCH}/methods/deepscm/checkpoints/celeba/complex/trained_scm
EOF
