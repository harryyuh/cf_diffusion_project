#!/bin/bash
#SBATCH --job-name=pend_reg_sanity
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"
python pendulum_ca_regressor_sanity.py \
  --ca-root "${PROJECT}/external/Causal-Adapter" \
  --checkpoint-dir "${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/checkpoints/pendulum/trained_classifiers" \
  --output /work/nvme/bhje/hyu29/evals/pendulum/ca_regressor_sanity_real_test.json
