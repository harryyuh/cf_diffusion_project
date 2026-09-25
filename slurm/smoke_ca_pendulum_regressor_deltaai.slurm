#!/bin/bash
#SBATCH --job-name=pend_ca_smoke
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
#SBATCH --error=/work/nvme/bhje/hyu29/logs/%x.%j.err

set -eo pipefail
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
set -ux

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
BENCH=${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark
export PYTHONPATH="${PROJECT}/external/Causal-Adapter:${BENCH}:${PYTHONPATH:-}"
python "${PROJECT}/smoke_ca_pendulum_regressor_deltaai.py"
