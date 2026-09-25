#!/bin/bash
#SBATCH --job-name=pend_ca_reg
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
cd "${BENCH}/methods/deepscm"

python -c 'import torch, pytorch_lightning; print(torch.__version__, pytorch_lightning.__version__)'
python train_classifier.py --classifier-config configs/pendulum/classifier.json
