#!/bin/bash
#SBATCH --job-name=pixart_ddim
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
conda activate base
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
set -u

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export DIFFUSERS_CACHE=${HF_HOME}/diffusers
export MPLCONFIGDIR=/work/nvme/bhje/hyu29/cache/matplotlib
mkdir -p /work/nvme/bhje/hyu29/logs "${MPLCONFIGDIR}"
cd "${PROJECT}"

# Avoid SIGPIPE from `head` under `set -o pipefail`.
IMAGE=$(find /work/nvme/bhje/hyu29/datasets/celebA -type f -name '*.jpg' -print -quit)
python dit_ddim_experiment/smoke_pixart_ddim.py \
  --image "${IMAGE}" \
  --output-dir /work/nvme/bhje/hyu29/evaluations/celeba/pixart_ddim_smoke \
  --steps 50
