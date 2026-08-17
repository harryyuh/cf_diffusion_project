#!/bin/bash
#SBATCH --job-name=pixart_ddim
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x.%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u

PROJECT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export DIFFUSERS_CACHE=${HF_HOME}/diffusers
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
mkdir -p /scratch/gilbreth/yu1331/cf_diffusion_tmp/logs "${MPLCONFIGDIR}"
cd "${PROJECT}"

# Avoid SIGPIPE from `head` under `set -o pipefail`.
IMAGE=$(find /scratch/gilbreth/yu1331/datasets/celebA -type f -name '*.jpg' -print -quit)
python dit_ddim_experiment/smoke_pixart_ddim.py \
  --image "${IMAGE}" \
  --output-dir /scratch/gilbreth/yu1331/evaluations/celeba/pixart_ddim_smoke \
  --steps 50
