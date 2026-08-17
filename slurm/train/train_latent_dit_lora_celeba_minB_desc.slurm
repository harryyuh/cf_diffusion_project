#!/bin/bash
#SBATCH --job-name=celeba_sd3_desc
#SBATCH --nodes=1
#SBATCH --partition=a10
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/train_latent_dit_lora_celeba_minB_desc.%j.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/train_latent_dit_lora_celeba_minB_desc.%j.err
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
export PROJECT_ROOT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export DIFFUSERS_CACHE=${HF_HOME}/diffusers
export PIP_CACHE_DIR=/scratch/gilbreth/yu1331/cache/pip
export TORCH_HOME=/scratch/gilbreth/yu1331/cache/torch
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}" /scratch/gilbreth/yu1331/cf_diffusion_tmp/logs
cd "${PROJECT_ROOT}"
CFG="${CFG:-configs/latent_dit_lora_celeba_gender_minB_desc_trial.yaml}"
bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CFG}"
python -m training.train_latent_dit_lora_celeba --config "${CFG}"
