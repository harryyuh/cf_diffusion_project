#!/bin/bash
# CelebA latent diffusion LoRA fine-tune (SDXL by default).
# Submit from repo root:
#   cd /home/yu1331/cf-diffusion-celeba && sbatch slurm/train/train_latent_lora_celeba.slurm

#SBATCH --job-name=celeba_sdxl_lora
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/train_latent_lora_celeba.%j.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/train_latent_lora_celeba.%j.err
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh"
elif [[ -f "${SCRIPT_DIR}/../_resolve_root.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/../_resolve_root.sh"
else
  echo "ERROR: cannot find slurm/_resolve_root.sh. Submit from repo root." >&2
  exit 1
fi

module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u

set -euo pipefail
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gilbreth/yu1331/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${HF_HOME}/diffusers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/gilbreth/yu1331/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-/scratch/gilbreth/yu1331/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/gilbreth/yu1331/cache/matplotlib}"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}" /scratch/gilbreth/yu1331/cf_diffusion_tmp/logs
cd "${PROJECT_ROOT}"

CFG="${CFG:-configs/latent_lora_celeba.yaml}"
bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CFG}"
python -m training.train_latent_lora_celeba --config "${CFG}"
