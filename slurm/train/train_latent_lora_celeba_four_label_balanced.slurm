#!/bin/bash
# CelebA MiniSD LoRA with four causal-label text prompts and rare-attribute balancing.

#SBATCH --job-name=prompt4_bal
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/train_prompt4_balanced.%j.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/train_prompt4_balanced.%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh"
elif [[ -f "${SCRIPT_DIR}/../_resolve_root.sh" ]]; then
  source "${SCRIPT_DIR}/../_resolve_root.sh"
else
  echo "ERROR: cannot find slurm/_resolve_root.sh. Submit from repo root." >&2
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
fi
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/gilbreth/yu1331/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/gilbreth/yu1331/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-/scratch/gilbreth/yu1331/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/gilbreth/yu1331/cache/matplotlib}"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}" /scratch/gilbreth/yu1331/cf_diffusion_tmp/logs
cd "${PROJECT_ROOT}"

CFG="configs/latent_lora_celeba_ca_pipeline_four_label_prompt_balanced.yaml"
bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CFG}"
python -m training.train_latent_lora_celeba --config "${CFG}"
