#!/bin/bash
# Generate a few samples from the CelebA latent LoRA trial checkpoint.

#SBATCH --job-name=celeba_lora_samples
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=01:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/generate_latent_lora_celeba.%j.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/generate_latent_lora_celeba.%j.err
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=END,FAIL

set -eo pipefail

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
mkdir -p "${PROJECT_ROOT}/logs"

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

CFG="${CFG:-configs/latent_lora_celeba_a30_trial.yaml}"
python -m inference.generate_latent_lora_celeba \
  --config "${CFG}" \
  --num-images-per-prompt 3 \
  --steps 30 \
  --guidance-scale 6.0 \
  --prompt "a high quality portrait photo of a smiling woman, realistic skin texture, sharp focus" \
  --prompt "a high quality portrait photo of a smiling man, black hair, realistic skin texture, sharp focus" \
  --prompt "a high quality portrait photo of a young woman wearing eyeglasses, realistic skin texture, sharp focus"
