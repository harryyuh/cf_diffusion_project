#!/bin/bash
# CelebA diffusion — **with VAE**: UNet cond = concat(z_rest, parent) (default CelebA setup).
# Checkpoints: ``.../with_vae_condition/checkpoints/`` (configs/diffusion_celeba.yaml).
#
# Submit:
#   cd /home/yu1331/cf-diffusion-celeba && sbatch slurm/train/train_diffusion_celeba_with_vae_condition.slurm
#
# Optional: CFG=configs/diffusion_celeba.yaml

#SBATCH --job-name=celeba_diff_vae
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

#SBATCH --output=logs/train_diffusion_celeba_with_vae.%j.out
#SBATCH --error=logs/train_diffusion_celeba_with_vae.%j.err
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
  echo "ERROR: cannot find slurm/_resolve_root.sh" >&2
  exit 1
fi
mkdir -p "${PROJECT_ROOT}/logs"

module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff

set -euo pipefail
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

CFG="${CFG:-configs/diffusion_celeba.yaml}"
bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CFG}" configs/vae_celeba.yaml
python -m training.train_diffusion --config "${CFG}"
