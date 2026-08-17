#!/bin/bash
# CelebA diffusion — **parent only** (no frozen VAE in cond; UNet sees ``parent_keys`` only).
# Checkpoints: ``${output_dir}/parent_only/checkpoints/`` (see configs/diffusion_celeba_parent_only.yaml).
#
# Submit from repo root:
#   cd /home/yu1331/cf-diffusion-celeba && sbatch slurm/train/train_diffusion_celeba_parent_only.slurm
#
# Chain four 4h jobs (resume ``diffusion_last.pt`` until yaml ``epochs``, default 100):
#   bash slurm/train/submit_celeba_parent_only_chain4.sh
#
# Mirrors cf_diffusion_project: ``slurm/train/train_diffusion_scm3_parent_only.slurm``.

#SBATCH --job-name=celeba_diff_po
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

#SBATCH --output=logs/train_diffusion_celeba_parent_only.%j.out
#SBATCH --error=logs/train_diffusion_celeba_parent_only.%j.err
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

set -euo pipefail
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

CFG="${CFG:-configs/diffusion_celeba_parent_only.yaml}"
bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CFG}"
python -m training.train_diffusion --config "${CFG}"
