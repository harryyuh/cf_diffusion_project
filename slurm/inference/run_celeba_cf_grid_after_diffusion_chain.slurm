#!/bin/bash
# Generate CelebA Male↔Female counterfactual image grids AFTER diffusion training chain ends.
#
# Uses ``inference/generate_celeba_counterfactual_male.py`` + ``diffusion_best.pt`` under
#   ``${output_dir}/<subdir>/checkpoints/`` where ``<subdir>`` matches training (see ``DIFFUSION_MODE``).
#
# Submit chained to the LAST training job (example LAST=10693499):
#   cd /home/yu1331/cf-diffusion-celeba \
#     && sbatch --dependency=afterok:${LAST} slurm/inference/run_celeba_cf_grid_after_diffusion_chain.slurm
#
# Optional env:
#   DIFFUSION_MODE=with_vae_z_rest|with_vae_parent_cond|parent_only  (default: with_vae_z_rest; ignored if CFG set)
#   CFG=configs/diffusion_celeba.yaml   # explicit diffusion yaml (overrides DIFFUSION_MODE)
#   DIFF_CKPT=... START_T=300 N_EACH=8 BATCH=4 SPLIT=val OUT_DIR=...
#   SELECTION_JSON=/path/to/selection.json  # optional; same images across modes

#SBATCH --job-name=celeba_cf_eval
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=01:30:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=logs/infer_celeba_cf_grid_after_train.%j.out
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
  echo "ERROR: cannot find slurm/_resolve_root.sh; submit from cf-diffusion-celeba root." >&2
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

if [[ -z "${CFG:-}" ]]; then
  DIFFUSION_MODE="${DIFFUSION_MODE:-with_vae_z_rest}"
  case "${DIFFUSION_MODE}" in
    with_vae_z_rest) CFG="configs/diffusion_celeba.yaml" ;;
    with_vae_parent_cond) CFG="configs/diffusion_celeba_with_vae_parent_cond_only.yaml" ;;
    parent_only) CFG="configs/diffusion_celeba_parent_only.yaml" ;;
    *)
      echo "ERROR: unknown DIFFUSION_MODE=${DIFFUSION_MODE}" >&2
      exit 1
      ;;
  esac
else
  CFG="${CFG}"
fi
echo "DIFFUSION_MODE=${DIFFUSION_MODE:-}"
echo "CFG=${CFG}"
START_T="${START_T:-300}"
N_EACH="${N_EACH:-6}"
BATCH="${BATCH:-4}"
SPLIT="${SPLIT:-val}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
OUT_DIR="${OUT_DIR:-/scratch/gilbreth/yu1331/evaluations/celeba/cf_male_flip_after_train_${STAMP}_job${SLURM_JOB_ID}}"

EXTRA=()
if [[ -n "${DIFF_CKPT:-}" ]]; then
  EXTRA+=(--diffusion_checkpoint "${DIFF_CKPT}")
fi
if [[ -n "${SELECTION_JSON:-}" ]]; then
  EXTRA+=(--selection-json "${SELECTION_JSON}")
fi

bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CFG}" configs/vae_celeba.yaml

python inference/generate_celeba_counterfactual_male.py \
  --config "${CFG}" \
  "${EXTRA[@]}" \
  --split "${SPLIT}" \
  --n-each "${N_EACH}" \
  --batch-size "${BATCH}" \
  --start-t "${START_T}" \
  --out-dir "${OUT_DIR}"

echo "DONE: grids under ${OUT_DIR}"
