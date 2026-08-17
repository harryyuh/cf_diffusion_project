#!/bin/bash
# CelebA Male↔Female counterfactual grids (standalone). Same Python as
# ``run_celeba_cf_grid_after_diffusion_chain.slurm`` but without a Slurm dependency chain.
#
# Submit:
#   cd /home/yu1331/cf-diffusion-celeba \\
#     && DIFFUSION_MODE=parent_only sbatch slurm/inference/run_celeba_counterfactual_male.slurm
#
# Env (match cf-diffusion-project train / ckpt layout):
#   DIFFUSION_MODE=with_vae_z_rest|with_vae_parent_cond|parent_only  (default: with_vae_z_rest)
#   CFG=configs/....yaml   # overrides DIFFUSION_MODE when set
#   DIFF_CKPT START_T N_EACH BATCH SPLIT OUT_DIR STAMP
#   SELECTION_JSON=/path/to/selection.json  # optional; reuse same faces as another run (with VAE vs PO)

#SBATCH --job-name=celeba_cf_male
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=01:30:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=logs/infer_celeba_counterfactual_male.%j.out
#SBATCH --error=logs/infer_celeba_counterfactual_male.%j.err
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
OUT_DIR="${OUT_DIR:-/scratch/gilbreth/yu1331/evaluations/celeba/counterfactual_male_flip_${DIFFUSION_MODE:-custom}_${STAMP}_job${SLURM_JOB_ID}}"

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
