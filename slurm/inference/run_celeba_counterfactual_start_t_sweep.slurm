#!/bin/bash
# CelebA diffusion counterfactual sweep: start_t in {300, 500, 1000} x {WV, PO}.
#
# Submit from repo root (defaults: a30, val split, n_each=16 -> 32 tiles/grid, shared selection):
#   cd /home/yu1331/cf-diffusion-celeba
#   sbatch slurm/inference/run_celeba_counterfactual_start_t_sweep.slurm
#
# Array SLURM_ARRAY_TASK_ID:
#   0-2  with_vae_z_rest   start_t 300, 500, 1000
#   3-5  parent_only       start_t 300, 500, 1000
#
# Optional env: EVAL_ROOT, SELECTION_JSON, N_EACH, BATCH, SPLIT, STAMP,
#   DIFF_CKPT_WV, DIFF_CKPT_PO (or DIFF_CKPT for both)

#SBATCH --job-name=celeba_cf_st
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=02:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=logs/infer_celeba_start_t_sweep.%A_%a.out
#SBATCH --error=logs/infer_celeba_start_t_sweep.%A_%a.err
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

# n_each=16: each grid is 16 (orig|cf) pairs => 32 images per PNG (nrow=2).
N_EACH="${N_EACH:-16}"
BATCH="${BATCH:-4}"
SPLIT="${SPLIT:-val}"
EVAL_ROOT="${EVAL_ROOT:-/scratch/gilbreth/yu1331/evaluations/celeba/diffusion_start_t_sweep}"
SELECTION_JSON="${SELECTION_JSON:-${EVAL_ROOT}/shared_val_n${N_EACH}_selection.json}"
mkdir -p "${EVAL_ROOT}"

if [[ ! -f "${SELECTION_JSON}" ]]; then
  TASK_ID_WAIT="${SLURM_ARRAY_TASK_ID:-0}"
  if [[ "${TASK_ID_WAIT}" -eq 0 ]]; then
    echo "Writing shared selection: ${SELECTION_JSON}"
    python inference/export_celeba_cf_selection.py \
      --config configs/diffusion_celeba_parent_only.yaml \
      --split "${SPLIT}" \
      --n-each "${N_EACH}" \
      --out "${SELECTION_JSON}"
  else
    echo "Waiting for shared selection: ${SELECTION_JSON}"
    for _ in $(seq 1 120); do
      [[ -f "${SELECTION_JSON}" ]] && break
      sleep 5
    done
  fi
fi
if [[ ! -f "${SELECTION_JSON}" ]]; then
  echo "ERROR: shared selection not found: ${SELECTION_JSON}" >&2
  exit 1
fi

START_T_LIST=(300 500 1000)
MODE_LIST=(with_vae_z_rest parent_only)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODE_IDX=$((TASK_ID / 3))
T_IDX=$((TASK_ID % 3))
DIFFUSION_MODE="${MODE_LIST[$MODE_IDX]}"
START_T="${START_T_LIST[$T_IDX]}"

case "${DIFFUSION_MODE}" in
  with_vae_z_rest) CFG="configs/diffusion_celeba.yaml" ;;
  parent_only) CFG="configs/diffusion_celeba_parent_only.yaml" ;;
  *)
    echo "ERROR: unknown DIFFUSION_MODE=${DIFFUSION_MODE}" >&2
    exit 1
    ;;
esac

ARRAY_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
STAMP="${STAMP:-${ARRAY_TAG}}"
OUT_DIR="${EVAL_ROOT}/${DIFFUSION_MODE}/start_t_${START_T}/${STAMP}_task${TASK_ID}"

echo "DIFFUSION_MODE=${DIFFUSION_MODE}"
echo "CFG=${CFG}"
echo "START_T=${START_T}"
echo "SELECTION_JSON=${SELECTION_JSON}"
echo "OUT_DIR=${OUT_DIR}"

EXTRA=(--selection-json "${SELECTION_JSON}")
if [[ "${DIFFUSION_MODE}" == "with_vae_z_rest" && -n "${DIFF_CKPT_WV:-}" ]]; then
  EXTRA+=(--diffusion_checkpoint "${DIFF_CKPT_WV}")
elif [[ "${DIFFUSION_MODE}" == "parent_only" && -n "${DIFF_CKPT_PO:-}" ]]; then
  EXTRA+=(--diffusion_checkpoint "${DIFF_CKPT_PO}")
elif [[ -n "${DIFF_CKPT:-}" ]]; then
  EXTRA+=(--diffusion_checkpoint "${DIFF_CKPT}")
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
