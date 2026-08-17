#!/bin/bash
# Generate CelebA Male<->Female grids with full-label SCM targets + DDIM.

#SBATCH --job-name=celeba_cf_fl
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/infer_celeba_counterfactual_male_full_label.%j.out
#SBATCH --error=logs/infer_celeba_counterfactual_male_full_label.%j.err
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=END,FAIL

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
mkdir -p "${PROJECT_ROOT}/logs"

if ! command -v module >/dev/null 2>&1; then
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
fi
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff

set -euo pipefail
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

CFG="${CFG:-configs/diffusion_celeba_full_label.yaml}"
START_T="${START_T:-300}"
N_EACH="${N_EACH:-16}"
BATCH="${BATCH:-4}"
SPLIT="${SPLIT:-val}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
OUT_DIR="${OUT_DIR:-/scratch/gilbreth/yu1331/evaluations/celeba/full_label_counterfactual_male_${STAMP}_job${SLURM_JOB_ID}}"
SCM_CKPT="${SCM_CKPT:-/scratch/gilbreth/yu1331/ckpts/celeba/full_label_scm/causal_adapter_graph/checkpoints/full_label_scm_best.pt}"

EXTRA=()
if [[ -n "${DIFF_CKPT:-}" ]]; then
  EXTRA+=(--diffusion-checkpoint "${DIFF_CKPT}")
fi
if [[ -n "${SELECTION_JSON:-}" ]]; then
  EXTRA+=(--selection-json "${SELECTION_JSON}")
fi

bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CFG}" configs/full_label_scm_celeba.yaml
python -m inference.generate_celeba_counterfactual_male_full_label \
  --config "${CFG}" \
  "${EXTRA[@]}" \
  --scm-checkpoint "${SCM_CKPT}" \
  --split "${SPLIT}" \
  --n-each "${N_EACH}" \
  --batch-size "${BATCH}" \
  --start-t "${START_T}" \
  --out-dir "${OUT_DIR}"

echo "DONE: full-label grids under ${OUT_DIR}"
