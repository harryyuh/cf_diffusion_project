#!/bin/bash
# Generate Causal-Adapter grids on the diffusion_start_t_sweep selection and
# compare Causal-Adapter vs parent-only diffusion with official-style metrics.

#SBATCH --job-name=ca_vs_po_celeba
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/ca_vs_po_celeba.%j.out
#SBATCH --error=logs/ca_vs_po_celeba.%j.err
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

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck source=/dev/null
    source /usr/share/Modules/init/bash
  fi
fi

module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff

set -euo pipefail

CA_ROOT="${CA_ROOT:-${PROJECT_ROOT}/external/Causal-Adapter}"
export PYTHONPATH="${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_XET_CACHE=${HF_HOME}/xet
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export DIFFUSERS_CACHE=${HF_HOME}/diffusers
export PIP_CACHE_DIR=/scratch/gilbreth/yu1331/cache/pip
export TORCH_HOME=/scratch/gilbreth/yu1331/cache/torch
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
mkdir -p "${HF_HOME}" "${HF_XET_CACHE}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

if ! python - <<'PY' 2>/dev/null
from diffusers import Causal_ControlNetModel
PY
then
  echo "Installing Causal-Adapter patched diffusers in editable mode..."
  pip install -q -e "${CA_ROOT}/diffusers"
fi

if ! python - <<'PY' 2>/dev/null
import cv2
PY
then
  echo "Installing opencv-python-headless for Causal-Adapter p2p utilities..."
  pip install -q opencv-python-headless
fi

if ! python - <<'PY' 2>/dev/null
import IPython
PY
then
  echo "Installing IPython for Causal-Adapter notebook utilities..."
  pip install -q IPython
fi

if ! python - <<'PY' 2>/dev/null
import torch_fidelity
PY
then
  echo "Installing torch-fidelity for FID metric..."
  pip install -q torch-fidelity
fi

cd "${PROJECT_ROOT}"

SELECTION_JSON="${SELECTION_JSON:-/scratch/gilbreth/yu1331/evaluations/celeba/diffusion_start_t_sweep/shared_val_n16_selection.json}"
METRICS_SELECTION_JSON="${METRICS_SELECTION_JSON-${SELECTION_JSON}}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
STAMP="${SLURM_JOB_ID:-local}"
OUT_ROOT="${OUT_ROOT:-/scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_job${STAMP}}"
CA_GRID_DIR="${OUT_ROOT}/causal_adapter_grids"
METRICS_JSON="${OUT_ROOT}/metrics_do_male_n${MAX_SAMPLES}.json"
mkdir -p "${OUT_ROOT}"

if [[ "${RUN_GRIDS:-1}" == "0" ]]; then
  echo "Skipping Causal-Adapter display grids because RUN_GRIDS=0"
elif [[ -f "${CA_GRID_DIR}/grid_male_to_female.png" && -f "${CA_GRID_DIR}/grid_female_to_male.png" ]]; then
  echo "Reusing existing Causal-Adapter grids under ${CA_GRID_DIR}"
else
  python -m inference.generate_causal_adapter_celeba_counterfactual_male \
    --selection-json "${SELECTION_JSON}" \
    --out-dir "${CA_GRID_DIR}" \
    --editing "${CA_EDITING:-p2p}" \
    --num-steps "${CA_STEPS:-50}" \
    --guidance-scale "${CA_GUIDANCE_SCALE:-3.0}" \
    --invert-guidance-scale "${CA_INVERT_GUIDANCE_SCALE:-1.0}"
fi

METRICS_SELECTION_ARGS=()
if [[ -n "${METRICS_SELECTION_JSON}" && "${METRICS_SELECTION_JSON}" != "NONE" ]]; then
  METRICS_SELECTION_ARGS=(--selection-json "${METRICS_SELECTION_JSON}")
fi

python -m evaluation.compare_causal_adapter_parent_only_celeba \
  "${METRICS_SELECTION_ARGS[@]}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH:-4}" \
  --parent-start-t "${PARENT_START_T:-300}" \
  --ca-editing "${CA_EDITING:-p2p}" \
  --ca-steps "${CA_STEPS:-50}" \
  --ca-guidance-scale "${CA_GUIDANCE_SCALE:-3.0}" \
  --ca-invert-guidance-scale "${CA_INVERT_GUIDANCE_SCALE:-1.0}" \
  --output-json "${METRICS_JSON}"

echo "DONE"
echo "Causal-Adapter grids: ${CA_GRID_DIR}"
echo "Metrics: ${METRICS_JSON}"
