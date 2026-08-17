#!/bin/bash
# CelebA complex benchmark for official Causal-Adapter.
#
# Required env vars before sbatch:
#   BENCH_ROOT=/home/yu1331/counterfactual-benchmark/counterfactual_benchmark
#   CELEBA_ROOT=/scratch/gilbreth/yu1331/datasets/celebA
#   CA_BASE_MODEL_PATH=<local miniSD folder or HF id>
#   CA_CONTROLNET_PATH=<Causal-Adapter celeba/controlnet folder or safetensors parent>
#   CA_TEXT_EMBEDDING_PATH=<learned_embeds-steps-*.safetensors>
#
# Optional:
#   CA_SCM_PATH=<celeba/scm/best_model.pt>
#   CA_ROOT=/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter
#   CA_METRICS="effectiveness fid minimality"
#   CA_EFFECTIVENESS_PARENTS="Young Male No_Beard Bald"
#   CA_INTERVENTION_ATTRIBUTES="Young Male No_Beard Bald"

#SBATCH --job-name=ca_adapter_bench
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/causal_adapter_celeba_benchmark_%j.out
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=FAIL

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

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-cf_vae_diff}"

set -euo pipefail

: "${BENCH_ROOT:?Set BENCH_ROOT}"
: "${CELEBA_ROOT:?Set CELEBA_ROOT}"
: "${CA_BASE_MODEL_PATH:?Set CA_BASE_MODEL_PATH}"
: "${CA_CONTROLNET_PATH:?Set CA_CONTROLNET_PATH}"
: "${CA_TEXT_EMBEDDING_PATH:?Set CA_TEXT_EMBEDDING_PATH}"

CA_ROOT="${CA_ROOT:-${PROJECT_ROOT}/external/Causal-Adapter}"
CLASSIFIER_JSON="${CLASSIFIER_JSON:-${BENCH_ROOT}/methods/deepscm/configs/celeba/complex/classifier.json}"
OUT_BASE="${SCRATCH:-/scratch/gilbreth/$USER}/evaluations/celeba"
mkdir -p "${OUT_BASE}"
OUT_JSON="${OUT_BASE}/causal_adapter_complex_benchmark_job${SLURM_JOB_ID:-local}.json"

export PYTHONPATH="${CA_ROOT}:${CA_ROOT}/notebook_benchmarks:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if ! python -c "import pytorch_lightning" 2>/dev/null; then
  echo "Installing pytorch-lightning + torchmetrics into ${CONDA_ENV:-cf_vae_diff}..."
  pip install -q "pytorch-lightning==2.5.1" "torchmetrics>=1.0"
fi

if ! python - <<'PY' 2>/dev/null
from diffusers import Causal_ControlNetModel
PY
then
  echo "Installing Causal-Adapter patched diffusers in editable mode..."
  pip install -q -e "${CA_ROOT}/diffusers"
fi

cd "${PROJECT_ROOT}"
python evaluation/run_causal_adapter_celeba_complex_benchmark_metrics.py \
  --causal-adapter-root "${CA_ROOT}" \
  --base-model-path "${CA_BASE_MODEL_PATH}" \
  --controlnet-path "${CA_CONTROLNET_PATH}" \
  --text-embedding-path "${CA_TEXT_EMBEDDING_PATH}" \
  --scm-path "${CA_SCM_PATH:-}" \
  --celeba-root "${CELEBA_ROOT}" \
  --benchmark-root "${BENCH_ROOT}" \
  --classifier-json "${CLASSIFIER_JSON}" \
  --metrics ${CA_METRICS:-composition effectiveness fid minimality} \
  --effectiveness-parents ${CA_EFFECTIVENESS_PARENTS:-Young Male No_Beard Bald} \
  --intervention-attributes ${CA_INTERVENTION_ATTRIBUTES:-Young Male No_Beard Bald} \
  --batch-size "${CA_BATCH_SIZE:-4}" \
  --num-workers "${CA_NUM_WORKERS:-4}" \
  --num-steps "${CA_NUM_STEPS:-50}" \
  --guidance-scale "${CA_GUIDANCE_SCALE:-3.0}" \
  --invert-guidance-scale "${CA_INVERT_GUIDANCE_SCALE:-1.0}" \
  --dtype "${CA_DTYPE:-fp32}" \
  --output-json "${OUT_JSON}"

echo "DONE Causal-Adapter CelebA benchmark -> ${OUT_JSON}"
