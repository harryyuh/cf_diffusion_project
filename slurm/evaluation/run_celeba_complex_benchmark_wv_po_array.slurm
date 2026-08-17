#!/bin/bash
# CelebA complex benchmark (wv + po) for **cf-diffusion-celeba** — array 0=wv, 1=po.
#
# Once:  bash scripts/prepare_celeba_benchmark_data.sh /scratch/gilbreth/yu1331/datasets/celebA
#
# Submit:
#   cd /home/yu1331/cf-diffusion-celeba
#   export BENCH_ROOT=/home/yu1331/counterfactual-benchmark/counterfactual_benchmark
#   export CELEBA_ROOT=/scratch/gilbreth/yu1331/datasets/celebA
#   sbatch slurm/evaluation/run_celeba_complex_benchmark_wv_po_array.slurm

#SBATCH --job-name=celeba_bench
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --output=logs/celeba_complex_benchmark_%A_%a.out
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

module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff

set -euo pipefail
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Benchmark CelebaComplexClassifier imports pytorch_lightning (not in base torch stack).
if ! python -c "import pytorch_lightning" 2>/dev/null; then
  echo "Installing pytorch-lightning + torchmetrics into cf_vae_diff..."
  pip install -q "pytorch-lightning==2.5.1" "torchmetrics>=1.0"
fi

: "${BENCH_ROOT:?Set BENCH_ROOT}"
: "${CELEBA_ROOT:?Set CELEBA_ROOT}"

CFGS=(
  configs/diffusion_celeba.yaml
  configs/diffusion_celeba_parent_only.yaml
)
TAGS=(celeba_wv celeba_po)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
CF_CONFIG="${CFGS[$IDX]}"
TAG="${TAGS[$IDX]}"

CLASSIFIER_JSON="${CLASSIFIER_JSON:-${BENCH_ROOT}/methods/deepscm/configs/celeba/complex/classifier.json}"
OUT_BASE="${SCRATCH:-/scratch/gilbreth/$USER}/evaluations/celeba"
mkdir -p "${OUT_BASE}"
OUT_JSON="${OUT_BASE}/complex_benchmark_${TAG}_job${SLURM_JOB_ID:-local}.json"

cd "${PROJECT_ROOT}"
python evaluation/run_celeba_complex_benchmark_metrics.py \
  --cf-config "${CF_CONFIG}" \
  --celeba-root "${CELEBA_ROOT}" \
  --benchmark-root "${BENCH_ROOT}" \
  --classifier-json "${CLASSIFIER_JSON}" \
  --metrics composition effectiveness fid minimality \
  --effectiveness-parents Male \
  --intervention-attributes Male Young No_Beard Bald \
  --output-json "${OUT_JSON}"

echo "DONE ${TAG} -> ${OUT_JSON}"
