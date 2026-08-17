#!/bin/bash
# CelebA **complex** (Young, Male, No_Beard, Bald) benchmark metrics for cf-diffusion-celeba
# (composition, effectiveness, FID, minimality) using pretrained benchmark classifiers.
#
# Submit (set paths for your cluster):
#   cd /home/yu1331/cf-diffusion-celeba
#   DIFFUSION_MODE=parent_only \
#   BENCH_ROOT=/path/to/counterfactual-benchmark/counterfactual_benchmark \
#   CELEBA_ROOT=/scratch/gilbreth/yu1331/datasets/celebA \
#   sbatch slurm/evaluation/run_celeba_complex_benchmark_metrics.slurm
#
# Optional env:
#   DIFFUSION_MODE — picks default ``CF_CONFIG`` if ``CF_CONFIG`` is unset (match train slurm / ckpt subdir):
#     with_vae_z_rest       -> configs/diffusion_celeba.yaml
#     with_vae_parent_cond  -> configs/diffusion_celeba_with_vae_parent_cond_only.yaml
#     parent_only           -> configs/diffusion_celeba_parent_only.yaml
#   CF_CONFIG=...           # explicit override (highest priority)
#   CLASSIFIER_JSON=.../methods/deepscm/configs/celeba/complex/classifier.json
#   METRICS_OVERRIDE="composition effectiveness"
#   EMBEDDINGS=   # empty = pixels; or vgg, clip, lpips, vae, clfs

#SBATCH --job-name=celeba_bench
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

#SBATCH --output=logs/eval_celeba_complex_benchmark.%j.out
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

if ! python -c "import pytorch_lightning" 2>/dev/null; then
  echo "Installing pytorch-lightning + torchmetrics into cf_vae_diff..."
  pip install -q "pytorch-lightning==2.5.1" "torchmetrics>=1.0"
fi

: "${CELEBA_ROOT:?Set CELEBA_ROOT to CelebA dataset root (img_align_celeba + list files)}"
: "${BENCH_ROOT:?Set BENCH_ROOT to counterfactual_benchmark package dir (contains datasets/, evaluation/)}"

if [[ -z "${CF_CONFIG:-}" ]]; then
  DIFFUSION_MODE="${DIFFUSION_MODE:-with_vae_z_rest}"
  case "${DIFFUSION_MODE}" in
    with_vae_z_rest) CF_CONFIG="configs/diffusion_celeba.yaml" ;;
    with_vae_parent_cond) CF_CONFIG="configs/diffusion_celeba_with_vae_parent_cond_only.yaml" ;;
    parent_only) CF_CONFIG="configs/diffusion_celeba_parent_only.yaml" ;;
    *)
      echo "ERROR: unknown DIFFUSION_MODE=${DIFFUSION_MODE} (use with_vae_z_rest|with_vae_parent_cond|parent_only or set CF_CONFIG)" >&2
      exit 1
      ;;
  esac
else
  CF_CONFIG="${CF_CONFIG}"
fi
echo "DIFFUSION_MODE=${DIFFUSION_MODE:-}"
echo "CF_CONFIG=${CF_CONFIG}"
CLASSIFIER_JSON="${CLASSIFIER_JSON:-}"
OUT_JSON="${OUT_JSON:-${SCRATCH:-/scratch/gilbreth/$USER}/evaluations/celeba/complex_benchmark_job${SLURM_JOB_ID:-local}.json}"

EXTRA_EMB=()
if [[ -n "${EMBEDDINGS:-}" ]]; then
  EXTRA_EMB=(--embeddings "${EMBEDDINGS}")
fi

CLASSIFIER_ARG=()
if [[ -n "${CLASSIFIER_JSON}" ]]; then
  CLASSIFIER_ARG=(--classifier-json "${CLASSIFIER_JSON}")
fi

BENCH_METRICS=(--metrics composition effectiveness fid minimality)
if [[ -n "${METRICS_OVERRIDE:-}" ]]; then
  # shellcheck disable=2206
  BENCH_METRICS=(--metrics ${METRICS_OVERRIDE})
fi

bash "${PROJECT_ROOT}/slurm/print_config_fingerprint.sh" "${CF_CONFIG}" configs/vae_celeba.yaml

python "${PROJECT_ROOT}/evaluation/run_celeba_complex_benchmark_metrics.py" \
  --cf-config "${CF_CONFIG}" \
  --celeba-root "${CELEBA_ROOT}" \
  --benchmark-root "${BENCH_ROOT}" \
  "${CLASSIFIER_ARG[@]}" \
  "${BENCH_METRICS[@]}" \
  "${EXTRA_EMB[@]}" \
  --output-json "${OUT_JSON}"

echo "DONE: ${OUT_JSON}"
