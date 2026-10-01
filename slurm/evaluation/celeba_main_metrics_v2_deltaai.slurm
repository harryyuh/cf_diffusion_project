#!/bin/bash
#SBATCH --job-name=celeba_main_v2
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100_semantics_v2
OLD_ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100
BENCH=${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark
VAE=/work/nvme/bhje/hyu29/models/counterfactual-benchmark/unconditional_vae/celeba_uncond_image_vae-epoch=44.ckpt

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python evaluation/merge_celeba_ca_protocol_fulltest.py \
      --root "${ROOT}" --manifest "${OLD_ROOT}/plans/manifest.json" \
      --output "${ROOT}/merged_metrics_fid_f1_lpips.json"
    ;;
  1)
    python evaluation/evaluate_celeba_cld_saved.py \
      --saved-root "${ROOT}" --method ca \
      --celeba-root /work/nvme/bhje/hyu29/datasets/celebA \
      --benchmark-root "${BENCH}" --vae-checkpoint "${VAE}" \
      --output "${ROOT}/ca_cld.json" --batch-size 128
    ;;
  2)
    python evaluation/evaluate_celeba_cld_saved.py \
      --saved-root "${ROOT}" --method label_only \
      --celeba-root /work/nvme/bhje/hyu29/datasets/celebA \
      --benchmark-root "${BENCH}" --vae-checkpoint "${VAE}" \
      --output "${ROOT}/label_only_cld.json" --batch-size 128
    ;;
esac
