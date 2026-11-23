#!/usr/bin/env bash
#SBATCH --job-name=cel_match_cld
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/matched_50k_bs40_fulltest_ddim100
BENCH=${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark
VAE=/work/nvme/bhje/hyu29/models/counterfactual-benchmark/unconditional_vae/celeba_uncond_image_vae-epoch=44.ckpt
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"
if (( SLURM_ARRAY_TASK_ID == 0 )); then METHOD=ca; else METHOD=label_only; fi
python evaluation/evaluate_celeba_cld_saved.py \
  --saved-root "${ROOT}" --method "${METHOD}" \
  --celeba-root /work/nvme/bhje/hyu29/datasets/celebA \
  --benchmark-root "${BENCH}" --vae-checkpoint "${VAE}" \
  --output "${ROOT}/${METHOD}_cld_by_attribute.json" --batch-size 128
