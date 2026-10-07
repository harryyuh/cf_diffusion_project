#!/usr/bin/env bash
#SBATCH --job-name=cel_match_merge
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/matched_50k_bs40_fulltest_ddim100
PLAN_ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100/plans
cd "${PROJECT}"
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
python evaluation/merge_celeba_ca_protocol_fulltest.py \
  --root "${ROOT}" --manifest "${PLAN_ROOT}/manifest.json" \
  --output "${ROOT}/merged_metrics_fid_f1_lpips.json"
