#!/bin/bash
#SBATCH --job-name=ca_curve_1k20k
#SBATCH --nodes=1
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
# Evaluate the final 20k checkpoint on the full fixed n=256 selection.
#SBATCH --array=0-3
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
conda activate base
set -u
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
BASE=lambdalabs/miniSD-diffusers
ROOT=/work/nvme/bhje/hyu29/ckpts/celeba/training_time/ca_20k
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export MPLCONFIGDIR=/work/nvme/bhje/hyu29/cache/matplotlib
cd "${PROJECT}"

STEPS=(20000 20000 20000 20000)
ATTRS=(Young Male No_Beard Bald)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/training_time/ca_20k_bs2_n256/step_${STEP}
mkdir -p "${OUT}"

python -m evaluation.compare_causal_adapter_parent_only_celeba \
  --selection-json /work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_val_balanced_n256_seed42.json \
  --max-samples 256 --batch-size 4 \
  --ca-base-model-path "${BASE}" \
  --ca-controlnet-path "${ROOT}/controlnet-steps-${STEP}.safetensors" \
  --ca-text-embedding-path "${ROOT}/learned_embeds-steps-${STEP}.safetensors" \
  --ca-scm-path /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt \
  --ca-editing standard --ca-steps 50 \
  --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
  --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/metrics_do_${ATTR}.json"
