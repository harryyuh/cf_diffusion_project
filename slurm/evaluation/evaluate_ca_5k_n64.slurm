#!/bin/bash
#SBATCH --job-name=ca_5k_eval64
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --array=0-3
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
PROJECT=/home/yu1331/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
BASE=/scratch/gilbreth/yu1331/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce
ROOT=/scratch/gilbreth/yu1331/ckpts/celeba/training_time/ca_celeba_complex_5k
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
cd "${PROJECT}"

STEPS=(5000 5000 5000 5000)
ATTRS=(Young Male No_Beard Bald)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
OUT=/scratch/gilbreth/yu1331/evaluations/celeba/training_time/ca_5k_n64/step_${STEP}
mkdir -p "${OUT}"

python -m evaluation.compare_causal_adapter_parent_only_celeba \
  --selection-json /scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json \
  --max-samples 64 --batch-size 4 \
  --ca-base-model-path "${BASE}" \
  --ca-controlnet-path "${ROOT}/controlnet-steps-${STEP}.safetensors" \
  --ca-text-embedding-path "${ROOT}/learned_embeds-steps-${STEP}.safetensors" \
  --ca-scm-path /scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt \
  --ca-editing standard --ca-steps 50 \
  --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
  --intervention-attr "${ATTR}" \
  --output-json "${OUT}/metrics_do_${ATTR}.json"
