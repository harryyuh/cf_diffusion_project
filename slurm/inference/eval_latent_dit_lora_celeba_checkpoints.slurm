#!/bin/bash
#SBATCH --job-name=celeba_sd3_eval_sweep
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_latent_dit_lora_celeba_sweep.%j.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_latent_dit_lora_celeba_sweep.%j.err
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
export PROJECT_ROOT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export DIFFUSERS_CACHE=${HF_HOME}/diffusers
export PIP_CACHE_DIR=/scratch/gilbreth/yu1331/cache/pip
export TORCH_HOME=/scratch/gilbreth/yu1331/cache/torch
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}" /scratch/gilbreth/yu1331/cf_diffusion_tmp/logs
cd "${PROJECT_ROOT}"
CFG="${CFG:-configs/latent_dit_lora_celeba_gender_only_2000.yaml}"
RUN_ROOT="${RUN_ROOT:-/scratch/gilbreth/yu1331/ckpts/celeba/latent_finetune/sd3_gender_lora_celeba_2000}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-unknown}"
OUT_ROOT="${OUT_ROOT:-/scratch/gilbreth/yu1331/evaluations/celeba/sd3_gender_lora_2000_job${TRAIN_JOB_ID}}"
SELECTION_JSON="${SELECTION_JSON:-/scratch/gilbreth/yu1331/evaluations/celeba/diffusion_start_t_sweep_job10724386/parent_only/start_t_300/10724386_task3/selection.json}"
STEPS=(000500 001000 001500 002000)
for STEP in "${STEPS[@]}"; do
  LORA_DIR="${RUN_ROOT}/lora_step_${STEP}"
  OUT_DIR="${OUT_ROOT}/step_${STEP}"
  echo "=== Evaluating ${LORA_DIR} -> ${OUT_DIR} ==="
  test -f "${LORA_DIR}/pytorch_lora_weights.safetensors"
  python -m inference.generate_celeba_counterfactual_male_dit_img2img \
    --config "${CFG}" \
    --lora-dir "${LORA_DIR}" \
    --split val \
    --n-each 6 \
    --batch-size 1 \
    --strength 0.45 \
    --steps 30 \
    --guidance-scale 5.0 \
    --lora-scale 0.7 \
    --out-dir "${OUT_DIR}" \
    --selection-json "${SELECTION_JSON}"
done
