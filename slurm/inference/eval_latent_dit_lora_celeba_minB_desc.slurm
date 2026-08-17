#!/bin/bash
#SBATCH --job-name=celeba_sd3_desc_eval
#SBATCH --nodes=1
#SBATCH --partition=a10
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_latent_dit_lora_celeba_minB_desc.%j.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_latent_dit_lora_celeba_minB_desc.%j.err
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
CFG="${CFG:-configs/latent_dit_lora_celeba_gender_minB_desc_trial.yaml}"
LORA_DIR="${LORA_DIR:-/scratch/gilbreth/yu1331/ckpts/celeba/latent_finetune/sd3_gender_minB_desc_lora_celeba_trial/lora}"
OUT_DIR="${OUT_DIR:-/scratch/gilbreth/yu1331/evaluations/celeba/sd3_gender_minB_desc_lora_trial_job11004240}"
SELECTION_JSON="${SELECTION_JSON:-/scratch/gilbreth/yu1331/evaluations/celeba/sd3_gender_minB_lora_trial_job10994204/selection.json}"
python -m inference.generate_celeba_counterfactual_male_dit_img2img \
  --config "${CFG}" \
  --lora-dir "${LORA_DIR}" \
  --split val \
  --n-each 16 \
  --batch-size 1 \
  --strength 0.45 \
  --steps 30 \
  --guidance-scale 5.0 \
  --lora-scale 0.7 \
  --out-dir "${OUT_DIR}" \
  --selection-json "${SELECTION_JSON}"
