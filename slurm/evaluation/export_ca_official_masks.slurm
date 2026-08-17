#!/bin/bash
#SBATCH --job-name=ca_masks
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=0-3
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/ca_masks.%A_%a.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/ca_masks.%A_%a.err

set -eo pipefail
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
export PYTHONPATH="/home/yu1331/cf-diffusion-celeba:${PYTHONPATH:-}"
export HF_HOME="/scratch/gilbreth/yu1331/huggingface"
export TORCH_HOME="/scratch/gilbreth/yu1331/cache/torch"
export MPLCONFIGDIR="/scratch/gilbreth/yu1331/cache/matplotlib"

ATTRS=(Young Male No_Beard Bald)
ATTR="${ATTRS[$SLURM_ARRAY_TASK_ID]}"
python /home/yu1331/cf-diffusion-celeba/evaluation/export_ca_official_masks.py \
  --attr "${ATTR}" \
  --selection-json /scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json \
  --out-root /scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_mask_diagnostics \
  --ca-root /home/yu1331/cf-diffusion-celeba/external/Causal-Adapter \
  --data-root /scratch/gilbreth/yu1331/datasets/celebA \
  --controlnet-path /scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/controlnet-steps-200000.safetensors \
  --text-embedding-path /scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/learned_embeds-steps-200000.safetensors \
  --scm-path /scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt
