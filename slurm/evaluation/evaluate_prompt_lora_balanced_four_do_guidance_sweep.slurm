#!/bin/bash
#SBATCH --job-name=eval_plora_sweep
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=0-15
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_plora_sweep.%A_%a.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_plora_sweep.%A_%a.err

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh"
elif [[ -f "${SCRIPT_DIR}/../_resolve_root.sh" ]]; then
  source "${SCRIPT_DIR}/../_resolve_root.sh"
else
  echo "ERROR: cannot find slurm/_resolve_root.sh. Submit from repo root." >&2
  exit 1
fi
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/gilbreth/yu1331/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export TORCH_HOME="/scratch/gilbreth/yu1331/cache/torch"
export MPLCONFIGDIR="/scratch/gilbreth/yu1331/cache/matplotlib"
mkdir -p /scratch/gilbreth/yu1331/cf_diffusion_tmp/logs "${HF_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

ATTRS=(Young Male No_Beard Bald)
GUIDES=(1.0 2.0 3.0 5.0)
ATTR="${ATTRS[$((SLURM_ARRAY_TASK_ID % 4))]}"
GUIDE="${GUIDES[$((SLURM_ARRAY_TASK_ID / 4))]}"
GUIDE_TAG="${GUIDE/./p}"
CFG="configs/latent_lora_celeba_ca_pipeline_four_label_prompt_balanced.yaml"
LORA="/scratch/gilbreth/yu1331/ckpts/celeba/latent_finetune/minisd_ca_pipeline_four_label_prompt_lora_balanced/lora"
SEL="/scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json"
OUT_DIR="/scratch/gilbreth/yu1331/evaluations/celeba/minisd_prompt_lora_balanced_n256_ca_scm_targets_four_do_guidance_sweep/guidance_${GUIDE_TAG}"
mkdir -p "${OUT_DIR}"

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config "$CFG" \
  --lora-dir "$LORA" \
  --selection-json "$SEL" \
  --split val \
  --max-samples 256 \
  --batch-size 4 \
  --steps 50 \
  --guidance-scale "$GUIDE" \
  --target-scm causal_adapter \
  --intervention-attr "$ATTR" \
  --output-json "${OUT_DIR}/metrics_do_${ATTR}_n256.json"
