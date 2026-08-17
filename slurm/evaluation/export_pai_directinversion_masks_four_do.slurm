#!/bin/bash
#SBATCH --job-name=pai_di_masks
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
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/pai_di_masks.%A_%a.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/pai_di_masks.%A_%a.err

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/slurm/_resolve_root.sh"
elif [[ -f "${SCRIPT_DIR}/../_resolve_root.sh" ]]; then
  source "${SCRIPT_DIR}/../_resolve_root.sh"
else
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
cd "${PROJECT_ROOT}"

ATTRS=(Young Male No_Beard Bald)
CROSS_REPLACE=(0.1 0.2 0.0 0.0)
SELF_REPLACE=(0.1 0.2 0.0 0.0)
BLEND_TH_1=(0.3 0.3 0.3 0.5)
BLEND_TH_2=(0.3 0.5 0.5 0.3)
I="${SLURM_ARRAY_TASK_ID}"
ATTR="${ATTRS[$I]}"
SUBSTRUCT_ARGS=()
if [[ "${ATTR}" == "No_Beard" ]]; then
  SUBSTRUCT_ARGS=(--ptp-substruct-attr-indices 3)
elif [[ "${ATTR}" == "Bald" ]]; then
  SUBSTRUCT_ARGS=(--ptp-substruct-attr-indices 1 2)
fi

PAI_ROOT="/scratch/gilbreth/yu1331/ckpts/celeba/pai_lora/minisd_ca_pipeline_four_label_continuous_pai_lora_balanced/final"
python -m evaluation.export_ca_style_prompt_lora_vs_causal_adapter_grid \
  --config configs/pai_lora_celeba_ca_pipeline_four_label.yaml \
  --lora-dir "${PAI_ROOT}/lora" \
  --pai-checkpoint "${PAI_ROOT}/mapper/pai_mapper.pt" \
  --selection-json /scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json \
  --out-dir /scratch/gilbreth/yu1331/evaluations/celeba/pai_directinversion_ptp_mask_diagnostics \
  --intervention-attr "${ATTR}" \
  --num-each 16 \
  --batch-size 2 \
  --steps 50 \
  --guidance-scale 3.0 \
  --invert-guidance-scale 1.0 \
  --negative-prompt "ugly, blurry, low res, unrealistic" \
  --scheduler-mode ca \
  --vae-latent-mode mean \
  --editing ptp \
  --condition-mode pai \
  --inversion direct \
  --ptp-controller ca \
  --ptp-local-blend \
  --ptp-blend-th "${BLEND_TH_1[$I]}" "${BLEND_TH_2[$I]}" \
  "${SUBSTRUCT_ARGS[@]}" \
  --ptp-cross-replace-steps "${CROSS_REPLACE[$I]}" \
  --ptp-self-replace-steps "${SELF_REPLACE[$I]}" \
  --export-ptp-masks \
  --ptp-mask-steps 0 4 9 24 49 \
  --skip-ca-grid
