#!/bin/bash
#SBATCH --job-name=eval_lora_2way
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=0-7
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_lora_2way.%A_%a.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_lora_2way.%A_%a.err

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
CROSS_REPLACE=(0.1 0.2 0.0 0.0)
SELF_REPLACE=(0.1 0.2 0.0 0.0)
BLEND_TH_1=(0.3 0.3 0.3 0.5)
BLEND_TH_2=(0.3 0.5 0.5 0.3)
METHOD_ID=$((SLURM_ARRAY_TASK_ID / 4))
ATTR_ID=$((SLURM_ARRAY_TASK_ID % 4))
ATTR="${ATTRS[$ATTR_ID]}"

if [[ "${METHOD_ID}" == "0" ]]; then
  METHOD="prompt_lora_standard"
  EDITING="standard"
  PTP_ARGS=()
else
  METHOD="prompt_lora_aligned_ptp"
  EDITING="ptp"
  PTP_ARGS=(
    --ptp-controller ca_refine
    --ptp-local-blend
    --ptp-blend-th "${BLEND_TH_1[$ATTR_ID]}" "${BLEND_TH_2[$ATTR_ID]}"
    --ptp-cross-replace-steps "${CROSS_REPLACE[$ATTR_ID]}"
    --ptp-self-replace-steps "${SELF_REPLACE[$ATTR_ID]}"
  )
fi

MAX_SAMPLES="${MAX_SAMPLES:-256}"
SEL="/scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json"
CFG="configs/latent_lora_celeba_ca_pipeline_four_label_prompt_balanced.yaml"
LORA="/scratch/gilbreth/yu1331/ckpts/celeba/latent_finetune/minisd_ca_pipeline_four_label_prompt_lora_balanced/lora"
ROOT="${OUT_ROOT_OVERRIDE:-/scratch/gilbreth/yu1331/evaluations/celeba/prompt_lora_ca_aligned_two_way_n256}"
OUT_DIR="${ROOT}/${METHOD}/guidance_3p0_invert_1p0"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --config "${CFG}"
  --lora-dir "${LORA}"
  --selection-json "${SEL}"
  --steps 50
  --guidance-scale 3.0
  --invert-guidance-scale 1.0
  --negative-prompt "ugly, blurry, low res, unrealistic"
  --scheduler-mode ca
  --vae-latent-mode mean
  --editing "${EDITING}"
  --condition-mode prompt
  --inversion ddim
  "${PTP_ARGS[@]}"
  --intervention-attr "${ATTR}"
)

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  "${COMMON_ARGS[@]}" \
  --split val \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size 2 \
  --target-scm causal_adapter \
  --output-json "${OUT_DIR}/metrics_do_${ATTR}_n256.json"

if [[ "${SKIP_GRID:-0}" != "1" ]]; then
  python -m evaluation.export_ca_style_prompt_lora_vs_causal_adapter_grid \
    "${COMMON_ARGS[@]}" \
    --out-dir "${OUT_DIR}" \
    --num-each 16 \
    --batch-size 2
fi
