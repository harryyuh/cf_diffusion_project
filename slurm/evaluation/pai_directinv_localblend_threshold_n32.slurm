#!/bin/bash
#SBATCH --job-name=pai_dilb32
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=0-5
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/pai_dilb32.%A_%a.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/pai_dilb32.%A_%a.err

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
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export TORCH_HOME="/scratch/gilbreth/yu1331/cache/torch"
export MPLCONFIGDIR="/scratch/gilbreth/yu1331/cache/matplotlib"

ATTRS=(No_Beard Bald)
THRESHOLDS=(0.30 0.50 0.70)
ATTR_ID=$((SLURM_ARRAY_TASK_ID / 3))
TH_ID=$((SLURM_ARRAY_TASK_ID % 3))
ATTR="${ATTRS[$ATTR_ID]}"
TH="${THRESHOLDS[$TH_ID]}"
TH_TAG="${TH/./p}"

REPLACEMENT_ROOT="/scratch/gilbreth/yu1331/evaluations/celeba/pai_simplified_p2p_replacement_sweeps_n32"
read -r CROSS SELF < <(
  python /home/yu1331/cf-diffusion-celeba/evaluation/select_best_replacement.py \
    --root "${REPLACEMENT_ROOT}" --attr "${ATTR}" --include-self-sweep
)
OUT_ROOT="/scratch/gilbreth/yu1331/evaluations/celeba/pai_directinv_localblend_calibration_n32"
OUT="${OUT_ROOT}/${ATTR}/threshold_${TH_TAG}_no_sub"
PAI_ROOT="/scratch/gilbreth/yu1331/ckpts/celeba/pai_lora/minisd_ca_pipeline_four_label_continuous_pai_lora_balanced/final"
BALANCED_SELECTION="/home/yu1331/cf-diffusion-celeba/configs/selection_val_balanced_n32_seed42.json"
GRID_SELECTION="/scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json"
mkdir -p "${OUT}"
cd /home/yu1331/cf-diffusion-celeba

COMMON=(
  --config configs/pai_lora_celeba_ca_pipeline_four_label.yaml
  --lora-dir "${PAI_ROOT}/lora"
  --pai-checkpoint "${PAI_ROOT}/mapper/pai_mapper.pt"
  --steps 50
  --guidance-scale 3.0
  --invert-guidance-scale 1.0
  --negative-prompt "ugly, blurry, low res, unrealistic"
  --scheduler-mode ca
  --vae-latent-mode mean
  --editing ptp
  --condition-mode pai
  --inversion direct
  --ptp-controller ca
  --ptp-local-blend
  --ptp-blend-th "${TH}" "${TH}"
  --ptp-cross-replace-steps "${CROSS}"
  --ptp-self-replace-steps "${SELF}"
  --intervention-attr "${ATTR}"
)

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  "${COMMON[@]}" \
  --selection-json "${BALANCED_SELECTION}" \
  --split val \
  --max-samples 32 \
  --batch-size 2 \
  --target-scm causal_adapter \
  --output-json "${OUT}/metrics_do_${ATTR}_n32.json"

python -m evaluation.export_ca_style_prompt_lora_vs_causal_adapter_grid \
  "${COMMON[@]}" \
  --selection-json "${GRID_SELECTION}" \
  --out-dir "${OUT}" \
  --num-each 16 \
  --batch-size 2 \
  --export-ptp-masks \
  --ptp-mask-steps 0 4 9 24 49 \
  --skip-ca-grid
