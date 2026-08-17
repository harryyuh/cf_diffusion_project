#!/bin/bash
#SBATCH --job-name=jointlbl_eval
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
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
PROJECT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PROJECT}/external/Causal-Adapter:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald Young Male No_Beard Bald)
MODES=(standard standard standard standard tuned_ptp tuned_ptp tuned_ptp tuned_ptp)
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
case "${ATTR}" in
  Young) CROSS=0.10; SELF=0.10 ;;
  Male) CROSS=0.20; SELF=0.20 ;;
  No_Beard) CROSS=0.20; SELF=0.10 ;;
  Bald) CROSS=0.05; SELF=0.00 ;;
esac

ROOT=/scratch/gilbreth/yu1331/ckpts/celeba/joint_label_lora/minisd_four_label_joint_tokens_balanced/final
SELECTION=/scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json
OUT=/scratch/gilbreth/yu1331/evaluations/celeba/joint_label_lora_n256/${ATTR}/${MODE}
mkdir -p "${OUT}"

COMMON=(
  --config configs/joint_label_lora_celeba.yaml
  --lora-dir "${ROOT}/lora"
  --pai-checkpoint "${ROOT}/mapper/pai_mapper.pt"
  --selection-json "${SELECTION}"
  --split val --max-samples 256 --batch-size 2
  --steps 50 --guidance-scale 3.0 --invert-guidance-scale 1.0
  --negative-prompt "ugly, blurry, low res, unrealistic"
  --scheduler-mode ca --vae-latent-mode mean
  --condition-mode joint_label --target-scm causal_adapter
  --intervention-attr "${ATTR}"
)

if [[ "${MODE}" == tuned_ptp ]]; then
  EDIT=(--editing ptp --inversion ddim --ptp-controller simple
        --ptp-cross-replace-steps "${CROSS}" --ptp-self-replace-steps "${SELF}")
else
  EDIT=(--editing standard --inversion ddim)
fi

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  "${COMMON[@]}" "${EDIT[@]}" --output-json "${OUT}/metrics.json"
python -m evaluation.export_ca_style_prompt_lora_vs_causal_adapter_grid \
  "${COMMON[@]}" "${EDIT[@]}" --out-dir "${OUT}" --num-each 16 --skip-ca-grid
