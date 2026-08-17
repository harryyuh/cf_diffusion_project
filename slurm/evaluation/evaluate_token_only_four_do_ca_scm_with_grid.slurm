#!/bin/bash
#SBATCH --job-name=eval_tok4_4do
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=0-3
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_tok4_4do.%A_%a.out
#SBATCH --error=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/eval_tok4_4do.%A_%a.err

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
if ! command -v module >/dev/null 2>&1; then
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
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
ATTR="${ATTRS[$SLURM_ARRAY_TASK_ID]}"
CFG="configs/token_only_celeba_ca_pipeline_four_label.yaml"
MAPPER="/scratch/gilbreth/yu1331/ckpts/celeba/token_only/minisd_ca_pipeline_four_label_token_only_balanced/mapper"
SEL="/scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json"
OUT_DIR="/scratch/gilbreth/yu1331/evaluations/celeba/minisd_token_only_n256_ca_scm_targets_four_do"
OUT="${OUT_DIR}/metrics_do_${ATTR}_n256.json"
mkdir -p "${OUT_DIR}"

python -m evaluation.evaluate_token_only_celeba \
  --config "$CFG" \
  --mapper-dir "$MAPPER" \
  --selection-json "$SEL" \
  --split val \
  --max-samples 256 \
  --batch-size 4 \
  --steps 50 \
  --guidance-scale 1.0 \
  --intervention-attr "$ATTR" \
  --output-json "$OUT"

python -m evaluation.export_ca_style_token_only_vs_causal_adapter_grid \
  --config "$CFG" \
  --mapper-dir "$MAPPER" \
  --selection-json "$SEL" \
  --out-dir "$OUT_DIR" \
  --num-each 16 \
  --batch-size 4 \
  --steps 50 \
  --guidance-scale 1.0 \
  --intervention-attr "$ATTR"
