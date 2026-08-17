#!/bin/bash
#SBATCH --job-name=joint_20k_grid
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
ROOT=/scratch/gilbreth/yu1331/ckpts/celeba/joint_label_lora/minisd_four_label_joint_tokens_balanced_20k_bs2/checkpoints
export PYTHONPATH=${PROJECT}:${PROJECT}/external/Causal-Adapter:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
cd "${PROJECT}"

STEPS=(20000 20000 20000 20000)
ATTRS=(Young Male No_Beard Bald)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
PADDED=$(printf '%06d' "${STEP}")
CKPT=${ROOT}/step_${PADDED}
OUT=/scratch/gilbreth/yu1331/evaluations/celeba/training_time/joint_label_20k_bs2_n256/step_${STEP}
mkdir -p "${OUT}"

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config configs/joint_label_lora_celeba.yaml \
  --lora-dir "${CKPT}/lora" \
  --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
  --selection-json /scratch/gilbreth/yu1331/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json \
  --split val --max-samples 32 --batch-size 2 \
  --steps 50 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
  --negative-prompt "ugly, blurry, low res, unrealistic" \
  --scheduler-mode ca --vae-latent-mode mean \
  --editing standard --condition-mode joint_label --inversion ddim \
  --target-scm causal_adapter --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/grid_metrics_do_${ATTR}.json"
