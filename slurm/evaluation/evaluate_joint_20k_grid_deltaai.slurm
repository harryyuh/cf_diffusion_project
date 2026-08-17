#!/bin/bash
#SBATCH --job-name=joint_20k_grid
#SBATCH --nodes=1
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --array=0-3
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
conda activate base
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
set -u
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
ROOT=/work/nvme/bhje/hyu29/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints
export PYTHONPATH=${PROJECT}:${PROJECT}/external/Causal-Adapter:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

STEPS=(20000 20000 20000 20000)
ATTRS=(Young Male No_Beard Bald)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
PADDED=$(printf '%06d' "${STEP}")
CKPT=${ROOT}/step_${PADDED}
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/training_time/joint_label_20k_bs2_n256/step_${STEP}
mkdir -p "${OUT}"

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config configs/joint_label_lora_celeba_deltaai.yaml \
  --lora-dir "${CKPT}/lora" \
  --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
  --selection-json /work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_val_balanced_n256_seed42.json \
  --split val --max-samples 32 --batch-size 2 \
  --steps 50 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
  --negative-prompt "ugly, blurry, low res, unrealistic" \
  --scheduler-mode ca --vae-latent-mode mean \
  --editing standard --condition-mode joint_label --inversion ddim \
  --causal-adapter-scm /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt \
  --target-scm causal_adapter --intervention-attr "${ATTR}" \
  --causal-adapter-root "${PROJECT}/external/Causal-Adapter" \
  --benchmark-root "${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/grid_metrics_do_${ATTR}.json"
