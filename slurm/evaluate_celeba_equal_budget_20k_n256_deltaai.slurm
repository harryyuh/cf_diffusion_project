#!/bin/bash
#SBATCH --job-name=cel20k_n256
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PROJECT}/external/Causal-Adapter:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
ATTR=${ATTRS[$((SLURM_ARRAY_TASK_ID % 4))]}
if [[ ${METHOD_INDEX} -eq 0 ]]; then
  METHOD=pai
  CFG=configs/pai_lora_celeba_200k_bs2_deltaai.yaml
  CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/pai_lora/pai_20k_bs2/checkpoints/step_020000
else
  METHOD=joint_label
  CFG=configs/joint_label_lora_celeba_deltaai.yaml
  CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000
fi
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/equal_budget_20k_n256/${METHOD}
mkdir -p "${OUT}"
python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config "${CFG}" --lora-dir "${CKPT}/lora" \
  --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
  --selection-json /work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_val_balanced_n256_seed42.json \
  --split val --max-samples 256 --batch-size 2 --steps 50 \
  --guidance-scale 3.0 --invert-guidance-scale 1.0 \
  --negative-prompt "ugly, blurry, low res, unrealistic" \
  --scheduler-mode ca --vae-latent-mode mean --editing standard \
  --condition-mode "${METHOD}" --inversion ddim --target-scm causal_adapter \
  --benchmark-root "${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark" \
  --causal-adapter-root "${PROJECT}/external/Causal-Adapter" \
  --causal-adapter-scm /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt \
  --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/metrics_do_${ATTR}.json"
