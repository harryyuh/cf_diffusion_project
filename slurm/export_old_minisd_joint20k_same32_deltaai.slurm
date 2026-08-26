#!/bin/bash
#SBATCH --job-name=oldminisd20_grid
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Male No_Beard)
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/old_minisd_joint20k_same32_grid
mkdir -p "${OUT}"

# The evaluator is used only as the existing deterministic image-export path.
# max-samples=32 avoids a full evaluation run; only its source|CF grid is retained.
python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config configs/joint_label_lora_celeba_deltaai.yaml \
  --lora-dir "${CKPT}/lora" --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
  --selection-json /work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json \
  --split test --max-samples 32 --batch-size 4 --steps 100 \
  --guidance-scale 3.0 --invert-guidance-scale 1.0 \
  --negative-prompt "ugly, blurry, low res, unrealistic" \
  --scheduler-mode ca --vae-latent-mode mean --editing standard \
  --condition-mode joint_label --inversion ddim --target-scm causal_adapter \
  --causal-adapter-scm /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt \
  --causal-adapter-root "${CA_ROOT}" \
  --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
  --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/temporary_${ATTR}.json"
