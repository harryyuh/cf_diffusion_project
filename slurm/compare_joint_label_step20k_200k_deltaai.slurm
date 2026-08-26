#!/bin/bash
#SBATCH --job-name=joint20v200_grid
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
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

STEPS=(020000 200000)
ATTRS=(Male No_Beard)
STEP=${STEPS[$((SLURM_ARRAY_TASK_ID / 2))]}
ATTR=${ATTRS[$((SLURM_ARRAY_TASK_ID % 2))]}
ROOT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/joint_label_lora/sd15_joint_bs40_ctc_200k/checkpoints/step_${STEP}
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/joint_label_step20k_vs_200k_same32/step_${STEP}
mkdir -p "${OUT}"

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config configs/joint_label_lora_celeba_ca_paper_200k_deltaai.yaml \
  --lora-dir "${ROOT}/lora" --pai-checkpoint "${ROOT}/mapper/pai_mapper.pt" \
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
  --output-json "${OUT}/metrics_do_${ATTR}.json"
