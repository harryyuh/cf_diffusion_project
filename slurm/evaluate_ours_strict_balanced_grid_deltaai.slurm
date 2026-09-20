#!/bin/bash
#SBATCH --job-name=ours_balgrid
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%8
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
SELECTION_ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/selection/balanced_grid_n32_seed42
OUTROOT=/work/nvme/bhje/hyu29/evaluations/celeba/strict_200k_attribute_balanced_grids_n32_ddim100

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
ATTR=${ATTRS[$((SLURM_ARRAY_TASK_ID % 4))]}
SELECTION=${SELECTION_ROOT}/selection_test_${ATTR}_balanced_n32_seed42.json

if [[ ${METHOD_INDEX} -eq 0 ]]; then
  METHOD=pai_strict_200k
  CONDITION=pai
  CFG=configs/pai_lora_celeba_official_paper_seed42_deltaai.yaml
  CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/pai_seed42/minisd_pai_fp32_bs40_200k/checkpoints/step_200000
else
  METHOD=joint_label_strict_200k
  CONDITION=joint_label
  CFG=configs/joint_label_lora_celeba_official_paper_seed42_deltaai.yaml
  CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/joint_label_seed42/minisd_joint_fp32_bs40_200k/checkpoints/step_200000
fi

OUT=${OUTROOT}/${METHOD}
mkdir -p "${OUT}"

python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config "${CFG}" --lora-dir "${CKPT}/lora" \
  --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
  --selection-json "${SELECTION}" --split test --max-samples 32 --batch-size 4 \
  --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
  --negative-prompt "ugly, blurry, low res, unrealistic" \
  --scheduler-mode ca --vae-latent-mode mean --editing standard \
  --condition-mode "${CONDITION}" --inversion ddim --target-scm causal_adapter \
  --causal-adapter-scm "${SCM}" --causal-adapter-root "${CA_ROOT}" \
  --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
  --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/metrics_do_${ATTR}.json"
