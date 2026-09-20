#!/bin/bash
#SBATCH --job-name=ca_2cohort_grid
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --array=0-7
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
CA_RUN=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/ca_200k/2026-08-19T22-33-25-causal-adapter-paper-protocolgeneration_text_global_after
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
OUTROOT=/work/nvme/bhje/hyu29/evaluations/celeba/cross_cohort_20k_200k_ddim100_n32

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald)
COHORT_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
ATTR=${ATTRS[$((SLURM_ARRAY_TASK_ID % 4))]}

if [[ ${COHORT_INDEX} -eq 0 ]]; then
  COHORT=sd15_200k_on_200k_test
  SELECTION=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json
  SPLIT=test
else
  COHORT=sd15_200k_on_20k_balanced_val
  SELECTION=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_val_balanced_n256_seed42.json
  SPLIT=val
fi

OUT=${OUTROOT}/${COHORT}/ca
mkdir -p "${OUT}"
python -m evaluation.compare_causal_adapter_parent_only_celeba \
  --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
  --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
  --selection-json "${SELECTION}" --split "${SPLIT}" --max-samples 32 --batch-size 2 \
  --ca-base-model-path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --ca-controlnet-path "${CA_RUN}/controlnet-steps-200000.safetensors" \
  --ca-text-embedding-path "${CA_RUN}/learned_embeds-steps-200000.safetensors" \
  --ca-scm-path "${SCM}" --ca-editing standard --ca-steps 100 \
  --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
  --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/metrics_do_${ATTR}.json"
