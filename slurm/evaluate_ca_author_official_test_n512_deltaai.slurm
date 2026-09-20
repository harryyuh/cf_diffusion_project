#!/bin/bash
#SBATCH --job-name=caofficial_n512
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OFFICIAL=/work/nvme/bhje/hyu29/ckpts/celeba/ca_author_official_200k
SELECTION=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_balanced_n512_seed42.json
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_author_official_200k_test_balanced_n512_ddim100
EMBEDS=${OFFICIAL}/learned_embeds-steps-200000-token-hash-alias.safetensors

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald)
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "${OUT}"

python -m evaluation.compare_causal_adapter_parent_only_celeba \
  --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
  --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
  --selection-json "${SELECTION}" --split test --max-samples 512 --batch-size 4 \
  --ca-base-model-path lambdalabs/miniSD-diffusers \
  --ca-controlnet-path "${OFFICIAL}/controlnet-steps-200000.safetensors" \
  --ca-text-embedding-path "${EMBEDS}" --ca-scm-path "${OFFICIAL}/best_model.pt" \
  --ca-editing standard --ca-batched-standard --ca-steps 100 \
  --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
  --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/metrics_do_${ATTR}.json"
