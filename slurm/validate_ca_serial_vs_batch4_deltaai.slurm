#!/bin/bash
#SBATCH --job-name=ca_batch4_check
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

CA_RUN=$(find /work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/ca_200k -maxdepth 1 -type d -name '*causal-adapter*' | sort | tail -n 1)
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
SELECTION=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json
MODE=serial
BATCH_FLAG=()
if [[ ${SLURM_ARRAY_TASK_ID} -eq 1 ]]; then
  MODE=batch4
  BATCH_FLAG=(--ca-batched-standard)
fi
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_serial_vs_batch4_same8/${MODE}
mkdir -p "${OUT}"

python -m evaluation.compare_causal_adapter_parent_only_celeba \
  --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
  --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
  --selection-json "${SELECTION}" --split test --max-samples 8 --batch-size 4 \
  --ca-base-model-path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --ca-controlnet-path "${CA_RUN}/controlnet-steps-200000.safetensors" \
  --ca-text-embedding-path "${CA_RUN}/learned_embeds-steps-200000.safetensors" \
  --ca-scm-path "${SCM}" --ca-editing standard --ca-steps 100 \
  --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
  --intervention-attr Male "${BATCH_FLAG[@]}" \
  --output-grid "${OUT}/grid_do_Male_source_cf.png" \
  --output-json "${OUT}/metrics_do_Male.json"
