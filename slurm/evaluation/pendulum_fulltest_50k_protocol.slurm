#!/usr/bin/env bash
#SBATCH --job-name=pend_full_50k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-57%8
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
BASE=/work/nvme/bhje/hyu29
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=${BASE}/huggingface
cd "${PROJECT}"

SHARD_SIZE=64
N_SHARDS=29
if [[ "${SMOKE:-0}" == 1 ]]; then
  METHOD_INDEX=${SLURM_ARRAY_TASK_ID}
  SHARD_INDEX=0
  LIMIT=2
  OUT=${BASE}/evals/pendulum/protocol_smoke_50k_ddim100
else
  METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / N_SHARDS))
  SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % N_SHARDS))
  LIMIT=0
  OUT=${BASE}/evals/pendulum/full_available_50k_ddim100
fi

COMMON=(
  --data-root ${BASE}/datasets/pendulum
  --ca-root ${PROJECT}/external/Causal-Adapter
  --ca-scm ${BASE}/models/Causal-Adapter/pendulum/scm/best_model.pt
  --regressor ${BASE}/ckpts/pendulum/regressor/best.pt
  --base-model lambdalabs/miniSD-diffusers
  --max-samples ${LIMIT}
  --shard-index ${SHARD_INDEX}
  --shard-size ${SHARD_SIZE}
  --steps 100
  --seed 42
  --grid-samples 4
  --conditioning-target full_scm
  --output-dir ${OUT}/shard_${SHARD_INDEX}
)

case ${METHOD_INDEX} in
  0)
    METHOD=ca
    CA_RUN=${BASE}/ckpts/pendulum/paper_setting_50k_bs40/ca/2026-09-14T15-33-52-pendulum-paper50k-bs40-seed42generation_text_global_after
    python -m evaluation.evaluate_pendulum --method ca "${COMMON[@]}" \
      --ca-controlnet ${CA_RUN}/controlnet-steps-50000.safetensors \
      --ca-embedding ${CA_RUN}/learned_embeds-steps-50000.safetensors
    ;;
  1)
    METHOD=joint_label
    python -m evaluation.evaluate_pendulum --method joint_label "${COMMON[@]}" \
      --joint-config configs/pendulum_label_50k_bs40_paper_setting.yaml \
      --joint-root ${BASE}/ckpts/pendulum/paper_setting_50k_bs40/label_only/minisd_label_50k_bs40_seed42/final
    ;;
  *) echo "invalid method index ${METHOD_INDEX}" >&2; exit 2 ;;
esac

LEAF=${OUT}/shard_${SHARD_INDEX}/${METHOD}
python -m evaluation.reevaluate_pendulum_saved_ca_regressors \
  --input-dir "${LEAF}" \
  --data-root ${BASE}/datasets/pendulum \
  --ca-root ${PROJECT}/external/Causal-Adapter \
  --ca-scm ${BASE}/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --checkpoint-dir ${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/checkpoints/pendulum/trained_classifiers
