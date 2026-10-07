#!/usr/bin/env bash
#SBATCH --job-name=pend_author_640
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%8
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
SHARD=${SLURM_ARRAY_TASK_ID}
OUT=${BASE}/evals/pendulum/ca_author20k_matched640_ddim100/shard_${SHARD}
python -m evaluation.evaluate_pendulum --method ca \
  --data-root ${BASE}/datasets/pendulum \
  --ca-root ${PROJECT}/external/Causal-Adapter \
  --ca-scm ${BASE}/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --regressor ${BASE}/ckpts/pendulum/regressor/best.pt \
  --base-model lambdalabs/miniSD-diffusers \
  --max-samples 0 --shard-index ${SHARD} --shard-size 64 \
  --steps 100 --seed 42 --grid-samples 4 \
  --conditioning-target full_scm \
  --ca-controlnet ${BASE}/models/Causal-Adapter/pendulum/controlnet/controlnet-steps-20000.safetensors \
  --ca-embedding ${BASE}/models/Causal-Adapter/pendulum/controlnet/learned_embeds-steps-20000.safetensors \
  --output-dir ${OUT}
python -m evaluation.reevaluate_pendulum_saved_ca_regressors \
  --input-dir ${OUT}/ca \
  --data-root ${BASE}/datasets/pendulum \
  --ca-root ${PROJECT}/external/Causal-Adapter \
  --ca-scm ${BASE}/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --checkpoint-dir ${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/checkpoints/pendulum/trained_classifiers
