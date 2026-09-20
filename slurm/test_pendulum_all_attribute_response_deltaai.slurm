#!/bin/bash
#SBATCH --job-name=pend_response
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
ATTRS=(pendulum light shadow_length shadow_position)
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
python -m evaluation.test_shadow_position_response \
  --attribute "${ATTR}" \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --config configs/joint_label_lora_pendulum_deltaai.yaml \
  --checkpoint-root /work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed42/final \
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
  --output-dir "/work/nvme/bhje/hyu29/evals/pendulum/all_attribute_response_label_only_cfg5/${ATTR}" \
  --samples 8 --steps 50 --guidance-scale 5.0 --seed 42
