#!/usr/bin/env bash
#SBATCH --job-name=pend_joint_eval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x-%j.out

set -eo pipefail
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
VENV=/work/nvme/bhje/hyu29/venvs/cf-celeba
ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_joint_label_lora_20k
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source "${VENV}/bin/activate"
mkdir -p /work/nvme/bhje/hyu29/logs

for STEP in 005000 020000; do
  if [[ "${STEP}" == "020000" ]]; then
    MODEL_ROOT=${ROOT}/final
  else
    MODEL_ROOT=${ROOT}/checkpoints/step_${STEP}
  fi
  python -m evaluation.evaluate_pendulum \
    --method joint_label \
    --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
    --ca-root "${PROJECT}/external/Causal-Adapter" \
    --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
    --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
    --joint-config configs/joint_label_lora_pendulum_deltaai.yaml \
    --joint-root "${MODEL_ROOT}" \
    --output-dir "/work/nvme/bhje/hyu29/evals/pendulum/joint_label_${STEP}_n256" \
    --max-samples 256 --grid-samples 8 --steps 50 --seed 42
done
