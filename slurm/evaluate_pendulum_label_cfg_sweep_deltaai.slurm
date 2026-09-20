#!/bin/bash
#SBATCH --job-name=pend_lbl_cfg
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
SCALES=(1.0 1.5 2.0 3.0 5.0)
SCALE=${SCALES[$SLURM_ARRAY_TASK_ID]}
python -m evaluation.evaluate_pendulum \
  --method joint_label \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --ca-root "${PROJECT}/external/Causal-Adapter" \
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
  --joint-config configs/joint_label_lora_pendulum_deltaai.yaml \
  --joint-root /work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed42/final \
  --output-dir /work/nvme/bhje/hyu29/evals/pendulum/label_only_cfg_sweep_n64/cfg_${SCALE} \
  --max-samples 64 --grid-samples 8 --steps 50 --seed 42 \
  --guidance-scale "${SCALE}"
