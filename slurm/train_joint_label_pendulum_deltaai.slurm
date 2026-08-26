#!/usr/bin/env bash
#SBATCH --job-name=pend_joint_20k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x-%j.out

set -eo pipefail
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
VENV=/work/nvme/bhje/hyu29/venvs/cf-celeba
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source "${VENV}/bin/activate"
mkdir -p /work/nvme/bhje/hyu29/logs
python -m training.train_pai_lora_pendulum \
  --config configs/joint_label_lora_pendulum_deltaai.yaml
