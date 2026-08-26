#!/usr/bin/env bash
#SBATCH --job-name=pend_data
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x-%j.out

set -eo pipefail
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
VENV=/work/nvme/bhje/hyu29/venvs/cf-celeba
DATA_PARENT=/work/nvme/bhje/hyu29/datasets
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source "${VENV}/bin/activate"
mkdir -p "${DATA_PARENT}" /work/nvme/bhje/hyu29/logs
python "${PROJECT}/training/generate_pendulum_parallel.py" \
  --output "${DATA_PARENT}/pendulum" --workers 12
