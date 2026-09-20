#!/bin/bash
#SBATCH --job-name=label_recover5k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
cd /work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=$PWD/external/Causal-Adapter/diffusers/src:$PWD:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
python -m training.train_pendulum_recovery --config configs/pendulum_label_50k_recovery_noctc.yaml
