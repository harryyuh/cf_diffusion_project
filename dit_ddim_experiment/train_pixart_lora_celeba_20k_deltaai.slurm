#!/bin/bash
#SBATCH --job-name=pixart_cel20k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export MPLCONFIGDIR=/work/nvme/bhje/hyu29/cache/matplotlib
cd "${PROJECT}"
python dit_ddim_experiment/train_pixart_lora_celeba.py \
  --config dit_ddim_experiment/pixart_lora_celeba_20k_deltaai.yaml
