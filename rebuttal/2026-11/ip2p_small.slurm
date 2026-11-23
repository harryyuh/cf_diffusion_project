#!/bin/bash
#SBATCH -p ghx4
#SBATCH -A bhje-dtai-gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
cd /work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH="$PWD/external/Causal-Adapter/diffusers/src:$PWD/external/Causal-Adapter:$PWD:${PYTHONPATH:-}"
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
python -u /work/nvme/bhje/hyu29/evaluations/ip2p_small.py "$1"
