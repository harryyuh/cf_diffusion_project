#!/bin/bash
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
cd /work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
ca="$PWD/external/Causal-Adapter"
export PYTHONPATH="$ca/diffusers/src:$ca/notebook_benchmarks:$ca:$PWD:${PYTHONPATH:-}"
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
python -u /work/nvme/bhje/hyu29/evaluations/rebuttal_cfg_v1.py "$@"
