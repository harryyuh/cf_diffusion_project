#!/bin/bash
#SBATCH --job-name=merge_fid_full
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
cd /work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python evaluation/merge_celeba_ca_protocol_fulltest.py
