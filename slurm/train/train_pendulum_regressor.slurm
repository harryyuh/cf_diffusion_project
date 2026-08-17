#!/usr/bin/env bash
#SBATCH --job-name=pend_regressor
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x-%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
PROJECT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"
python -m training.train_pendulum_regressor \
  --data-root /scratch/gilbreth/yu1331/datasets/pendulum \
  --output /scratch/gilbreth/yu1331/ckpts/pendulum/regressor/best.pt
