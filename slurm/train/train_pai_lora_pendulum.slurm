#!/usr/bin/env bash
#SBATCH --job-name=pend_pai_lora
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x-%j.out

set -eo pipefail
PROJECT=/home/yu1331/cf-diffusion-celeba
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
python -m training.train_pai_lora_pendulum \
  --config configs/pai_lora_pendulum.yaml
