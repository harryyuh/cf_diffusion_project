#!/bin/bash
#SBATCH --job-name=train_jointlbl
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x.%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
PROJECT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
cd "${PROJECT}"
python -m training.train_joint_label_lora_celeba \
  --config configs/joint_label_lora_celeba.yaml
