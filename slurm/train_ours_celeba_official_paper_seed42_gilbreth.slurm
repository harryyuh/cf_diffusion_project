#!/bin/bash
#SBATCH --job-name=ours_official200k
#SBATCH --partition=a100-80gb
#SBATCH --account=statdept
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gilbreth/yu1331/logs/%x.%A_%a.out

set -eo pipefail
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff

PROJECT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export HF_HUB_OFFLINE=1
mkdir -p /scratch/gilbreth/yu1331/logs
cd "${PROJECT}"
if [[ ${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  python -m training.train_pai_lora_celeba --config configs/pai_lora_celeba_official_paper_seed42_gilbreth.yaml
else
  python -m training.train_joint_label_lora_celeba --config configs/joint_label_lora_celeba_official_paper_seed42_gilbreth.yaml
fi
