#!/bin/bash
#SBATCH --job-name=ours_paper200k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
if [[ ${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  python -m training.train_pai_lora_celeba --config configs/pai_lora_celeba_ca_paper_200k_deltaai.yaml
else
  python -m training.train_joint_label_lora_celeba --config configs/joint_label_lora_celeba_ca_paper_200k_deltaai.yaml
fi
