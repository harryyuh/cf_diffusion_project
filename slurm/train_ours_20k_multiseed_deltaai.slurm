#!/bin/bash
#SBATCH --job-name=ours20k_seeds
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

CONFIGS=(
  configs/pendulum_pai_20k_seed43_deltaai.yaml
  configs/pendulum_pai_20k_seed44_deltaai.yaml
  configs/pendulum_joint_20k_seed43_deltaai.yaml
  configs/pendulum_joint_20k_seed44_deltaai.yaml
  configs/pai_lora_celeba_20k_bs2_seed43_deltaai.yaml
  configs/pai_lora_celeba_20k_bs2_seed44_deltaai.yaml
  configs/joint_label_lora_celeba_20k_bs2_seed43_deltaai.yaml
  configs/joint_label_lora_celeba_20k_bs2_seed44_deltaai.yaml
)

if [[ ${SLURM_ARRAY_TASK_ID} -lt 2 ]]; then
  MODULE=training.train_pai_lora_pendulum
elif [[ ${SLURM_ARRAY_TASK_ID} -lt 4 ]]; then
  MODULE=training.train_pai_lora_pendulum
elif [[ ${SLURM_ARRAY_TASK_ID} -lt 6 ]]; then
  MODULE=training.train_pai_lora_celeba
else
  MODULE=training.train_joint_label_lora_celeba
fi

python -m "${MODULE}" --config "${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
