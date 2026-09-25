#!/bin/bash
#SBATCH --job-name=label_curve50k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
OUT=/work/nvme/bhje/hyu29/ckpts/celeba/strict_learning_curve/label_seed42_50k
mkdir -p "${OUT}"
cd "${PROJECT}"
/usr/bin/time -v -o "${OUT}/time_50k.txt" \
  python -m training.train_joint_label_lora_celeba \
    --config configs/joint_label_celeba_strict_curve_50k.yaml
