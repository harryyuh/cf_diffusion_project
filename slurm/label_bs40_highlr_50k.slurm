#!/bin/bash
#SBATCH --job-name=lbl40_hi50k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OUT=/work/nvme/bhje/hyu29/ckpts/celeba/label_bs40_highlr_20260930
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$PROJECT"
nvidia-smi
date -u
/usr/bin/time -v -o "$OUT/time_50k.txt" python -m training.train_joint_label_lora_celeba --config configs/label_curve_bs40_highlr_50k.yaml
date -u
