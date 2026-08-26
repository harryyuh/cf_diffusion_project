#!/bin/bash
#SBATCH --job-name=ca_cel200k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OUT=/work/nvme/bhje/hyu29/ckpts/celeba/training_time/ca_official_200k
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export MPLCONFIGDIR=/work/nvme/bhje/hyu29/cache/matplotlib
mkdir -p "${OUT}" "${MPLCONFIGDIR}"
cd "${CA_ROOT}"
/usr/bin/time -v -o "${OUT}/time_200k.txt" python train.py \
  --output_name causal-adapter \
  --output_dir "${OUT}" \
  --pretrained_model_name_or_path lambdalabs/miniSD-diffusers \
  --train_data_dir /work/nvme/bhje/hyu29/datasets/celebA \
  --dataset celeA_complex \
  --resolution 256 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 200000 \
  --learning_rate 5e-6 \
  --mixed_precision bf16 \
  --save_steps 10000 \
  --checkpointing_steps 10000 \
  --checkpoints_total_limit 25 \
  --placeholder_string 'an image of @ * & !' \
  --presudo_words '@,*,&,!' \
  --presudo_words_infonce '@,*,&,!' \
  --scm_path /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
