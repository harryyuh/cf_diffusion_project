#!/bin/bash
#SBATCH --job-name=ca_official200k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OUT=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/ca_seed42
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
mkdir -p "${OUT}"
cd "${CA_ROOT}"
/usr/bin/time -v -o "${OUT}/time_200k.txt" python train.py \
  --seed 42 --output_name ca-official-paper-seed42 --output_dir "${OUT}" \
  --pretrained_model_name_or_path lambdalabs/miniSD-diffusers \
  --train_data_dir /work/nvme/bhje/hyu29/datasets/celebA \
  --dataset celeA_complex --resolution 256 --train_batch_size 40 \
  --gradient_accumulation_steps 1 --max_train_steps 200000 \
  --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --adam_beta1 0.9 --adam_beta2 0.999 --adam_weight_decay 1e-2 \
  --adam_epsilon 1e-8 --max_grad_norm 1.0 --mixed_precision no \
  --save_steps 10000 --checkpointing_steps 10000 --checkpoints_total_limit 25 \
  --placeholder_string 'a human is @ and * and & and #' \
  --presudo_words '@,*,&,#' --presudo_words_infonce '@,*,&,#' \
  --infonce_temperature 0.2 --infonce_scale 0.0005 \
  --task_cond generation_text_global_after \
  --scm_path /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
