#!/bin/bash
#SBATCH --job-name=ca_official200k
#SBATCH --partition=a100-80gb
#SBATCH --account=statdept
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gilbreth/yu1331/logs/%x.%j.out

set -eo pipefail
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff

PROJECT=/home/yu1331/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OUT=/scratch/gilbreth/yu1331/ckpts/celeba/official_paper_protocol/ca_seed42
MODEL=/scratch/gilbreth/yu1331/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export HF_HUB_OFFLINE=1
mkdir -p "${OUT}" /scratch/gilbreth/yu1331/logs
cd "${CA_ROOT}"
/usr/bin/time -v -o "${OUT}/time_200k.txt" python train.py \
  --seed 42 --output_name ca-official-paper-seed42 --output_dir "${OUT}" \
  --pretrained_model_name_or_path "${MODEL}" \
  --train_data_dir /scratch/gilbreth/yu1331/datasets/celebA \
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
  --scm_path /scratch/gilbreth/yu1331/ckpts/celeba/full_label_scm/causal_adapter_graph/checkpoints/full_label_scm_best.pt
