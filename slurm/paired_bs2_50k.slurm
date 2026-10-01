#!/bin/bash
#SBATCH --job-name=paired_bs2_50k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OUT=/work/nvme/bhje/hyu29/ckpts/celeba/paired_bs2_20260929
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$OUT/ca" "$OUT/label"
nvidia-smi
date -u
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
 cd "$CA_ROOT"
 /usr/bin/time -v -o "$OUT/ca/time_50k.txt" python train.py \
 --seed 42 --output_name ca-bs2-50k --output_dir "$OUT/ca" \
 --pretrained_model_name_or_path lambdalabs/miniSD-diffusers \
 --train_data_dir /work/nvme/bhje/hyu29/datasets/celebA \
 --dataset celeA_complex --resolution 256 --train_batch_size 2 \
 --gradient_accumulation_steps 1 --max_train_steps 50000 \
 --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
 --adam_beta1 0.9 --adam_beta2 0.999 --adam_weight_decay 1e-2 \
 --adam_epsilon 1e-8 --max_grad_norm 1.0 --mixed_precision bf16 \
 --save_steps 5000 --checkpointing_steps 5000 --checkpoints_total_limit 12 \
 --placeholder_string 'a human is @ and * and & and #' \
 --presudo_words '@,*,&,#' --presudo_words_infonce '@,*,&,#' \
 --infonce_temperature 0.2 --infonce_scale 0.0005 \
 --task_cond generation_text_global_after \
 --scm_path /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
else
 cd "$PROJECT"
 /usr/bin/time -v -o "$OUT/label/time_50k.txt" python -m training.train_joint_label_lora_celeba --config configs/label_curve_bs2_50k.yaml
fi
date -u
