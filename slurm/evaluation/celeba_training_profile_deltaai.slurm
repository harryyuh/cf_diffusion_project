#!/bin/bash
#SBATCH --job-name=celeba_trprof
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/efficiency_profile_training_bs40_fp32
mkdir -p "${OUT}"
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface

if (( SLURM_ARRAY_TASK_ID == 0 )); then METHOD=ca; else METHOD=label_only; fi
GPU_LOG=${OUT}/${METHOD}_gpu_memory_mib.csv
TIME_LOG=${OUT}/${METHOD}_time.txt
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 200 > "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

if [[ "${METHOD}" == ca ]]; then
  TRAIN_OUT=/work/nvme/bhje/hyu29/ckpts/celeba/efficiency_profile/ca
  mkdir -p "${TRAIN_OUT}"
  cd "${CA_ROOT}"
  /usr/bin/time -v -o "${TIME_LOG}" python train.py \
    --seed 42 --output_name ca-efficiency-100step --output_dir "${TRAIN_OUT}" \
    --pretrained_model_name_or_path lambdalabs/miniSD-diffusers \
    --train_data_dir /work/nvme/bhje/hyu29/datasets/celebA \
    --dataset celeA_complex --resolution 256 --train_batch_size 40 \
    --gradient_accumulation_steps 1 --max_train_steps 100 \
    --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
    --adam_beta1 0.9 --adam_beta2 0.999 --adam_weight_decay 1e-2 \
    --adam_epsilon 1e-8 --max_grad_norm 1.0 --mixed_precision no \
    --save_steps 100 --checkpointing_steps 100 --checkpoints_total_limit 1 \
    --placeholder_string 'a human is @ and * and & and #' \
    --presudo_words '@,*,&,#' --presudo_words_infonce '@,*,&,#' \
    --infonce_temperature 0.2 --infonce_scale 0.0005 \
    --task_cond generation_text_global_after \
    --scm_path /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
else
  cd "${PROJECT}"
  /usr/bin/time -v -o "${TIME_LOG}" python -m training.train_joint_label_lora_celeba \
    --config audit_fix/joint_label_celeba_efficiency_profile.yaml
fi

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT
awk -F, 'BEGIN{m=0} {gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{print m}' "${GPU_LOG}" > "${OUT}/${METHOD}_peak_memory_mib.txt"
