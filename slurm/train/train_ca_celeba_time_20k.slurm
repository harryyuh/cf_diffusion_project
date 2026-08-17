#!/bin/bash
#SBATCH --job-name=ca_time20k
#SBATCH --nodes=1
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x.%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u

PROJECT=/home/yu1331/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
BASE=/scratch/gilbreth/yu1331/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce
OUT=/scratch/gilbreth/yu1331/ckpts/celeba/training_time/ca_celeba_complex_20k
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
mkdir -p "${OUT}" "${MPLCONFIGDIR}"
cd "${CA_ROOT}"

echo "TIMING_START_UTC=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v -o "${OUT}/time_20k.txt" python train.py \
  --output_name causal-adapter \
  --output_dir "${OUT}" \
  --pretrained_model_name_or_path "${BASE}" \
  --train_data_dir /scratch/gilbreth/yu1331/datasets/celebA \
  --dataset celeA_complex \
  --resolution 256 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 20000 \
  --learning_rate 5e-6 \
  --mixed_precision bf16 \
  --save_steps 1000 \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 25 \
  --placeholder_string 'an image of @ * & !' \
  --presudo_words '@,*,&,!' \
  --presudo_words_infonce '@,*,&,!' \
  --scm_path /scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt
echo "TIMING_END_UTC=$(date -u +%FT%TZ)"
