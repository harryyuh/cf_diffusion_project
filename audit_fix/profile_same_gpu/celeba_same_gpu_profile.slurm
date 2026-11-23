#!/usr/bin/env bash
#SBATCH --job-name=ca_label_samegpu
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
PROFILE=${PROJECT}/audit_fix/profile_same_gpu
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/efficiency_profile_same_gpu/${SLURM_JOB_ID}
mkdir -p "${OUT}"
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export CF_STEP_PROFILE_WARMUP=30
export CF_STEP_PROFILE_MEASURE=200
export PYTHONPATH=${PROFILE}:${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}

for repeat in 1 2 3; do
  if (( repeat % 2 == 1 )); then order=(ca label_only); else order=(label_only ca); fi
  for method in "${order[@]}"; do
    export CF_STEP_PROFILE_METHOD=${method}
    export CF_STEP_PROFILE_REPEAT=${repeat}
    export CF_STEP_PROFILE_OUT=${OUT}/${method}_${repeat}.json
    if [[ "${method}" == ca ]]; then
      cd "${CA_ROOT}"
      python train.py \
        --seed 42 --output_name "samegpu-r${repeat}" --output_dir "${OUT}/ca" \
        --pretrained_model_name_or_path lambdalabs/miniSD-diffusers \
        --train_data_dir /work/nvme/bhje/hyu29/datasets/celebA \
        --dataset celeA_complex --resolution 256 --train_batch_size 40 \
        --gradient_accumulation_steps 1 --max_train_steps 230 \
        --dataloader_num_workers 8 \
        --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
        --adam_beta1 0.9 --adam_beta2 0.999 --adam_weight_decay 1e-2 \
        --adam_epsilon 1e-8 --max_grad_norm 1.0 --mixed_precision no \
        --save_steps 100000 --checkpointing_steps 100000 \
        --placeholder_string 'a human is @ and * and & and #' \
        --presudo_words '@,*,&,#' --presudo_words_infonce '@,*,&,#' \
        --infonce_temperature 0.2 --infonce_scale 0.0005 \
        --task_cond generation_text_global_after \
        --scm_path /work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt \
        > "${OUT}/${method}_${repeat}.log" 2>&1
    else
      cd "${PROJECT}"
      python -m training.train_joint_label_lora_celeba \
        --config "${PROFILE}/label_profile.yaml" \
        > "${OUT}/${method}_${repeat}.log" 2>&1
    fi
    test -s "${CF_STEP_PROFILE_OUT}"
  done
done

python - "${OUT}" <<'PY'
import json
import pathlib
import statistics
import sys
root = pathlib.Path(sys.argv[1])
rows = [json.loads(p.read_text()) for p in root.glob('*.json')]
summary = {}
for method in ('ca', 'label_only'):
    vals = [r['seconds_per_step'] for r in rows if r['method'] == method]
    assert len(vals) == 3, (method, vals)
    summary[method] = {'mean_seconds_per_step': statistics.mean(vals),
                       'sd_seconds_per_step': statistics.stdev(vals)}
summary['label_vs_ca_throughput_ratio'] = summary['ca']['mean_seconds_per_step'] / summary['label_only']['mean_seconds_per_step']
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
