#!/bin/bash
#SBATCH --job-name=curve_eval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=0-23%8
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
PLAN_ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100/plans
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/matched_learning_curve_seed42_n512_randomdo_ddim100
OFFICIAL_SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
CA_CURVE=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/ca_seed42/2026-08-27T11-02-42-ca-official-paper-seed42generation_text_global_after
LABEL_CURVE=/work/nvme/bhje/hyu29/ckpts/celeba/strict_learning_curve/label_seed42_50k/minisd_label_fp32_bs40_50k/checkpoints

METHOD_ID=$((SLURM_ARRAY_TASK_ID / 12))
REM=$((SLURM_ARRAY_TASK_ID % 12))
STEP_ID=$((REM / 4))
ATTR_ID=$((REM % 4))
STEPS=(5000 20000 50000)
ATTRS=(Young Male No_Beard Bald)
PLANS=(0 20 40 61)
STEP=${STEPS[$STEP_ID]}
ATTR=${ATTRS[$ATTR_ID]}
PLAN=${PLANS[$ATTR_ID]}
SEL=$(printf "%s/plan_%03d.json" "${PLAN_ROOT}" "${PLAN}")
PADDED=$(printf "%06d" "${STEP}")

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

if (( METHOD_ID == 0 )); then
  METHOD=ca
  if (( STEP == 5000 )); then
    CA5_ROOT=$(find /work/nvme/bhje/hyu29/ckpts/celeba/strict_learning_curve/ca_seed42_5k \
      -type d -name "controlnet-steps-5000.safetensors" -printf '%h\n' | sort | tail -1)
    test -n "${CA5_ROOT}"
    CKPT_ROOT=${CA5_ROOT}
  else
    CKPT_ROOT=${CA_CURVE}
  fi
  RUN_OUT=${OUT}/${METHOD}/step_${PADDED}/do_${ATTR}
  mkdir -p "${RUN_OUT}/tensors"
  python -m evaluation.compare_causal_adapter_parent_only_celeba_n2048 \
    --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
    --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
    --selection-json "${SEL}" --split test --max-samples 128 --batch-size 4 \
    --ca-base-model-path lambdalabs/miniSD-diffusers \
    --ca-controlnet-path "${CKPT_ROOT}/controlnet-steps-${STEP}.safetensors" \
    --ca-text-embedding-path "${CKPT_ROOT}/learned_embeds-steps-${STEP}.safetensors" \
    --ca-scm-path "${OFFICIAL_SCM}" --ca-editing standard --ca-batched-standard \
    --ca-steps 100 --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
    --intervention-attr "${ATTR}" --save-fake-dir "${RUN_OUT}/tensors" \
    --output-json "${RUN_OUT}/metrics.json"
else
  METHOD=label_only
  CKPT=${LABEL_CURVE}/step_${PADDED}
  test -d "${CKPT}/lora"
  RUN_OUT=${OUT}/${METHOD}/step_${PADDED}/do_${ATTR}
  mkdir -p "${RUN_OUT}/tensors"
  python -m evaluation.evaluate_latent_prompt_lora_celeba_n2048 \
    --config configs/joint_label_celeba_strict_curve_50k.yaml \
    --lora-dir "${CKPT}/lora" --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
    --selection-json "${SEL}" --split test --max-samples 128 --batch-size 4 \
    --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
    --negative-prompt "ugly, blurry, low res, unrealistic" \
    --scheduler-mode ca --vae-latent-mode mean --condition-mode joint_label \
    --target-scm causal_adapter --causal-adapter-root "${CA_ROOT}" \
    --causal-adapter-scm "${OFFICIAL_SCM}" \
    --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
    --intervention-attr "${ATTR}" --editing standard --inversion ddim \
    --save-fake-dir "${RUN_OUT}/tensors" --output-json "${RUN_OUT}/metrics.json"
fi
