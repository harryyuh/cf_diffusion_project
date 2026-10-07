#!/usr/bin/env bash
#SBATCH --job-name=cel_matched50k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
PLAN_ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100/plans
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/matched_50k_bs40_fulltest_ddim100
CA_CURVE=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/ca_seed42/2026-08-27T11-02-42-ca-official-paper-seed42generation_text_global_after
LABEL_CURVE=/work/nvme/bhje/hyu29/ckpts/celeba/strict_learning_curve/label_seed42_50k/minisd_label_fp32_bs40_50k/checkpoints/step_050000
OFFICIAL_SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
MODE=${MODE:-full}
if [[ "${MODE}" == smoke ]]; then OUT=${OUT}_smoke; fi

NUM_PLANS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_plans"])' "${PLAN_ROOT}/manifest.json")
TASK=${SLURM_ARRAY_TASK_ID}
if (( TASK < NUM_PLANS )); then METHOD=ca; PLAN=${TASK}; else METHOD=label_only; PLAN=$((TASK-NUM_PLANS)); fi
SEL=$(printf "%s/plan_%03d.json" "${PLAN_ROOT}" "${PLAN}")
ATTR=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["intervention_attr"])' "${SEL}")
COUNT=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["male_indices"]))' "${SEL}")
if [[ "${MODE}" == smoke ]]; then COUNT=8; fi
METHOD_OUT=$(printf "%s/%s/plan_%03d_%s" "${OUT}" "${METHOD}" "${PLAN}" "${ATTR}")
mkdir -p "${METHOD_OUT}/tensors"
if [[ -f "${METHOD_OUT}/DONE" ]]; then exit 0; fi

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

if [[ "${METHOD}" == ca ]]; then
  test -s "${CA_CURVE}/controlnet-steps-50000.safetensors/diffusion_pytorch_model.safetensors"
  test -s "${CA_CURVE}/learned_embeds-steps-50000.safetensors"
  python -m evaluation.compare_causal_adapter_parent_only_celeba_n2048 \
    --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
    --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
    --selection-json "${SEL}" --split test --max-samples "${COUNT}" --batch-size 4 \
    --ca-base-model-path lambdalabs/miniSD-diffusers \
    --ca-controlnet-path "${CA_CURVE}/controlnet-steps-50000.safetensors" \
    --ca-text-embedding-path "${CA_CURVE}/learned_embeds-steps-50000.safetensors" \
    --ca-scm-path "${OFFICIAL_SCM}" --ca-editing standard --ca-batched-standard \
    --ca-steps 100 --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
    --intervention-attr "${ATTR}" --save-fake-dir "${METHOD_OUT}/tensors" \
    --output-json "${METHOD_OUT}/metrics.json"
else
  test -s "${LABEL_CURVE}/lora/pytorch_lora_weights.safetensors"
  test -s "${LABEL_CURVE}/mapper/pai_mapper.pt"
  python -m evaluation.evaluate_latent_prompt_lora_celeba_n2048 \
    --config configs/joint_label_celeba_strict_curve_50k.yaml \
    --lora-dir "${LABEL_CURVE}/lora" --pai-checkpoint "${LABEL_CURVE}/mapper/pai_mapper.pt" \
    --selection-json "${SEL}" --split test --max-samples "${COUNT}" --batch-size 4 \
    --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
    --negative-prompt "ugly, blurry, low res, unrealistic" \
    --scheduler-mode ca --vae-latent-mode mean --condition-mode joint_label \
    --target-scm causal_adapter --causal-adapter-root "${CA_ROOT}" \
    --causal-adapter-scm "${OFFICIAL_SCM}" \
    --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
    --intervention-attr "${ATTR}" --editing standard --inversion ddim \
    --save-fake-dir "${METHOD_OUT}/tensors" --output-json "${METHOD_OUT}/metrics.json"
fi
test -s "${METHOD_OUT}/metrics.json"
touch "${METHOD_OUT}/DONE"
