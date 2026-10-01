#!/bin/bash
#SBATCH --job-name=prof_cf
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/efficiency_profile
PLAN=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100/plans/plan_000.json
OFFICIAL=/work/nvme/bhje/hyu29/ckpts/celeba/ca_author_official_200k
LABEL=/work/nvme/bhje/hyu29/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000

export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"
mkdir -p "${ROOT}"

COMMON=(--num-images 32 --selection-json "${PLAN}" --split test --max-samples 32 --batch-size 4)

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  python scripts/profile_cuda_module.py \
    --module evaluation.compare_causal_adapter_parent_only_celeba_n2048 \
    --output "${ROOT}/ca_profile.json" "${COMMON[@]}" \
    --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
    --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
    --ca-base-model-path lambdalabs/miniSD-diffusers \
    --ca-controlnet-path "${OFFICIAL}/controlnet-steps-200000.safetensors" \
    --ca-text-embedding-path "${OFFICIAL}/learned_embeds-steps-200000-token-hash-alias.safetensors" \
    --ca-scm-path "${OFFICIAL}/best_model.pt" --ca-editing standard --ca-batched-standard \
    --ca-steps 100 --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
    --intervention-attr Young --save-fake-dir "${ROOT}/ca_tensors" \
    --output-json "${ROOT}/ca_metrics.json"
else
  python scripts/profile_cuda_module.py \
    --module evaluation.evaluate_latent_prompt_lora_celeba_n2048 \
    --output "${ROOT}/label_profile.json" "${COMMON[@]}" \
    --config configs/joint_label_lora_celeba_20k_bs2_seed42_eval_deltaai.yaml \
    --lora-dir "${LABEL}/lora" --pai-checkpoint "${LABEL}/mapper/pai_mapper.pt" \
    --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
    --negative-prompt "ugly, blurry, low res, unrealistic" \
    --scheduler-mode ca --vae-latent-mode mean --condition-mode joint_label \
    --target-scm causal_adapter --causal-adapter-root "${CA_ROOT}" \
    --causal-adapter-scm "${OFFICIAL}/best_model.pt" \
    --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
    --intervention-attr Young --editing standard --inversion ddim \
    --save-fake-dir "${ROOT}/label_tensors" --output-json "${ROOT}/label_metrics.json"
fi
