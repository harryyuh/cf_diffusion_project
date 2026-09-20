#!/bin/bash
#SBATCH --job-name=ca_lbl_full
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-159%8
#SBATCH --signal=B:USR1@300
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_protocol_fulltest_ddim100
OFFICIAL=/work/nvme/bhje/hyu29/ckpts/celeba/ca_author_official_200k
LABEL=/work/nvme/bhje/hyu29/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000
NUM_SHARDS=80

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

TASK=${SLURM_ARRAY_TASK_ID}
if (( TASK < NUM_SHARDS )); then METHOD=ca; SHARD=${TASK}; else METHOD=label_only; SHARD=$((TASK-NUM_SHARDS)); fi
ATTRS=(Young Male No_Beard Bald)
ATTR=${ATTRS[$((SHARD%4))]}
SEL=${OUT}/plans/shard_${SHARD}.json
COUNT=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["male_indices"]))' "${SEL}")
METHOD_OUT=${OUT}/${METHOD}/shard_${SHARD}_${ATTR}
DONE=${METHOD_OUT}/DONE
if [[ -f "${DONE}" ]]; then echo "already complete ${DONE}"; exit 0; fi
mkdir -p "${METHOD_OUT}/tensors"

if [[ "${METHOD}" == ca ]]; then
  python -m evaluation.compare_causal_adapter_parent_only_celeba_n2048 \
    --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
    --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
    --selection-json "${SEL}" --split test --max-samples "${COUNT}" --batch-size 4 \
    --ca-base-model-path lambdalabs/miniSD-diffusers \
    --ca-controlnet-path "${OFFICIAL}/controlnet-steps-200000.safetensors" \
    --ca-text-embedding-path "${OFFICIAL}/learned_embeds-steps-200000-token-hash-alias.safetensors" \
    --ca-scm-path "${OFFICIAL}/best_model.pt" --ca-editing standard --ca-batched-standard \
    --ca-steps 100 --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
    --intervention-attr "${ATTR}" --save-fake-dir "${METHOD_OUT}/tensors" \
    --output-json "${METHOD_OUT}/metrics.json"
else
  python -m evaluation.evaluate_latent_prompt_lora_celeba_n2048 \
    --config configs/joint_label_lora_celeba_20k_bs2_seed42_eval_deltaai.yaml \
    --lora-dir "${LABEL}/lora" --pai-checkpoint "${LABEL}/mapper/pai_mapper.pt" \
    --selection-json "${SEL}" --split test --max-samples "${COUNT}" --batch-size 4 \
    --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
    --negative-prompt "ugly, blurry, low res, unrealistic" \
    --scheduler-mode ca --vae-latent-mode mean --condition-mode joint_label \
    --target-scm causal_adapter --causal-adapter-root "${CA_ROOT}" \
    --causal-adapter-scm "${OFFICIAL}/best_model.pt" \
    --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
    --intervention-attr "${ATTR}" --editing standard --inversion ddim \
    --save-fake-dir "${METHOD_OUT}/tensors" --output-json "${METHOD_OUT}/metrics.json"
fi
touch "${DONE}"
