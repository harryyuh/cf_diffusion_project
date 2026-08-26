#!/bin/bash
#SBATCH --job-name=paper200k_eval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --array=0-11
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
ATTR=${ATTRS[$((SLURM_ARRAY_TASK_ID % 4))]}
SELECTION=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
OUTROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_paper_protocol_200k_test_full

if [[ ${METHOD_INDEX} -eq 0 ]]; then
  CA_RUN=$(find /work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/ca_200k -maxdepth 1 -type d -name '*causal-adapter*' | sort | tail -n 1)
  OUT=${OUTROOT}/ca
  mkdir -p "${OUT}"
  python -m evaluation.compare_causal_adapter_parent_only_celeba \
    --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
    --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
    --selection-json "${SELECTION}" --split test --max-samples 19962 --batch-size 4 \
    --ca-base-model-path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --ca-controlnet-path "${CA_RUN}/controlnet-steps-200000.safetensors" \
    --ca-text-embedding-path "${CA_RUN}/learned_embeds-steps-200000.safetensors" \
    --ca-scm-path "${SCM}" --ca-editing standard --ca-steps 100 \
    --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
    --intervention-attr "${ATTR}" \
    --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
    --output-json "${OUT}/metrics_do_${ATTR}.json"
else
  if [[ ${METHOD_INDEX} -eq 1 ]]; then
    METHOD=pai
    CFG=configs/pai_lora_celeba_ca_paper_200k_deltaai.yaml
    CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/pai_lora/sd15_pai_bs40_ctc_200k/checkpoints/step_200000
  else
    METHOD=joint_label
    CFG=configs/joint_label_lora_celeba_ca_paper_200k_deltaai.yaml
    CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/joint_label_lora/sd15_joint_bs40_ctc_200k/checkpoints/step_200000
  fi
  OUT=${OUTROOT}/${METHOD}
  mkdir -p "${OUT}"
  python -m evaluation.evaluate_latent_prompt_lora_celeba \
    --config "${CFG}" --lora-dir "${CKPT}/lora" \
    --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
    --selection-json "${SELECTION}" --split test --max-samples 19962 \
    --batch-size 4 --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
    --negative-prompt "ugly, blurry, low res, unrealistic" \
    --scheduler-mode ca --vae-latent-mode mean --editing standard \
    --condition-mode "${METHOD}" --inversion ddim --target-scm causal_adapter \
    --causal-adapter-scm "${SCM}" --causal-adapter-root "${CA_ROOT}" \
    --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
    --intervention-attr "${ATTR}" \
    --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
    --output-json "${OUT}/metrics_do_${ATTR}.json"
fi
