#!/bin/bash
#SBATCH --job-name=sixmethod_n512
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-23%12
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
SELECTION=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_balanced_n512_seed42.json
OUTROOT=/work/nvme/bhje/hyu29/evaluations/celeba/six_methods_test_balanced_n512_ddim100
CA20=/work/nvme/bhje/hyu29/ckpts/celeba/training_time/ca_20k
CA200=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/ca_200k/2026-08-19T22-33-25-causal-adapter-paper-protocolgeneration_text_global_after

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
ATTR=${ATTRS[$((SLURM_ARRAY_TASK_ID % 4))]}

case ${METHOD_INDEX} in
  0) METHOD=ca_20k ;;
  1) METHOD=pai_20k ;;
  2) METHOD=joint_label_20k ;;
  3) METHOD=ca_200k ;;
  4) METHOD=pai_200k ;;
  5) METHOD=joint_label_200k ;;
esac

OUT=${OUTROOT}/${METHOD}
mkdir -p "${OUT}"

if [[ ${METHOD} == ca_20k || ${METHOD} == ca_200k ]]; then
  if [[ ${METHOD} == ca_20k ]]; then
    BASE=lambdalabs/miniSD-diffusers
    CONTROL=${CA20}/controlnet-steps-20000.safetensors
    # This checkpoint was trained with `!` as slot 4, while the current
    # inference loader uses `#`; the alias preserves the learned tensor.
    EMBEDS=/work/nvme/bhje/hyu29/evaluations/celeba/assets/ca20_learned_embeds_token_hash_alias.safetensors
  else
    BASE=stable-diffusion-v1-5/stable-diffusion-v1-5
    CONTROL=${CA200}/controlnet-steps-200000.safetensors
    EMBEDS=${CA200}/learned_embeds-steps-200000.safetensors
  fi
  python -m evaluation.compare_causal_adapter_parent_only_celeba \
    --skip-parent-only --causal-adapter-root "${CA_ROOT}" \
    --parent-config configs/diffusion_celeba_parent_only_deltaai.yaml \
    --selection-json "${SELECTION}" --split test --max-samples 512 --batch-size 4 \
    --ca-base-model-path "${BASE}" --ca-controlnet-path "${CONTROL}" \
    --ca-text-embedding-path "${EMBEDS}" --ca-scm-path "${SCM}" \
    --ca-editing standard --ca-batched-standard --ca-steps 100 \
    --ca-guidance-scale 3.0 --ca-invert-guidance-scale 1.0 \
    --intervention-attr "${ATTR}" \
    --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
    --output-json "${OUT}/metrics_do_${ATTR}.json"
else
  case ${METHOD} in
    pai_20k)
      CONDITION=pai
      CFG=configs/pai_lora_celeba_200k_bs2_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/pai_lora/pai_20k_bs2/checkpoints/step_020000
      ;;
    joint_label_20k)
      CONDITION=joint_label
      CFG=configs/joint_label_lora_celeba_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000
      ;;
    pai_200k)
      CONDITION=pai
      CFG=configs/pai_lora_celeba_ca_paper_200k_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/pai_lora/sd15_pai_bs40_ctc_200k/checkpoints/step_200000
      ;;
    joint_label_200k)
      CONDITION=joint_label
      CFG=configs/joint_label_lora_celeba_ca_paper_200k_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/joint_label_lora/sd15_joint_bs40_ctc_200k/checkpoints/step_200000
      ;;
  esac
  python -m evaluation.evaluate_latent_prompt_lora_celeba \
    --config "${CFG}" --lora-dir "${CKPT}/lora" \
    --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
    --selection-json "${SELECTION}" --split test --max-samples 512 --batch-size 4 \
    --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
    --negative-prompt "ugly, blurry, low res, unrealistic" \
    --scheduler-mode ca --vae-latent-mode mean --editing standard \
    --condition-mode "${CONDITION}" --inversion ddim --target-scm causal_adapter \
    --causal-adapter-scm "${SCM}" --causal-adapter-root "${CA_ROOT}" \
    --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
    --intervention-attr "${ATTR}" \
    --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
    --output-json "${OUT}/metrics_do_${ATTR}.json"
fi
