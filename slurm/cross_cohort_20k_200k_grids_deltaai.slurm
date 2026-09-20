#!/bin/bash
#SBATCH --job-name=20k200k_crossgrid
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
SELECTION_20K=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_val_balanced_n256_seed42.json
SELECTION_200K=/work/nvme/bhje/hyu29/evaluations/celeba/selection/selection_test_full_19962.json
OUTROOT=/work/nvme/bhje/hyu29/evaluations/celeba/cross_cohort_20k_200k_ddim100_n32

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

ATTRS=(Young Male No_Beard Bald)
EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
WITHIN_EXPERIMENT=$((SLURM_ARRAY_TASK_ID % 8))
METHOD_INDEX=$((WITHIN_EXPERIMENT / 4))
ATTR=${ATTRS[$((WITHIN_EXPERIMENT % 4))]}

if [[ ${METHOD_INDEX} -eq 0 ]]; then
  METHOD=pai
else
  METHOD=joint_label
fi

case ${EXPERIMENT_INDEX} in
  0)
    # SD1.5 200k checkpoint on the current 200k-grid test cohort.
    MODEL_TAG=sd15_200k_on_200k_test
    SELECTION=${SELECTION_200K}
    SPLIT=test
    if [[ ${METHOD} == pai ]]; then
      CFG=configs/pai_lora_celeba_ca_paper_200k_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/pai_lora/sd15_pai_bs40_ctc_200k/checkpoints/step_200000
    else
      CFG=configs/joint_label_lora_celeba_ca_paper_200k_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/joint_label_lora/sd15_joint_bs40_ctc_200k/checkpoints/step_200000
    fi
    ;;
  1)
    # SD1.5 200k checkpoint on the old latest-20k balanced validation cohort.
    MODEL_TAG=sd15_200k_on_20k_balanced_val
    SELECTION=${SELECTION_20K}
    SPLIT=val
    if [[ ${METHOD} == pai ]]; then
      CFG=configs/pai_lora_celeba_ca_paper_200k_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/pai_lora/sd15_pai_bs40_ctc_200k/checkpoints/step_200000
    else
      CFG=configs/joint_label_lora_celeba_ca_paper_200k_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/ca_paper_protocol/joint_label_lora/sd15_joint_bs40_ctc_200k/checkpoints/step_200000
    fi
    ;;
  2)
    # MiniSD 20k checkpoint on the current 200k-grid test cohort.
    MODEL_TAG=minisd_20k_on_200k_test
    SELECTION=${SELECTION_200K}
    SPLIT=test
    if [[ ${METHOD} == pai ]]; then
      CFG=configs/pai_lora_celeba_200k_bs2_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/pai_lora/pai_20k_bs2/checkpoints/step_020000
    else
      CFG=configs/joint_label_lora_celeba_deltaai.yaml
      CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000
    fi
    ;;
esac

OUT=${OUTROOT}/${MODEL_TAG}/${METHOD}
mkdir -p "${OUT}"
python -m evaluation.evaluate_latent_prompt_lora_celeba \
  --config "${CFG}" --lora-dir "${CKPT}/lora" \
  --pai-checkpoint "${CKPT}/mapper/pai_mapper.pt" \
  --selection-json "${SELECTION}" --split "${SPLIT}" --max-samples 32 \
  --batch-size 2 --steps 100 --guidance-scale 3.0 --invert-guidance-scale 1.0 \
  --negative-prompt "ugly, blurry, low res, unrealistic" \
  --scheduler-mode ca --vae-latent-mode mean --editing standard \
  --condition-mode "${METHOD}" --inversion ddim --target-scm causal_adapter \
  --causal-adapter-scm "${SCM}" --causal-adapter-root "${CA_ROOT}" \
  --benchmark-root "${CA_ROOT}/counterfactual-benchmark/counterfactual_benchmark" \
  --intervention-attr "${ATTR}" \
  --output-grid "${OUT}/grid_do_${ATTR}_source_cf.png" \
  --output-json "${OUT}/metrics_do_${ATTR}.json"
