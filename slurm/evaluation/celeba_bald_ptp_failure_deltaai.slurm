#!/bin/bash
#SBATCH --job-name=bald_ptp_fail
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
export PYTHONPATH="${CA_ROOT}/diffusers/src:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}"
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

SEL=/work/nvme/bhje/hyu29/evaluations/celeba/causal_adapter_vs_parent_only_n256/selection_val_balanced_n256_seed42.json
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/bald_ptp_masculinization_matched_n32

COMMON=(
  --selection-json "${SEL}"
  --intervention-attr Bald
  --num-each 16
  --batch-size 2
  --steps 50
  --guidance-scale 3.0
  --invert-guidance-scale 1.0
  --scheduler-mode ca
  --vae-latent-mode mean
  --ca-scm-path "${SCM}"
  --skip-ca-grid
)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    PAI=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/pai_seed42/minisd_pai_fp32_bs40_200k/final
    python -m evaluation.export_ca_style_prompt_lora_vs_causal_adapter_grid \
      "${COMMON[@]}" \
      --config configs/pai_lora_celeba_official_paper_seed42_deltaai.yaml \
      --lora-dir "${PAI}/lora" --pai-checkpoint "${PAI}/mapper/pai_mapper.pt" \
      --condition-mode pai --editing ptp --inversion ddim --ptp-controller simple \
      --ptp-cross-replace-steps 0.05 --ptp-self-replace-steps 0.0 \
      --out-dir "${ROOT}/ptp_bald_optimized"
    ;;
  1)
    PAI=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/pai_seed42/minisd_pai_fp32_bs40_200k/final
    python -m evaluation.export_ca_style_prompt_lora_vs_causal_adapter_grid \
      "${COMMON[@]}" \
      --config configs/pai_lora_celeba_official_paper_seed42_deltaai.yaml \
      --lora-dir "${PAI}/lora" --pai-checkpoint "${PAI}/mapper/pai_mapper.pt" \
      --condition-mode pai --editing ptp --inversion ddim --ptp-controller simple \
      --ptp-cross-replace-steps 0.2 --ptp-self-replace-steps 0.2 \
      --out-dir "${ROOT}/ptp_bald_generic"
    ;;
  2)
    JOINT=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/joint_label_seed42/minisd_joint_fp32_bs40_200k/final
    python -m evaluation.export_ca_style_prompt_lora_vs_causal_adapter_grid \
      "${COMMON[@]}" \
      --config configs/joint_label_lora_celeba_official_paper_seed42_deltaai.yaml \
      --lora-dir "${JOINT}/lora" --pai-checkpoint "${JOINT}/mapper/pai_mapper.pt" \
      --condition-mode joint_label --editing standard --inversion ddim \
      --out-dir "${ROOT}/label_only_bald"
    ;;
esac
