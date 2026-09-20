#!/bin/bash
#SBATCH --job-name=pend50k40eval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_RUN_ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/paper_setting_50k_bs40/ca
OUT=/work/nvme/bhje/hyu29/evals/pendulum/paper_setting_50k_bs40_seed42_n256_ddim100
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
COMMON=(
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum
  --selection-json /work/nvme/bhje/hyu29/evals/pendulum/equal_budget_official_protocol_seed42_n256/pai/pai/selection.json
  --ca-root "${PROJECT}/external/Causal-Adapter"
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt
  --base-model lambdalabs/miniSD-diffusers
  --max-samples 256 --grid-samples 8 --steps 100 --seed 42
)
case ${SLURM_ARRAY_TASK_ID} in
  0)
    CA_RUN=$(find "${CA_RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '*pendulum-paper50k-bs40-seed42*' | sort | tail -1)
    python -m evaluation.evaluate_pendulum --method ca "${COMMON[@]}" \
      --ca-controlnet "${CA_RUN}/controlnet-steps-50000.safetensors" \
      --ca-embedding "${CA_RUN}/learned_embeds-steps-50000.safetensors" \
      --output-dir "${OUT}/ca"
    ;;
  1)
    python -m evaluation.evaluate_pendulum --method pai "${COMMON[@]}" \
      --pai-config configs/pendulum_pai_50k_bs40_paper_setting.yaml \
      --pai-root /work/nvme/bhje/hyu29/ckpts/pendulum/paper_setting_50k_bs40/pai/minisd_pai_50k_bs40_seed42/final \
      --output-dir "${OUT}/pai"
    ;;
  2)
    python -m evaluation.evaluate_pendulum --method joint_label "${COMMON[@]}" \
      --joint-config configs/pendulum_label_50k_bs40_paper_setting.yaml \
      --joint-root /work/nvme/bhje/hyu29/ckpts/pendulum/paper_setting_50k_bs40/label_only/minisd_label_50k_bs40_seed42/final \
      --output-dir "${OUT}/label_only"
    ;;
esac
