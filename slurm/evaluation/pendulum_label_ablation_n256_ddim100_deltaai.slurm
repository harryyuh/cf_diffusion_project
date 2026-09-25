#!/bin/bash
#SBATCH --job-name=pend_lbl_ablate
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  CONFIG=configs/joint_label_lora_pendulum_equal_budget_official_protocol.yaml
  ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed42/final
  TARGET=intervention_only
  NAME=rank16_no_scm
else
  CONFIG=configs/joint_label_lora_pendulum_rank64_official_protocol.yaml
  ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_rank64_equal_budget_official_protocol_20k_seed42/final
  TARGET=full_scm
  NAME=rank64_full_scm
fi

python -m evaluation.evaluate_pendulum \
  --method joint_label \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --ca-root "${PROJECT}/external/Causal-Adapter" \
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
  --joint-config "${CONFIG}" \
  --joint-root "${ROOT}" \
  --conditioning-target "${TARGET}" \
  --output-dir "/work/nvme/bhje/hyu29/evals/pendulum/label_ablation_seed42_n256_ddim100/${NAME}" \
  --max-samples 256 --batch-size 4 --grid-samples 8 --steps 100 --seed 42 --guidance-scale 1.0
