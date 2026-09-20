#!/bin/bash
#SBATCH --job-name=pend20k_seedeval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"

case ${SLURM_ARRAY_TASK_ID} in
  0) METHOD=pai; SEED=43; CFG=configs/pendulum_pai_20k_seed43_deltaai.yaml; ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/pai_lora/minisd_pai_equal_budget_official_protocol_20k_seed43/final ;;
  1) METHOD=pai; SEED=44; CFG=configs/pendulum_pai_20k_seed44_deltaai.yaml; ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/pai_lora/minisd_pai_equal_budget_official_protocol_20k_seed44/final ;;
  2) METHOD=joint_label; SEED=43; CFG=configs/pendulum_joint_20k_seed43_deltaai.yaml; ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed43/final ;;
  3) METHOD=joint_label; SEED=44; CFG=configs/pendulum_joint_20k_seed44_deltaai.yaml; ROOT=/work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed44/final ;;
esac
OUT=/work/nvme/bhje/hyu29/evals/pendulum/equal_budget_official_protocol_multiseed/seed${SEED}/${METHOD}
COMMON=(
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum
  --ca-root "${PROJECT}/external/Causal-Adapter"
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt
  --base-model lambdalabs/miniSD-diffusers
  --max-samples 256 --grid-samples 8 --steps 50 --seed 42
)
if [[ ${METHOD} == pai ]]; then
  python -m evaluation.evaluate_pendulum --method pai "${COMMON[@]}" \
    --pai-config "${CFG}" --pai-root "${ROOT}" --output-dir "${OUT}"
else
  python -m evaluation.evaluate_pendulum --method joint_label "${COMMON[@]}" \
    --joint-config "${CFG}" --joint-root "${ROOT}" --output-dir "${OUT}"
fi
