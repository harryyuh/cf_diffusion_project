#!/bin/bash
#SBATCH --job-name=pend_ca_off20k
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_RUN=/work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/controlnet
OUT=/work/nvme/bhje/hyu29/evals/pendulum/ca_author_official20k_seed42_n256_ddim50
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
COMMON=(
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum
  --ca-root "${PROJECT}/external/Causal-Adapter"
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt
  --base-model lambdalabs/miniSD-diffusers
  --max-samples 256 --grid-samples 8 --steps 50 --seed 42
)
case ${SLURM_ARRAY_TASK_ID} in
  0)
    python -m evaluation.evaluate_pendulum --method ca "${COMMON[@]}" \
      --ca-controlnet "${CA_RUN}/controlnet-steps-20000.safetensors" \
      --ca-embedding "${CA_RUN}/learned_embeds-steps-20000.safetensors" \
      --output-dir "${OUT}/ca"
    ;;
esac
