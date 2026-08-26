#!/bin/bash
#SBATCH --job-name=pend_ca_pai_eval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
# Causal-Adapter vendors a modified Diffusers build that defines
# Causal_ControlNetModel.  Put it ahead of the standard site-package while
# retaining this repository for the shared evaluator.
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
COMMON=(
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum
  --ca-root "${PROJECT}/external/Causal-Adapter"
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt
  --max-samples 256 --grid-samples 8 --steps 50 --seed 42
)
if [[ ${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  python -m evaluation.evaluate_pendulum --method ca "${COMMON[@]}" \
    --base-model lambdalabs/miniSD-diffusers \
    --ca-controlnet /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/controlnet/controlnet-steps-20000.safetensors \
    --ca-embedding /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/controlnet/learned_embeds-steps-20000.safetensors \
    --output-dir /work/nvme/bhje/hyu29/evals/pendulum/ca_20k_n256
else
  python -m evaluation.evaluate_pendulum --method pai "${COMMON[@]}" \
    --pai-config configs/pai_lora_pendulum_20k_deltaai.yaml \
    --pai-root /work/nvme/bhje/hyu29/ckpts/pendulum/pai_lora/minisd_continuous_pai_lora_20k/final \
    --output-dir /work/nvme/bhje/hyu29/evals/pendulum/pai_20k_n256
fi
