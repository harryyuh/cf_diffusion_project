#!/usr/bin/env bash
#SBATCH --job-name=pend_eval
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x-%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u

PROJECT=/home/yu1331/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}/external/Causal-Adapter:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/scratch/gilbreth/yu1331/huggingface
export MPLCONFIGDIR=/scratch/gilbreth/yu1331/cache/matplotlib
METHOD=${METHOD:?set METHOD=ca or pai}
MAX_SAMPLES=${MAX_SAMPLES:-256}
TAG=${TAG:-n${MAX_SAMPLES}}
OUT=/scratch/gilbreth/yu1331/evaluations/pendulum/ca_vs_pai_${TAG}
mkdir -p "${OUT}" "${MPLCONFIGDIR}"
cd "${PROJECT}"
python -m evaluation.evaluate_pendulum \
  --method "${METHOD}" \
  --base-model /scratch/gilbreth/yu1331/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "$([[ "${METHOD}" == pai ]] && echo 4 || echo 1)" \
  --output-dir "${OUT}"
