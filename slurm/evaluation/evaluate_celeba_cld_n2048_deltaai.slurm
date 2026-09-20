#!/bin/bash
#SBATCH --job-name=cld2048
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
BENCH=${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark
ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_protocol_n2048_ddim100
METHODS=(ca label_only)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
export PYTHONPATH=${BENCH}:${PROJECT}:${PYTHONPATH:-}
cd ${PROJECT}
python evaluation/evaluate_celeba_cld_saved.py \
  --saved-root ${ROOT} --method ${METHOD} \
  --celeba-root /work/nvme/bhje/hyu29/datasets/celebA \
  --benchmark-root ${BENCH} \
  --vae-checkpoint /work/nvme/bhje/hyu29/models/counterfactual-benchmark/unconditional_vae/celeba_uncond_image_vae-epoch=44.ckpt \
  --output ${ROOT}/${METHOD}/cld_metrics.json
