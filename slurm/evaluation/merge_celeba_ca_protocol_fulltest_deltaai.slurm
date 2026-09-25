#!/bin/bash
#SBATCH --job-name=merge_fid_full
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
cd /work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python evaluation/merge_celeba_ca_protocol_fulltest.py
ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100
BENCH=$PWD/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark
VAE=/work/nvme/bhje/hyu29/models/counterfactual-benchmark/unconditional_vae/celeba_uncond_image_vae-epoch=44.ckpt
for METHOD in ca label_only; do
  python evaluation/evaluate_celeba_cld_saved.py \
    --saved-root "${ROOT}" --method "${METHOD}" \
    --celeba-root /work/nvme/bhje/hyu29/datasets/celebA \
    --benchmark-root "${BENCH}" --vae-checkpoint "${VAE}" \
    --output "${ROOT}/cld_${METHOD}.json" --batch-size 128
done
