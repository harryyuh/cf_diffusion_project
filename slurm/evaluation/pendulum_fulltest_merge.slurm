#!/usr/bin/env bash
#SBATCH --job-name=pend_full_merge
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -euo pipefail
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
cd "${PROJECT}"
python -m evaluation.merge_pendulum_fulltest_50k \
  --root /work/nvme/bhje/hyu29/evals/pendulum/full_available_50k_ddim100 \
  --shards 29 --expected-n 1826
