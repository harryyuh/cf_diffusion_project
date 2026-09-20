#!/bin/bash
#SBATCH --job-name=eval_lbl50k_fix
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
python -m evaluation.evaluate_pendulum \
  --method joint_label \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --ca-root "${PROJECT}/external/Causal-Adapter" \
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
  --base-model lambdalabs/miniSD-diffusers \
  --joint-config configs/pendulum_label_50k_mapper1e3_noctc_fromscratch.yaml \
  --joint-root /work/nvme/bhje/hyu29/ckpts/pendulum/label_50k_mapper1e3_noctc_fromscratch/minisd_label_50k_mapper1e3_noctc_seed42/final \
  --max-samples 256 --grid-samples 8 --steps 100 --guidance-scale 1 --seed 42 \
  --output-dir /work/nvme/bhje/hyu29/evals/pendulum/label_50k_mapper1e3_noctc_seed42_n256_ddim100
