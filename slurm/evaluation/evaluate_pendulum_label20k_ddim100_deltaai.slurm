#!/bin/bash
#SBATCH --job-name=label20k100
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
cd /work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=$PWD/external/Causal-Adapter/diffusers/src:$PWD:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
python evaluation/diagnose_pendulum_label_20k_50k.py
python -m evaluation.evaluate_pendulum --method joint_label \
 --data-root /work/nvme/bhje/hyu29/datasets/pendulum --ca-root "$PWD/external/Causal-Adapter" \
 --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
 --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
 --joint-config configs/joint_label_lora_pendulum_equal_budget_official_protocol.yaml \
 --joint-root /work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed42/checkpoints/step_020000 \
 --max-samples 256 --grid-samples 8 --steps 100 --guidance-scale 1 --seed 42 \
 --output-dir /work/nvme/bhje/hyu29/evals/pendulum/label20k_seed42_n256_ddim100
