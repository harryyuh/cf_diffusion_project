#!/bin/bash
#SBATCH --job-name=pend_null_corr
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "${PROJECT}"
python pendulum_null_edit_correlation.py \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --ca-root "${PROJECT}/external/Causal-Adapter" \
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --checkpoint-dir "${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/checkpoints/pendulum/trained_classifiers" \
  --joint-config configs/joint_label_lora_pendulum_equal_budget_official_protocol.yaml \
  --joint-root /work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed42/final \
  --cf-root /work/nvme/bhje/hyu29/evals/pendulum/label20k_seed42_n256_ddim100/joint_label \
  --output-dir /work/nvme/bhje/hyu29/evals/pendulum/null_edit_correlation/label20k_seed42_ddim100 \
  --steps 100 --batch-size 4 --bootstrap 2000
