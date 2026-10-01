#!/bin/bash
#SBATCH --job-name=pend_noscm_reval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"

python -m evaluation.reevaluate_pendulum_saved_ca_regressors \
  --input-dir /work/nvme/bhje/hyu29/evals/pendulum/equal_budget_official_protocol_seed42_n256/label_only/joint_label \
  --input-dir /work/nvme/bhje/hyu29/evals/pendulum/label_ablation_seed42_n256_ddim100/rank16_no_scm/joint_label \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --ca-root "${PROJECT}/external/Causal-Adapter" \
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --checkpoint-dir "${PROJECT}/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/checkpoints/pendulum/trained_classifiers" \
  --batch-size 128
