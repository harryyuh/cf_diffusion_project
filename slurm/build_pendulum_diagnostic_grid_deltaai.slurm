#!/bin/bash
#SBATCH --job-name=pend_diag_grid
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
export PYTHONPATH=${PROJECT}/external/Causal-Adapter/diffusers/src:${PROJECT}:${PYTHONPATH:-}
cd "${PROJECT}"
python -m evaluation.build_diagnostic_matched_grid \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --ca-root "${PROJECT}/external/Causal-Adapter" \
  --ca-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
  --method ca /work/nvme/bhje/hyu29/evals/pendulum/ca_author_official20k_seed42_n256_ddim50/ca/ca \
  --method pai /work/nvme/bhje/hyu29/evals/pendulum/equal_budget_official_protocol_seed42_n256/pai/pai \
  --method joint_label /work/nvme/bhje/hyu29/evals/pendulum/equal_budget_official_protocol_seed42_n256/label_only/joint_label \
  --reference-a ca --reference-b joint_label \
  --top 6 --counterexamples 2 \
  --output-dir /work/nvme/bhje/hyu29/evals/pendulum/diagnostic_matched_target_prediction
