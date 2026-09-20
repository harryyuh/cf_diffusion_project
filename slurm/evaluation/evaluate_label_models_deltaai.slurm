#!/bin/bash
#SBATCH --job-name=label_models_eval
#SBATCH --account=bhje-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x-%j.out
#SBATCH --error=/work/nvme/bhje/hyu29/logs/%x-%j.err

set -euo pipefail
cd /work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
export PYTHONPATH="$PWD:$PWD/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark:${PYTHONPATH:-}"

python evaluation/evaluate_label_models.py \
  --celeba-root /work/nvme/bhje/hyu29/datasets/celebA \
  --pendulum-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --benchmark-root "$PWD/external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark" \
  --ca-root "$PWD/external/Causal-Adapter" \
  --celeba-scm /work/nvme/bhje/hyu29/ckpts/celeba/full_label_scm/causal_adapter_graph/checkpoints/full_label_scm_best.pt \
  --pendulum-scm /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt \
  --pendulum-regressor /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt \
  --output /work/nvme/bhje/hyu29/evaluations/label_models/heldout_calibration.json
