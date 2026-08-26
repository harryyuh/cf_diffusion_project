#!/usr/bin/env bash
#SBATCH --job-name=pend_eval_assets
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x-%j.out

set -eo pipefail
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
VENV=/work/nvme/bhje/hyu29/venvs/cf-celeba
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source "${VENV}/bin/activate"
mkdir -p /work/nvme/bhje/hyu29/models/Causal-Adapter /work/nvme/bhje/hyu29/logs

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='LeiTong/Causal-Adapter', local_dir='/work/nvme/bhje/hyu29/models/Causal-Adapter', allow_patterns=['pendulum/*'])"

python -m training.train_pendulum_regressor \
  --data-root /work/nvme/bhje/hyu29/datasets/pendulum \
  --output /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt

test -f /work/nvme/bhje/hyu29/models/Causal-Adapter/pendulum/scm/best_model.pt
test -f /work/nvme/bhje/hyu29/ckpts/pendulum/regressor/best.pt
