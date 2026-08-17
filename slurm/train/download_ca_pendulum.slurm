#!/usr/bin/env bash
#SBATCH --job-name=pend_ca_weights
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x-%j.out

set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u

export HF_HOME=/scratch/gilbreth/yu1331/huggingface
DEST=/scratch/gilbreth/yu1331/models/Causal-Adapter/pendulum
mkdir -p "${DEST}"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='LeiTong/Causal-Adapter',
    local_dir='/scratch/gilbreth/yu1331/models/Causal-Adapter',
    allow_patterns=['pendulum/*'],
)
PY
find "${DEST}" -type f -maxdepth 4 -print
