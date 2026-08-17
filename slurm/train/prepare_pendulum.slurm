#!/usr/bin/env bash
#SBATCH --job-name=pend_data
#SBATCH --partition=a30
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/yu1331/cf_diffusion_tmp/logs/%x-%j.out

set -eo pipefail
PROJECT=/home/yu1331/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
DATA_PARENT=/scratch/gilbreth/yu1331/datasets

mkdir -p "${DATA_PARENT}" /scratch/gilbreth/yu1331/cf_diffusion_tmp/logs
source /etc/profile.d/modules.sh 2>/dev/null || true
module load external
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cf_vae_diff
set -u
LOCAL_ROOT="${TMPDIR:-/tmp}/pendulum"
rm -rf "${LOCAL_ROOT}"
python "${PROJECT}/training/generate_pendulum_parallel.py" --output "${LOCAL_ROOT}" --workers 12
rm -rf "${DATA_PARENT}/pendulum"
cp -R "${LOCAL_ROOT}" "${DATA_PARENT}/pendulum"
python - <<'PY'
from pathlib import Path
root = Path('/scratch/gilbreth/yu1331/datasets/pendulum')
for split in ('train', 'test'):
    paths = sorted((root / split).glob('*.png'))
    print(split, len(paths), paths[0].name if paths else None, paths[-1].name if paths else None)
    if not paths:
        raise SystemExit(f'missing {split} data')
PY
