#!/usr/bin/env bash
#SBATCH --job-name=cel_match_validate
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%j.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
python - <<'PY'
from pathlib import Path
import torch

root = Path('/work/nvme/bhje/hyu29/evaluations/celeba/matched_50k_bs40_fulltest_ddim100_smoke')
attrs = ('Young', 'Male', 'No_Beard', 'Bald')
for plan in (0, 20, 40, 61):
    found = {}
    for method in ('ca', 'label_only'):
        files = sorted((root / method).glob(f'plan_{plan:03d}_*/tensors/batch_*.pt'))
        if not files:
            raise RuntimeError(f'Missing {method} plan {plan}')
        chunks = [torch.load(p, map_location='cpu', weights_only=False) for p in files]
        found[method] = {
            'indices': torch.cat([torch.as_tensor(c['source_indices']) for c in chunks]),
            'targets': {a: torch.cat([torch.as_tensor(c['targets'][a]) for c in chunks]) for a in attrs},
        }
    a, b = found['ca'], found['label_only']
    if not torch.equal(a['indices'], b['indices']):
        raise RuntimeError(f'Source mismatch plan {plan}')
    for attr in attrs:
        if not torch.equal(a['targets'][attr] > 0, b['targets'][attr] > 0):
            raise RuntimeError(f'Target mismatch plan {plan} attr {attr}')
    print(f'plan {plan}: matched {len(a["indices"])} sources and binary targets')
PY
