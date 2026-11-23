#!/bin/sh
# This script was created by sbatch --wrap.

source /etc/profile.d/modules.sh; module load python/miniforge3_pytorch/2.10.0 && source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate && export HF_HOME=/work/nvme/bhje/hyu29/huggingface && python -u /work/nvme/bhje/hyu29/evaluations/paired_bg_smoke.py
