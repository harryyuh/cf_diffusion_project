#!/bin/bash
#SBATCH --job-name=cf_diffusion
#SBATCH --nodes=1
#SBATCH --partition=a100-40gb
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=inference.%j.out
#SBATCH --mail-user=yu1331@purdue.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

set -eo pipefail

module load external
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cf_vae_diff

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

on_usr1() {
  echo "[`date`] Got SIGUSR1. Requeueing job $SLURM_JOB_ID ..."
  scontrol requeue "$SLURM_JOB_ID" || echo "WARN: requeue failed."
}
trap on_usr1 USR1

python training/train_diffusion.py --config configs/diffusion.yaml

