#!/bin/bash
#SBATCH --job-name=cf_vae_morpho
#SBATCH --nodes=1
#SBATCH --partition=a10
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=job.%j.out
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
echo "[debug] PROJECT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"

# Checkpoints and logs on scratch
export CKPT_ROOT="${SCRATCH:-$HOME}/cf_vae_morpho"
mkdir -p "$CKPT_ROOT"
mkdir -p "$CKPT_ROOT/logs"

# Redirect outputs from config into this run (optional)
export VAE_OUTPUT_DIR="$CKPT_ROOT"

on_usr1() {
  echo "[`date`] Got SIGUSR1. Requeueing job $SLURM_JOB_ID ..."
  scontrol requeue "$SLURM_JOB_ID" || echo "WARN: requeue failed."
}
trap on_usr1 USR1

python training/train_vae.py --config configs/vae.yaml

