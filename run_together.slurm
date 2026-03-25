#!/bin/bash
#SBATCH --job-name=cf_vae_diffusion
#SBATCH --nodes=1
#SBATCH --partition=a100-40gb
#SBATCH --account=statdept
#SBATCH --qos=standby
#SBATCH --time=04:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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
  # Exit so this instance can finish cleanly and the requeued job restarts
  exit 0
}
trap on_usr1 USR1

# 1) Train VAE (writes to configs/vae.yaml: output_dir)
python training/train_vae.py --config configs/vae.yaml

# 2) Train diffusion using VAE checkpoint and config in configs/diffusion.yaml
python training/train_diffusion.py --config configs/diffusion.yaml

# 3) After both finish, submit inference job
sbatch run_inference.slurm
