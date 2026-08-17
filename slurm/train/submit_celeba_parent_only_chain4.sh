#!/usr/bin/env bash
# Submit **four** Slurm jobs for CelebA **parent_only** diffusion training, chained with
# ``afterany`` so the next starts whenever the previous ends (success or fail — avoids deadlock).
#
# Training resumes from ``diffusion_last.pt`` each time until ``epochs`` (100) in the YAML.
# Use when walltime may stop a single job short of 100 epochs.
#
# Usage (from repo root):
#   cd /home/yu1331/cf-diffusion-celeba
#   bash slurm/train/submit_celeba_parent_only_chain4.sh
#
# Optional: start the chain only after an existing job finishes:
#   DEP_AFTER=10706891 bash slurm/train/submit_celeba_parent_only_chain4.sh
#
# Override config:
#   CFG=configs/diffusion_celeba_parent_only.yaml bash slurm/train/submit_celeba_parent_only_chain4.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOB_REL="${JOB_SCRIPT_REL:-slurm/train/train_diffusion_celeba_parent_only.slurm}"
JOB="${REPO}/${JOB_REL}"

if [[ ! -f "${JOB}" ]]; then
  echo "ERROR: missing ${JOB}" >&2
  exit 1
fi

cd "${REPO}"

deps () {
  local dep="$1"
  if [[ -z "${dep}" ]]; then
    sbatch --parsable "${JOB}"
  else
    sbatch --parsable --dependency="afterany:${dep}" "${JOB}"
  fi
}

if [[ -n "${DEP_AFTER:-}" ]]; then
  echo "First job depends on afterany:${DEP_AFTER}"
  J="$(deps "${DEP_AFTER}")"
else
  echo "First job (no dependency)"
  J="$(deps "")"
fi
echo "  submitted ${J}"

for i in 2 3 4; do
  J="$(deps "${J}")"
  echo "chain ${i}: submitted ${J} (afterany previous)"
done

echo "Done. Monitor: squeue -u \"\$USER\" -n celeba_diff_po"
