#!/usr/bin/env bash
# Print hashes of config files so Slurm logs prove which YAML the job read.
# Run from repo root after _resolve_root.sh (cwd must be PROJECT_ROOT).
# Usage: bash slurm/print_config_fingerprint.sh configs/vae_scm1.yaml [more files...]
set -euo pipefail
echo "=== config fingerprint ==="
echo "date: $(date -Is 2>/dev/null || date)"
echo "PWD: $(pwd)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
fi
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
fi
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git HEAD: $(git rev-parse --short HEAD 2>/dev/null)"
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "WARN: git working tree has uncommitted or untracked changes (configs may differ from last commit)"
  fi
else
  echo "git: (not a repo or git unavailable)"
fi
for f in "$@"; do
  if [[ -f "$f" ]]; then
    sha256sum "$f"
  else
    echo "MISSING: $f" >&2
    exit 1
  fi
done
echo "=== end fingerprint ==="
