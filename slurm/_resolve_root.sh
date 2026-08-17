#!/usr/bin/env bash
# Sourced by slurm/**/*.slurm — do not run standalone.
# Prefer SLURM_SUBMIT_DIR when it is the repo root (has training/). Otherwise resolve from this
# file's location: _resolve_root.sh lives in slurm/, so repo root is the parent of that directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/training" ]]; then
  export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export PROJECT_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "${PROJECT_ROOT}" || {
  echo "ERROR: cd failed PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
}
if [[ ! -d "${PROJECT_ROOT}/training" ]]; then
  echo "ERROR: PROJECT_ROOT=${PROJECT_ROOT} is not cf-diffusion-celeba (missing training/)." >&2
  echo "Submit from the repo root, e.g.:  cd .../cf-diffusion-celeba && sbatch slurm/train/....slurm" >&2
  exit 1
fi
