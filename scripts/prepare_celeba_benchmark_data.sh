#!/usr/bin/env bash
# Symlink scratch CelebA files into torchvision layout: $ROOT/celeba/{img_align_celeba,list_*.txt,...}
# Required by counterfactual-benchmark datasets.celeba.dataset.Celeba.
set -euo pipefail

ROOT="${1:-/scratch/gilbreth/yu1331/datasets/celebA}"
DEST="${ROOT}/celeba"
mkdir -p "${DEST}"

link_if_missing() {
  local name="$1"
  if [[ -e "${DEST}/${name}" ]]; then
    return 0
  fi
  if [[ -e "${ROOT}/${name}" ]]; then
    ln -sfn "../${name}" "${DEST}/${name}"
    echo "linked ${name}"
  else
    echo "WARN: missing ${ROOT}/${name}" >&2
  fi
}

link_if_missing img_align_celeba
for f in list_attr_celeba.txt list_bbox_celeba.txt list_eval_partition.txt \
  list_landmarks_align_celeba.txt identity_CelebA.txt; do
  link_if_missing "${f}"
done

echo "Done. Test:"
echo "  python -c \"from torchvision.datasets import CelebA; print(len(CelebA('${ROOT}', split='train', download=False)))\""
