#!/usr/bin/env bash
#SBATCH --job-name=cel_null_lpips
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=0-163%8
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate

PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
BASE=/work/nvme/bhje/hyu29
PLANS=${BASE}/evaluations/celeba/ca_official_fulltest_ddim100/plans
OUT=${BASE}/evaluations/celeba/null_composition_author_ca_vs_label20k
if [[ "${SMOKE:-0}" == 1 ]]; then OUT=${OUT}_smoke; fi

TASK=${SLURM_ARRAY_TASK_ID}
if (( TASK < 82 )); then METHOD=ca; PLAN=${TASK}; else METHOD=label_only; PLAN=$((TASK-82)); fi
SEL=$(printf "%s/plan_%03d.json" "${PLANS}" "${PLAN}")
DEST=$(printf "%s/%s/plan_%03d.json" "${OUT}" "${METHOD}" "${PLAN}")
if [[ -s "${DEST}" ]]; then exit 0; fi

export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=${BASE}/huggingface
cd "${PROJECT}"

LIMIT=0
if [[ "${SMOKE:-0}" == 1 ]]; then LIMIT=2; fi
python -m evaluation.evaluate_celeba_null_lpips_fulltest \
  --method "${METHOD}" --selection-json "${SEL}" --output "${DEST}" \
  --data-root "${BASE}/datasets/celebA" --ca-root "${CA_ROOT}" \
  --ca-run "${BASE}/ckpts/celeba/ca_author_official_200k" \
  --label-config "${PROJECT}/configs/joint_label_lora_celeba_20k_bs2_seed42_eval_deltaai.yaml" \
  --label-checkpoint "${BASE}/ckpts/celeba/joint_label_lora/joint_20k_bs2/checkpoints/step_020000" \
  --batch-size 4 --limit "${LIMIT}"
