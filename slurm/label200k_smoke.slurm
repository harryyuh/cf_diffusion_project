#!/bin/bash
#SBATCH --job-name=label200k_smoke
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --array=0-3%4
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
CKPT=/work/nvme/bhje/hyu29/ckpts/celeba/official_paper_protocol/joint_label_seed42/minisd_joint_fp32_bs40_200k/checkpoints/step_200000
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/label200k_bs40_smoke_20260929
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "$PROJECT"
ATTRS=(Young Male No_Beard Bald)
ATTR=${ATTRS[$SLURM_ARRAY_TASK_ID]}
python -m evaluation.evaluate_label200k_smoke \
 --config configs/joint_label_lora_celeba_official_paper_seed42_deltaai.yaml \
 --lora-dir "$CKPT/lora" --pai-checkpoint "$CKPT/mapper/pai_mapper.pt" \
 --selection-json "$OUT/selection.json" --split test --max-samples 32 --batch-size 4 \
 --steps 100 --guidance-scale 3 --invert-guidance-scale 1 \
 --negative-prompt "ugly, blurry, low res, unrealistic" \
 --scheduler-mode ca --vae-latent-mode mean --condition-mode joint_label \
 --editing standard --inversion ddim --target-scm causal_adapter \
 --causal-adapter-root "$CA_ROOT" \
 --causal-adapter-scm /work/nvme/bhje/hyu29/ckpts/celeba/ca_author_official_200k/best_model.pt \
 --benchmark-root "$CA_ROOT/counterfactual-benchmark/counterfactual_benchmark" \
 --intervention-attr "$ATTR" --output-grid "$OUT/grid_${ATTR}.png" \
 --save-fake-dir "$OUT/tensors_${ATTR}" --output-json "$OUT/metrics_${ATTR}.json"
