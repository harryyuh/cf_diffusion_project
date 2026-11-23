#!/bin/bash
#SBATCH --job-name=lbl40_hi_eval
#SBATCH --partition=ghx4
#SBATCH --account=bhje-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/work/nvme/bhje/hyu29/logs/%x.%A_%a.out
set -eo pipefail
source /work/nvme/bhje/hyu29/venvs/cf-celeba/bin/activate
PROJECT=/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba
CA_ROOT=${PROJECT}/external/Causal-Adapter
PLAN_ROOT=/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100/plans
OUT=/work/nvme/bhje/hyu29/evaluations/celeba/label_bs40_highlr_20260930_n512_ddim100
CKPTS=/work/nvme/bhje/hyu29/ckpts/celeba/label_bs40_highlr_20260930/minisd_label_bs40_mapper1e3_50k/checkpoints
SCM=/work/nvme/bhje/hyu29/ckpts/celeba/causal_adapter_scm/best_model.pt
export PYTHONPATH=${CA_ROOT}/diffusers/src:${CA_ROOT}/notebook_benchmarks:${CA_ROOT}:${PROJECT}:${PYTHONPATH:-}
export HF_HOME=/work/nvme/bhje/hyu29/huggingface
cd "$PROJECT"
STEPS=(5000 10000 20000 50000)
ATTRS=(Young Male No_Beard Bald)
PLANS=(0 20 40 61)
STEP_ID=$((SLURM_ARRAY_TASK_ID / 4))
ATTR_ID=$((SLURM_ARRAY_TASK_ID % 4))
STEP=${STEPS[$STEP_ID]}
ATTR=${ATTRS[$ATTR_ID]}
PLAN=${PLANS[$ATTR_ID]}
PADDED=$(printf "%06d" "$STEP")
SEL=$(printf "%s/plan_%03d.json" "$PLAN_ROOT" "$PLAN")
RUN_OUT=$OUT/step_$PADDED/do_$ATTR
mkdir -p "$RUN_OUT/tensors"
python -m evaluation.evaluate_label200k_smoke \
 --config configs/label_curve_bs40_highlr_50k.yaml \
 --lora-dir "$CKPTS/step_$PADDED/lora" --pai-checkpoint "$CKPTS/step_$PADDED/mapper/pai_mapper.pt" \
 --selection-json "$SEL" --split test --max-samples 128 --batch-size 4 \
 --steps 100 --guidance-scale 3 --invert-guidance-scale 1 \
 --negative-prompt "ugly, blurry, low res, unrealistic" \
 --scheduler-mode ca --vae-latent-mode mean --condition-mode joint_label \
 --target-scm causal_adapter --causal-adapter-root "$CA_ROOT" --causal-adapter-scm "$SCM" \
 --benchmark-root "$CA_ROOT/counterfactual-benchmark/counterfactual_benchmark" \
 --intervention-attr "$ATTR" --editing standard --inversion ddim \
 --save-fake-dir "$RUN_OUT/tensors" --output-json "$RUN_OUT/metrics.json"
