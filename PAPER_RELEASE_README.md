# Counterfactual image editing: paper reproduction code

This branch is a curated source snapshot for the CelebA and Pendulum experiments. It keeps the label-only/LoRA training code, causal-label/SCM modules, CA comparison evaluators, current full-test merging, ablations, profiling, and the corresponding cluster job scripts. Exploratory DiT, PAI-only, label-FiLM, and obsolete sweep entrypoints are omitted from this branch; they remain recoverable in the `celeba-experiments` Git history.

The archive contains **source code and configuration only**. It does not contain CelebA or Pendulum image datasets, pretrained diffusion models, CA's third-party repository, learned checkpoints, generated grids, or cluster-specific runtime environments. These assets must be supplied separately. Some Slurm scripts contain the original DeltaAI/Gilbreth absolute paths; adapt them before running elsewhere.

The release snapshot includes evaluator fixes made during the October 2026 paper audit, including explicit FP32 selection, saved-logit F1 checks, and separate CelebA CLD and null-edit LPIPS workflows. A script's inclusion does not imply its experiment completed or that its output was validated; consult the experiment audit and verify checkpoint, sample split, intervention values, precision, and metric aggregation before reporting numbers.

Core locations:

- `training/`: label-only/LoRA and Pendulum training entrypoints.
- `models/`, `data/`, `utils/`: conditioning, SCM, datasets, and shared utilities.
- `evaluation/`: CelebA and Pendulum evaluators and result mergers.
- `configs/`: configurations retained for paper comparisons.
- `slurm/`, `audit_fix/`: corresponding cluster workflows and protocol checks.

This branch is deliberately separate from the checkout used by running Slurm jobs. Cleaning the release does not change those jobs or their input files.
