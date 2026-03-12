# Counterfactual Thickness Editing on MorphoMNIST with VAE + Conditional Diffusion

This project implements a two-stage counterfactual image generation pipeline on MorphoMNIST digits, focusing on controlled manipulation of **thickness** while preserving digit identity and other non-target attributes.

The core idea:

1. **Abduction**: Train a convolutional Beta-VAE on MorphoMNIST images and use its encoder to obtain latent codes \(z\).
2. **Latent Decomposition**:
   - Identify **thickness-related** latent dimensions (`z_father`) using logistic regression and p-values.
   - Define the complement dimensions as `z_rest`.
3. **Action**: Intervene on the **continuous thickness variable** \(f\) while conditioning on `z_rest`.
4. **Prediction**: Train a conditional DDPM-style diffusion model as a **decoder**:
   \[
   p_\theta(x \mid z_{\text{rest}}, f)
   \]
   and use it to generate counterfactual images where thickness changes but digit identity and style are preserved.

Crucially, the **VAE and diffusion are not trained jointly**:
- First, train the VAE.
- Then **freeze the encoder**.
- Then train the conditional diffusion model using frozen encoder latents as conditions.

---

## Method Overview

### Stage 1: Train VAE

- Train a convolutional Beta-VAE on 28×28 grayscale MorphoMNIST images.
- The VAE learns:
  - Encoder: \(x \rightarrow (\mu, \log \sigma^2)\)
  - Latent: \(z = \mu + \sigma \odot \epsilon\)
  - Decoder: \(z \rightarrow \hat{x}\)
- Loss:
  \[
  \mathcal{L} = \text{ReconLoss}(x, \hat{x}) + \beta \cdot \text{KL}(q(z \mid x) \| p(z))
  \]

We save encoder and decoder checkpoints and (optionally) latent statistics.

### Stage 2: Logistic Regression for Latent Selection

- Freeze the trained VAE encoder.
- Encode all training images to latent means \(z\).
- Let `father_name = "thickness"`.
- Construct a **binary father label**:
  - `father_binary = 1` if `thickness > median(train_thickness)`, else `0` (configurable thresholding).
- Fit a **logistic regression**:
  \[
  P(\text{father\_binary} = 1 \mid z)
  \]
- Using statsmodels, extract:
  - Regression summary.
  - Coefficients per latent dimension.
  - p-values for each coefficient.
- Define:
  - `father_dims` = indices where p-value \< `pvalue_threshold` (default 0.05).
  - `rest_dims` = complement of `father_dims`.
- Save:
  - `regression_summary.txt`
  - `coefficients.csv`
  - `pvalues.csv`
  - `father_dims.json`
  - `rest_dims.json`
  - `threshold_info.json`

### Stage 3: Train Conditional Diffusion Decoder

We train a simple DDPM-like diffusion model as a **high-quality conditional decoder**:

- Condition on:
  - `z_rest` (non-thickness latent dimensions).
  - Continuous thickness scalar \(f\).
- Build condition vector: `cond = concat(z_rest, f)`.
- A small MLP encodes `cond` into an embedding that modulates a U-Net noise predictor.
- The model predicts Gaussian noise in the forward diffusion process:
  \[
  \epsilon_\theta(x_t, t, \text{cond})
  \]
- Training loop per batch:
  1. Freeze VAE encoder.
  2. Encode `x` with encoder to get `z`.
  3. Extract `z_rest` using `rest_dims`.
  4. Form `cond = concat(z_rest, thickness)`.
  5. Sample timestep `t` and noise `\epsilon`.
  6. Compute `x_t` via the diffusion forward process.
  7. Train U-Net to predict `\epsilon` with MSE loss.

We **never** update the VAE encoder during diffusion training, and we **do not** use the VAE decoder in this stage.

### Stage 4: Counterfactual Inference

Given an image \(x\) and target thickness value \(f_{\text{target}}\):

1. Abduction:
   - Encode \(x\) with frozen VAE encoder to get latent mean \(z\).
   - Split into `z_father` and `z_rest` using learned indices.
2. Action:
   - Replace the thickness variable with `f_target`.
   - Build `cond = concat(z_rest, f_target)`.
3. Prediction:
   - Sample a counterfactual image:
     \[
     x_{\text{cf}} \sim p_\theta(x \mid z_{\text{rest}}, f_{\text{target}})
     \]
4. Optionally, decode \(z\) with the VAE decoder to visualize the reconstruction baseline.
5. Save side-by-side outputs:
   - Original
   - VAE reconstruction
   - One or more counterfactual edits at different target thickness values.

---

## Project Structure

project_root/
  README.md
  requirements.txt
  setup.py

  configs/
    vae.yaml
    logistic.yaml
    diffusion.yaml
    inference.yaml

  data/
    __init__.py
    morphomnist_dataset.py

  models/
    __init__.py
    vae.py
    diffusion_unet.py
    condition_mlp.py

  training/
    __init__.py
    train_vae.py
    train_diffusion.py

  analysis/
    __init__.py
    extract_latents.py
    fit_logistic_regression.py
    select_latent_dims.py

  inference/
    __init__.py
    counterfactual_edit.py

  utils/
    __init__.py
    seed.py
    logger.py
    checkpoint.py
    visualization.py
    diffusion_utils.py
    latent_utils.py
    metrics.py

  scripts/
    run_train_vae.sh
    run_extract_latents.sh
    run_fit_logistic.sh
    run_train_diffusion.sh
    run_counterfactual_edit.sh