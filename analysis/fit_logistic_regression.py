"""
Fit logistic regression P(father_binary | z) using statsmodels for p-values.
Saves regression summary, coefficients, and p-values.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

try:
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit logistic regression for latent selection.")
    parser.add_argument("--config", type=str, required=True, help="Path to logistic config YAML.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_father_binary(thickness: np.ndarray, mode: str = "median", value: float = None) -> np.ndarray:
    """
    Binary label: 1 if thickness > threshold, else 0.

    Args:
        thickness: Array of thickness values.
        mode: 'median', 'quantile', or 'fixed'.
        value: For quantile, value in (0,1). For fixed, numeric threshold.

    Returns:
        Binary array and the threshold used.
    """
    if mode == "median":
        thresh = np.median(thickness)
    elif mode == "quantile":
        thresh = np.quantile(thickness, value)
    elif mode == "fixed":
        thresh = value
    else:
        raise ValueError(f"Unknown mode: {mode}")
    binary = (thickness > thresh).astype(np.int64)
    return binary, float(thresh)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    latents_dir = output_dir / "latents"
    npz_path = latents_dir / "train_latents.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Run extract_latents first. Expected {npz_path}")

    data = np.load(npz_path)
    z = data["z"]
    thickness = data["thickness"]

    father_name = cfg.get("father_name", "thickness")
    mode = cfg.get("father_threshold_mode", "median")
    value = cfg.get("father_threshold_value")
    if mode == "quantile" and value is None:
        value = 0.5

    father_binary, threshold = get_father_binary(thickness, mode=mode, value=value)
    threshold_info = {
        "father_name": father_name,
        "threshold_mode": mode,
        "threshold_value": value,
        "threshold": threshold,
        "n_positive": int(father_binary.sum()),
        "n_negative": int((1 - father_binary).sum()),
    }

    # Optional scaling for stability
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    n_dims = z.shape[1]
    coefficients = np.zeros(n_dims)
    pvalues = np.ones(n_dims)
    use_sklearn_fallback = False

    if HAS_STATSMODELS:
        try:
            model = Logit(father_binary, z_scaled)
            result = model.fit(disp=0)
            coefficients = result.params
            pvalues = result.pvalues
            summary_str = result.summary().as_text()
        except PerfectSeparationError:
            print("Perfect separation detected; using regularized sklearn fallback.")
            use_sklearn_fallback = True
        except Exception as e:
            print(f"statsmodels failed: {e}; using sklearn fallback.")
            use_sklearn_fallback = True

    if not HAS_STATSMODELS or use_sklearn_fallback:
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(z_scaled, father_binary)
        coefficients = lr.coef_.ravel()
        # No p-values from sklearn; leave pvalues as ones so no dims selected by p-value
        summary_str = "Logistic regression fit with sklearn (no p-values)."

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "regression_summary.txt").write_text(summary_str)
    pd.DataFrame({"coef": coefficients}).to_csv(output_dir / "coefficients.csv", index=True)
    pd.DataFrame({"pvalue": pvalues}).to_csv(output_dir / "pvalues.csv", index=True)
    (output_dir / "threshold_info.json").write_text(json.dumps(threshold_info, indent=2))

    print(f"Saved regression summary, coefficients, pvalues, threshold_info to {output_dir}")


if __name__ == "__main__":
    main()
