"""HSIC (Hilbert–Schmidt Independence Criterion) with RBF kernels for independence regularization."""

from __future__ import annotations

import torch


def _pairwise_sq_l2(X: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean pairwise distances, shape (n, n)."""
    return torch.cdist(X, X, p=2).pow(2)


def _rbf_from_sq_dists(sq_dists: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    gamma = 0.5 / (sigma * sigma + 1e-12)
    return torch.exp(-sq_dists * gamma)


def _median_sigma(X: torch.Tensor) -> torch.Tensor:
    """Median heuristic: scale from within-batch pairwise distances."""
    with torch.no_grad():
        # pdist avoids full n×n when possible; fallback for small batches
        if X.size(0) <= 1:
            return torch.tensor(1.0, device=X.device, dtype=X.dtype)
        d = torch.pdist(X, p=2)
        med = d.median()
        if not torch.isfinite(med) or med < 1e-8:
            med = torch.tensor(1e-3, device=X.device, dtype=X.dtype)
        return med.detach()


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Doubly center a Gram matrix (H K H)."""
    n = K.size(0)
    one_n = torch.ones(n, n, device=K.device, dtype=K.dtype) / float(n)
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def hsic_rbf(X: torch.Tensor, Y: torch.Tensor, sigma_x_scale: float = 1.0, sigma_y_scale: float = 1.0) -> torch.Tensor:
    """
    Biased HSIC estimate with Gaussian RBF kernels (median bandwidth per variable block).

    Args:
        X: (n, d1), Y: (n, d2) — same batch size.
        sigma_*_scale: multiply median heuristic bandwidth (optional tuning).

    Returns:
        Scalar HSIC ≥ 0; independent ⇒ ~0 (finite sample noise remains).
    """
    n = X.size(0)
    if n < 2:
        return torch.zeros((), device=X.device, dtype=X.dtype)

    sx = _median_sigma(X) * float(sigma_x_scale)
    sy = _median_sigma(Y) * float(sigma_y_scale)

    dist_x = _pairwise_sq_l2(X)
    dist_y = _pairwise_sq_l2(Y)

    K = _rbf_from_sq_dists(dist_x, sx)
    L = _rbf_from_sq_dists(dist_y, sy)

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    # Standard biased HSIC / n^2 formulation
    hs = torch.trace(Kc @ Lc) / (float(n * n) + 1e-12)
    return hs.clamp(min=0.0)
