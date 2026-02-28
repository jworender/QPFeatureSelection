"""Linear algebra helpers used across the MATLAB port."""

from __future__ import annotations

import numpy as np
from scipy import stats


def lscov(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MATLAB-like least squares without an intercept."""
    coef, *_ = np.linalg.lstsq(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rcond=None)
    return coef


def residual_sum_squares(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(x, dtype=float) @ np.asarray(w, dtype=float).reshape(-1)
    return float(np.sum((y - pred) ** 2))


def ols_with_intercept(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Fit OLS with intercept and expose coefficient p-values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = x.shape[0]
    design = np.column_stack([np.ones(m), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ beta
    rss = np.sum((y - y_hat) ** 2)
    p = design.shape[1]
    dof = m - p
    if dof <= 0:
        p_values = np.ones(p)
        stderr = np.full(p, np.nan)
    else:
        sigma2 = rss / dof
        xtx_inv = np.linalg.pinv(design.T @ design)
        var_beta = sigma2 * np.diag(xtx_inv)
        stderr = np.sqrt(np.maximum(var_beta, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.divide(beta, stderr, out=np.zeros_like(beta), where=stderr > 0)
        p_values = 2.0 * stats.t.sf(np.abs(t_stats), dof)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    return {
        "beta": beta,
        "p_values": p_values,
        "stderr": stderr,
        "rss": float(rss),
        "dof": float(dof),
    }
