"""Quadratic objective construction and constrained solver."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from sklearn.metrics import mutual_info_score

from ._linear import ols_with_intercept

SimilarityMode = Literal["correl", "mi"]
RelevanceMode = Literal["correl", "mi", "signif"]


def _discretize(x: np.ndarray, bins: int = 16) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if np.allclose(x, x[0]):
        return np.zeros_like(x, dtype=int)
    edges = np.histogram_bin_edges(x, bins=bins)
    # np.digitize returns 1..len(edges); use 0-based.
    return np.digitize(x, edges[1:-1], right=False)


def _information(x: np.ndarray, y: np.ndarray) -> float:
    x_disc = _discretize(x)
    y_disc = _discretize(y)
    return float(mutual_info_score(x_disc, y_disc))


def _corr_with_target(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_centered = y - np.mean(y)
    y_norm = np.linalg.norm(y_centered)
    out = np.zeros(x.shape[1], dtype=float)
    for i in range(x.shape[1]):
        col = x[:, i]
        col_centered = col - np.mean(col)
        denom = np.linalg.norm(col_centered) * y_norm
        if denom == 0:
            out[i] = 0.0
        else:
            out[i] = np.dot(col_centered, y_centered) / denom
    return out


def create_opt_problem(
    x: np.ndarray,
    y: np.ndarray,
    sim: SimilarityMode = "correl",
    rel: RelevanceMode = "correl",
) -> tuple[np.ndarray, np.ndarray]:
    """Port of MATLAB CreateOptProblem.m."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if sim == "correl":
        q = np.corrcoef(x, rowvar=False)
        q = np.nan_to_num(q, nan=0.0)
    elif sim == "mi":
        q = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        for i in range(q.shape[0]):
            for j in range(i, q.shape[1]):
                q[i, j] = _information(x[:, i], x[:, j])
        q = q + q.T - np.diag(np.diag(q))
        lambdas = np.linalg.eigvalsh(q)
        min_lambda = float(np.min(lambdas))
        if min_lambda < 0.0:
            q = q - min_lambda * np.eye(q.shape[0])
    else:
        raise ValueError(f"Unsupported similarity mode: {sim}")

    if rel == "correl":
        b = np.abs(_corr_with_target(x, y))
        return q, b
    if rel == "mi":
        b = np.zeros(x.shape[1], dtype=float)
        for i in range(x.shape[1]):
            b[i] = _information(y, x[:, i])
        return q, b
    if rel == "signif":
        fit = ols_with_intercept(x, y)
        p_val = np.asarray(fit["p_values"], dtype=float)[1:]
        beta = np.asarray(fit["beta"], dtype=float)[1:]
        idx_zero_coeff = np.where(np.abs(beta) < 1e-7)[0]
        p_val = np.nan_to_num(p_val, nan=1.0, posinf=1.0, neginf=1.0)
        denom = np.sum(p_val)
        if denom == 0:
            b = np.zeros_like(p_val)
        else:
            b = 1.0 - p_val / denom
        b[idx_zero_coeff] = 0.0
        return q, b
    raise ValueError(f"Unsupported relevance mode: {rel}")


def solve_opt_problem(
    q: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    maxiter: int = 2_000,
    ftol: float = 1e-12,
) -> np.ndarray:
    """Port of MATLAB SolveOptProblem.m using SciPy constrained optimization."""
    q = np.asarray(q, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = q.shape[0]
    if x0 is None:
        x0 = np.full(n, 1.0 / n)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(-1)

    def obj(x: np.ndarray) -> float:
        return float(x.T @ q @ x - b @ x)

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * (q @ x) - b

    bounds = [(0.0, None)] * n
    lin_con = LinearConstraint(np.ones((1, n)), -np.inf, 1.0)
    res = minimize(
        obj,
        x0,
        method="SLSQP",
        jac=grad,
        bounds=bounds,
        constraints=[lin_con],
        options={"maxiter": maxiter, "ftol": ftol},
    )
    if not res.success:
        raise RuntimeError(f"QP optimization failed: {res.message}")
    return np.asarray(res.x, dtype=float)
