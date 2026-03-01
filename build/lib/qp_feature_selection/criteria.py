"""Criteria functions ported from MATLAB criteria/*.m."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
from scipy import stats

from ._linear import residual_sum_squares


def complexity(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    return float(x.shape[1])


def cp(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    # Keep MATLAB behavior: p = size(w,2), which is 1 for column vectors.
    p = float(w.shape[1]) if np.asarray(w).ndim == 2 else 1.0
    m = float(x.shape[0])
    rss_all = float(par["rss"])
    return residual_sum_squares(x, y, w) / rss_all - m + 2.0 * p


def rss(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    if x.size == 0:
        return float("inf")
    return residual_sum_squares(x, y, w)


def cond_number(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    if x.size == 0:
        return float("inf")
    a = x.T @ x
    d = np.linalg.eigvalsh(a)
    d_abs = np.abs(d)
    d_min = np.min(d_abs)
    d_max = np.max(d_abs)
    if d_min == 0.0:
        return float("inf")
    return float(np.log(d_max / d_min))


def vif(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    if x.size == 0:
        return float("inf")
    _, m = x.shape
    vif_list = np.zeros(m, dtype=float)
    for j in range(m):
        x_j = x[:, j]
        x_without_j = x[:, np.arange(m) != j]
        if x_without_j.shape[1] == 0:
            x_hat_j = np.zeros_like(x_j)
        else:
            lhs = x_without_j.T @ x_without_j
            rhs = x_without_j.T @ x_j
            regr_j = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            x_hat_j = x_without_j @ regr_j
        denom = np.sum((x_hat_j - x_j) ** 2)
        if denom == 0:
            vif_list[j] = float("inf")
        else:
            vif_list[j] = np.sum((x_j - np.mean(x_j)) ** 2) / denom
    return float(np.max(vif_list))


def bic(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    if x.size == 0:
        return float("inf")
    return residual_sum_squares(x, y, w) + x.shape[1] * np.log(x.shape[0])


def ftest(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    if x.size == 0:
        return float("inf")
    rss_val = residual_sum_squares(x, y, w)
    p = int(np.sum(np.asarray(w) != 0))
    tss = float(np.sum((np.mean(y) - y) ** 2))
    if rss_val > tss or p <= 0:
        return float("inf")
    n = x.shape[0]
    denom_df = n - p - 1
    if denom_df <= 0:
        return float("inf")
    f_stat = (abs(tss - rss_val) / p) / (rss_val / denom_df)
    if f_stat == 0:
        return 0.0
    left = stats.f.cdf(f_stat, p, denom_df)
    right = stats.f.cdf(1.0 / f_stat, denom_df, p)
    return float(2.0 * min(left, right))


def rsq_adj(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    if x.size == 0:
        return float("inf")
    m = x.shape[0]
    p = int(np.sum(np.asarray(w) != 0))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rss_val = residual_sum_squares(x, y, w)
    if m == p + 1:
        return 1.0 - (rss_val / np.finfo(float).eps) / (tss / (m - 1))
    return 1.0 - (rss_val / (m - p - 1)) / (tss / (m - 1))


def my_collintest(x: np.ndarray) -> np.ndarray:
    num_obs, num_vars = x.shape
    col_norms = np.sqrt(np.sum(x**2, axis=0))
    col_norms[col_norms == 0.0] = 1.0
    x_s = x / col_norms
    _, s_vals, v_t = np.linalg.svd(x_s, full_matrices=False)
    v = v_t.T
    s_vals = np.asarray(s_vals, dtype=float)
    s_vals[s_vals < np.finfo(float).eps] = np.finfo(float).eps
    phi = (v**2) / (s_vals**2)
    phi_sum = np.sum(phi, axis=1, keepdims=True)
    var_decomp = (phi.T) / phi_sum.T
    if var_decomp.shape != (num_vars, num_vars):
        # In tall/skinny cases this should already be square, but keep
        # behavior predictable if rank deficiency changes dimensions.
        pad = np.zeros((num_vars, num_vars))
        r = min(num_vars, var_decomp.shape[0])
        c = min(num_vars, var_decomp.shape[1])
        pad[:r, :c] = var_decomp[:r, :c]
        return pad
    return var_decomp


def _alg_belsley(x: np.ndarray, idx_features: np.ndarray) -> int:
    var_decomp = my_collintest(x[:, idx_features])
    idx_max_var_prop = int(np.argmax(var_decomp[-1, :]))
    return int(idx_features[idx_max_var_prop])


def stability(x: np.ndarray, y: np.ndarray, w: np.ndarray, par: Mapping[str, object]) -> float:
    if x.size == 0:
        return float("inf")
    s_0 = float(par["s_0"])
    x_unnorm = np.asarray(par["X_unnorm"], dtype=float)
    idx_all = np.arange(np.asarray(w).shape[0], dtype=int)
    d = 0
    s = residual_sum_squares(x, y, w)
    w_flat = np.asarray(w, dtype=float).reshape(-1)
    while (s < s_0) and (idx_all.size > 1):
        idx_del = _alg_belsley(x_unnorm, idx_all)
        idx_all = idx_all[idx_all != idx_del]
        d += 1
        s = float(np.sum((y - x[:, idx_all] @ w_flat[idx_all]) ** 2))
    return float(d)


CRITERIA: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]], float | np.ndarray]] = {
    "complexity": complexity,
    "Cp": cp,
    "RSS": rss,
    "CondNumber": cond_number,
    "Vif": vif,
    "bic": bic,
    "ftest": ftest,
    "Rsq_adj": rsq_adj,
    "stability": stability,
    "my_collintest": lambda x, y, w, par: my_collintest(x),
}


def run_criterion(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    par: Mapping[str, object],
) -> float | np.ndarray:
    if name not in CRITERIA:
        raise KeyError(f"Unknown criterion: {name}")
    return CRITERIA[name](x, y, w, par)
