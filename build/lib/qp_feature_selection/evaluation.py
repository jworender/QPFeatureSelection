"""Evaluation helpers ported from AlgCrit.m and TestAlgCrit.m."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ._linear import lscov
from .algorithms import run_algorithm
from .criteria import run_criterion


def alg_crit(
    alg: Sequence[str],
    crit: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    parameters: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Port of MATLAB AlgCrit.m."""
    iters = int(parameters.get("iter", 1))
    threshold = float(parameters.get("threshold", 1e-10))
    mat_alg_crit = np.zeros((len(alg), len(crit)), dtype=float)
    w_store = np.zeros((x.shape[1], len(alg)), dtype=float)

    for _ in range(iters):
        x_def = np.array(x, copy=True)
        beta = lscov(x, y)
        rss_all = float(np.sum((y - x @ beta) ** 2))
        tss_all = float(np.sum((y - np.mean(y)) ** 2))
        par = {
            "s_0": parameters.get("s_0", 0.5),
            "rss": rss_all,
            "tss": tss_all,
        }
        algo_params = dict(parameters)
        algo_params["crit"] = list(crit)
        algo_params["rss"] = rss_all

        for i, alg_name in enumerate(alg):
            x_sh = np.array(x, copy=True)
            x_unnorm = np.array(x_def, copy=True)
            algo_params["X_unnorm"] = x_unnorm

            w = run_algorithm(alg_name, x, y, algo_params)
            w_store[:, i] = w

            idx_del = np.abs(w) < threshold
            w_reduced = w[~idx_del]
            w_store[idx_del, i] = 0.0
            x_sh = x_sh[:, ~idx_del]
            x_unnorm = x_unnorm[:, ~idx_del]
            par["X_unnorm"] = x_unnorm

            for j, crit_name in enumerate(crit):
                mat_alg_crit[i, j] += float(run_criterion(crit_name, x_sh, y, w_reduced, par))

    mat_alg_crit /= iters
    return mat_alg_crit, w_store


def test_alg_crit(
    w: np.ndarray,
    crit: Sequence[str],
    x_test: np.ndarray,
    y_test: np.ndarray,
    param: Mapping[str, object],
) -> np.ndarray:
    """Port of MATLAB TestAlgCrit.m."""
    out = np.zeros((w.shape[1], len(crit)), dtype=float)
    beta = lscov(x_test, y_test)
    local_param = dict(param)
    local_param["rss"] = float(np.sum((y_test - x_test @ beta) ** 2))

    for i, crit_name in enumerate(crit):
        for j in range(w.shape[1]):
            idx_del = w[:, j] == 0
            x_unnorm = x_test[:, ~idx_del]
            local_param["X_unnorm"] = x_unnorm
            out[j, i] = float(
                run_criterion(
                    crit_name,
                    x_test[:, ~idx_del],
                    y_test,
                    w[~idx_del, j],
                    local_param,
                )
            )
    return out
