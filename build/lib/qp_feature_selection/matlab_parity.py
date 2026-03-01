"""Optional MATLAB-engine integration for direct parity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def run_matlab_reference(
    repo_root: str | Path,
    max_features: int | None = None,
) -> dict[str, np.ndarray]:
    """Run the MATLAB workflow if matlab.engine is available.

    The run uses the real data path for deterministic results and returns
    vectors required for parity checks against the Python implementation.
    """
    try:
        import matlab.engine  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "matlab.engine is not available. Install MATLAB Engine for Python "
            "to run direct MATLAB parity checks."
        ) from exc

    repo_root = Path(repo_root).resolve()
    mcode = repo_root / "mcode"
    data_dir = repo_root / "data"
    eng = matlab.engine.start_matlab()
    try:
        eng.addpath(str(mcode), nargout=0)
        eng.addpath(str(mcode / "criteria"), nargout=0)
        eng.addpath(str(mcode / "alg"), nargout=0)
        eng.addpath(str(mcode / "mi"), nargout=0)
        eng.addpath(str(data_dir), nargout=0)
        eng.eval("rng(0);", nargout=0)
        eng.eval("load('BP50GATEST.mat');", nargout=0)
        eng.eval("X = bp50_s1d_ll_a; y = bp50_y1_ll_a;", nargout=0)
        if max_features is not None:
            eng.workspace["max_features"] = float(max_features)
            eng.eval("X = X(:, 1:floor(max_features));", nargout=0)
        eng.eval("len = sum(X.^2).^0.5; X = X./repmat(len, size(X, 1), 1); y = y ./ norm(y);", nargout=0)
        eng.eval("test_set_ratio = 0.7;", nargout=0)
        eng.eval("X_train = X(1:floor(test_set_ratio*size(X, 1)), :);", nargout=0)
        eng.eval("y_train = y(1:floor(test_set_ratio*size(X, 1)));", nargout=0)
        eng.eval("X_test = X(floor(test_set_ratio*size(X, 1)) + 1:size(X, 1), :);", nargout=0)
        eng.eval("y_test = y(floor(test_set_ratio*size(X, 1)) + 1:size(X, 1));", nargout=0)
        eng.eval("[Q, b] = CreateOptProblem(X_train, y_train, 'correl', 'correl');", nargout=0)
        eng.eval("x = SolveOptProblem(Q, b);", nargout=0)
        eng.eval("threshold = sort(x)';", nargout=0)
        eng.eval("rss = zeros(1, length(threshold));", nargout=0)
        eng.eval("rss_test = zeros(1, length(threshold));", nargout=0)
        eng.eval("vif = zeros(1, length(threshold));", nargout=0)
        eng.eval("complexity = zeros(1, length(threshold));", nargout=0)
        eng.eval("for i=1:length(threshold), active_idx = x >= threshold(i); if sum(active_idx)>0, lm = fitlm(X_train(:, active_idx), y_train); w = lm.Coefficients.Estimate(2:end); rss(i)=sumsqr(X_train(:, active_idx)*w - y_train); rss_test(i)=sumsqr(X_test(:, active_idx)*w - y_test); vif(i)=Vif(X_train(:, active_idx)); complexity(i)=sum(active_idx); else, break; end; end", nargout=0)

        out = {
            "x": _to_numpy(eng.workspace["x"]).reshape(-1),
            "threshold": _to_numpy(eng.workspace["threshold"]).reshape(-1),
            "rss": _to_numpy(eng.workspace["rss"]).reshape(-1),
            "rss_test": _to_numpy(eng.workspace["rss_test"]).reshape(-1),
            "vif": _to_numpy(eng.workspace["vif"]).reshape(-1),
            "complexity": _to_numpy(eng.workspace["complexity"]).reshape(-1),
            "q": _to_numpy(eng.workspace["Q"]),
            "b": _to_numpy(eng.workspace["b"]).reshape(-1),
        }
        return out
    finally:
        eng.quit()
