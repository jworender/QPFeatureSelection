"""Data generation and loading utilities ported from MATLAB."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.io import loadmat
from scipy.linalg import null_space


def _get_feature_count(features: Mapping[str, int], key: str) -> int:
    return int(features.get(key, 0))


def _null_basis(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return null_space(arr)


def load_real_data(
    filename: str | Path,
    x_var: str,
    y_var: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load real dataset from a MATLAB .mat file."""
    raw = loadmat(filename)
    if x_var not in raw or y_var not in raw:
        missing = [k for k in [x_var, y_var] if k not in raw]
        raise KeyError(f"Missing variables in {filename}: {missing}")
    x = np.asarray(raw[x_var], dtype=float)
    y = np.asarray(raw[y_var], dtype=float).reshape(-1)
    return x, y


def create_data(
    m: int,
    features: Mapping[str, int],
    par: Mapping[str, object],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Port of MATLAB CreateData.m."""
    rng = np.random.default_rng() if rng is None else rng

    random = _get_feature_count(features, "rand_features")
    ortfeat = _get_feature_count(features, "ortfeat_features")
    coltarget = _get_feature_count(features, "coltarget_features")
    colfeat = _get_feature_count(features, "colfeat_features")
    ortcol = _get_feature_count(features, "ortcol_features")
    k = float(par["multpar"])
    target = np.asarray(par["target"], dtype=float).reshape(-1)

    total_features = random + ortfeat + coltarget + colfeat + ortcol
    if m < total_features:
        raise ValueError("Not enough objects, objects must be more than features")
    if ortfeat == 1:
        raise ValueError("Ortogonal features must be more than 1")

    mat_random = np.empty((m, 0))
    if random > 0:
        rand_cols = rng.random((m, max(random - 1, 0)))
        mat_random = np.column_stack([rand_cols, target + 0.01 * rng.standard_normal(m)])

    if ortfeat > 0:
        vec_1 = np.zeros(m)
        vec_1[::2] = target[::2]
        vec_2 = np.zeros(m)
        vec_2[1::2] = target[1::2]
        if ortfeat < 3:
            mat_ortfeat = np.column_stack([vec_1, vec_2])
        else:
            mat_ort_ortfeat = _null_basis(np.vstack([vec_1, vec_2]))
            perm = rng.permutation(mat_ort_ortfeat.shape[1])[: ortfeat - 2]
            mat_ortfeat = np.column_stack([vec_1, vec_2, mat_ort_ortfeat[:, perm]])
    else:
        mat_ortfeat = np.empty((m, 0))

    mat_coltarget = np.zeros((m, coltarget))
    if coltarget > 0:
        mat_ort_coltarget = _null_basis(target)
        for i in range(coltarget):
            mat_coltarget[:, i] = k * target + (1.0 - k) * mat_ort_coltarget[:, i]

    mat_colfeat = np.zeros((m, colfeat))
    if ortfeat > 1 and colfeat > 0:
        idx_first = 0
        idx_last = -1
        colfeat_per_ortfeat = np.full(ortfeat, colfeat // ortfeat, dtype=int)
        for i in range(colfeat - (colfeat // ortfeat) * ortfeat):
            colfeat_per_ortfeat[i] += 1
        for i in range(ortfeat):
            mat_ort_ortfeat = _null_basis(mat_ortfeat[:, i])
            mat_col_perfeat = np.zeros((m, colfeat_per_ortfeat[i]))
            idx_last = idx_last + colfeat_per_ortfeat[i]
            # Keep MATLAB indexing logic as-is (uses j column, not i column).
            for j in range(colfeat_per_ortfeat[i]):
                mat_col_perfeat[:, j] = (
                    k * mat_ortfeat[:, j] + (1.0 - k) * mat_ort_ortfeat[:, j]
                )
            mat_colfeat[:, idx_first : idx_last + 1] = mat_col_perfeat
            idx_first = idx_last + 1

    mat_ortcol = np.zeros((m, ortcol))
    if ortcol > 0:
        mat_ort_coltarget = _null_basis(target)
        mid = mat_ort_coltarget.shape[1] // 2
        for i in range(ortcol):
            mat_ortcol[:, i] = (
                k * mat_ort_coltarget[:, mid] + (1.0 - k) * mat_ort_coltarget[:, i]
            )

    x = np.column_stack([mat_ortcol, mat_ortfeat, mat_colfeat, mat_coltarget, mat_random])
    return x


def create_data2(
    m: int,
    features: Mapping[str, int],
    par: Mapping[str, object],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of MATLAB CreateData2.m."""
    rng = np.random.default_rng() if rng is None else rng
    if str(par.get("data", "")) == "real":
        x, target = load_real_data(
            str(par["real_data_filename"]),
            str(par["real_data_X"]),
            str(par["real_data_y"]),
        )
        return x, target

    random = _get_feature_count(features, "rand_features")
    ortfeat = _get_feature_count(features, "ortfeat_features")
    coltarget = _get_feature_count(features, "coltarget_features")
    colfeat = _get_feature_count(features, "colfeat_features")
    ortcol = _get_feature_count(features, "ortcol_features")
    k = float(par["multpar"])
    target = np.asarray(par["target"], dtype=float).reshape(-1)

    total_features = random + ortfeat + coltarget + colfeat + ortcol
    if m < total_features:
        raise ValueError("Not enough objects, objects must be more than features")
    if ortfeat == 1:
        raise ValueError("Ortogonal features must be more than 1")

    mat_random = np.empty((m, 0))
    if random > 0:
        rand_cols = rng.random((m, max(random - 1, 0)))
        mat_random = np.column_stack([rand_cols, target + 0.01 * rng.standard_normal(m)])

    if ortfeat > 0:
        vec_1 = np.zeros(m)
        odd_idx = np.arange(0, m, 2)
        vec_1[odd_idx] = target[odd_idx] - np.mean(target[odd_idx])
        vec_2 = np.zeros(m)
        even_idx = np.arange(1, m, 2)
        vec_2[even_idx] = target[even_idx] - np.mean(target[even_idx])
        target = vec_1 + vec_2
        if ortfeat < 3:
            mat_ortfeat = np.column_stack([vec_1, vec_2])
        else:
            mat_ort_ortfeat = _null_basis(np.vstack([vec_1, vec_2]))
            perm = rng.permutation(mat_ort_ortfeat.shape[1])[: ortfeat - 2]
            mat_ortfeat = np.column_stack([vec_1, vec_2, mat_ort_ortfeat[:, perm]])
    else:
        mat_ortfeat = np.empty((m, 0))

    mat_coltarget = np.zeros((m, coltarget))
    if coltarget > 0:
        mat_ort_coltarget = _null_basis(target)
        for i in range(coltarget):
            mat_coltarget[:, i] = k * target + (1.0 - k) * mat_ort_coltarget[:, i]

    mat_colfeat = np.zeros((m, colfeat))
    if ortfeat > 1 and colfeat > 0:
        idx_first = 0
        idx_last = -1
        colfeat_per_ortfeat = np.full(ortfeat, colfeat // ortfeat, dtype=int)
        for i in range(colfeat - (colfeat // ortfeat) * ortfeat):
            colfeat_per_ortfeat[i] += 1
        for i in range(ortfeat):
            mat_ort_ortfeat = _null_basis(mat_ortfeat[:, i])
            mat_col_perfeat = np.zeros((m, colfeat_per_ortfeat[i]))
            idx_last = idx_last + colfeat_per_ortfeat[i]
            # Keep MATLAB indexing logic as-is (uses j column, not i column).
            for j in range(colfeat_per_ortfeat[i]):
                mat_col_perfeat[:, j] = (
                    k * mat_ortfeat[:, j] + (1.0 - k) * mat_ort_ortfeat[:, j]
                )
            mat_colfeat[:, idx_first : idx_last + 1] = mat_col_perfeat
            idx_first = idx_last + 1

    mat_ortcol = np.zeros((m, ortcol))
    if ortcol > 0:
        target = target - np.mean(target)
        mat_ort_coltarget = _null_basis(target)
        mid = mat_ort_coltarget.shape[1] // 2
        for i in range(ortcol):
            mat_ortcol[:, i] = (
                k * mat_ort_coltarget[:, mid] + (1.0 - k) * mat_ort_coltarget[:, i]
            )

    x = np.column_stack([mat_ortcol, mat_ortfeat, mat_colfeat, mat_coltarget, mat_random])
    return x, target


def normalize_design_and_target(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match normalization in MATLAB main.m."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    lens = np.sqrt(np.sum(x**2, axis=0))
    lens[lens == 0.0] = 1.0
    x_norm = x / lens
    y_norm = y / np.linalg.norm(y)
    return x_norm, y_norm


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    test_set_ratio: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split exactly as done in MATLAB main.m (no shuffling)."""
    m = x.shape[0]
    split_idx = int(np.floor(test_set_ratio * m))
    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    return x_train, y_train, x_test, y_test
