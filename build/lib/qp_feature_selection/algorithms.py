"""Algorithm wrappers ported from MATLAB alg/*.m."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, lars_path

from ._linear import lscov, ols_with_intercept
from .criteria import run_criterion


def lasso_algorithm(x: np.ndarray, y: np.ndarray, par: Mapping[str, object]) -> np.ndarray:
    model = LassoCV(cv=5, fit_intercept=False, alphas=100, max_iter=10_000, random_state=0)
    model.fit(x, y)
    return model.coef_.astype(float)


def elastic_net_algorithm(x: np.ndarray, y: np.ndarray, par: Mapping[str, object]) -> np.ndarray:
    model = ElasticNetCV(cv=5, l1_ratio=0.5, fit_intercept=False, max_iter=10_000, random_state=0)
    model.fit(x, y)
    return model.coef_.astype(float)


def ridge_like_algorithm(x: np.ndarray, y: np.ndarray, par: Mapping[str, object]) -> np.ndarray:
    # MATLAB Ridge.m actually uses lasso(..., 'Alpha', 1e-2).
    model = ElasticNetCV(cv=5, l1_ratio=1e-2, fit_intercept=False, max_iter=10_000, random_state=0)
    model.fit(x, y)
    return model.coef_.astype(float)


def lars_algorithm(x: np.ndarray, y: np.ndarray, par: Mapping[str, object]) -> np.ndarray:
    _, _, coefs = lars_path(x, y, method="lar", verbose=False)
    rss = np.sum((y.reshape(-1, 1) - x @ coefs) ** 2, axis=0)
    idx_min = int(np.argmin(rss))
    return coefs[:, idx_min].astype(float)


def _stepwise_p_values(x: np.ndarray, y: np.ndarray, included: list[int]) -> dict[int, float]:
    if not included:
        return {}
    fit = ols_with_intercept(x[:, included], y)
    pvals = np.asarray(fit["p_values"], dtype=float)[1:]
    return {feature: float(pvals[i]) for i, feature in enumerate(included)}


def stepwise_algorithm(
    x: np.ndarray,
    y: np.ndarray,
    par: Mapping[str, object],
    p_enter: float = 0.05,
    p_remove: float = 0.10,
    max_iter: int = 200,
) -> np.ndarray:
    n_features = x.shape[1]
    included: list[int] = []
    excluded = set(range(n_features))
    for _ in range(max_iter):
        changed = False
        # Forward step.
        best_feature = None
        best_p = float("inf")
        for feat in sorted(excluded):
            trial = included + [feat]
            fit = ols_with_intercept(x[:, trial], y)
            trial_p = float(np.asarray(fit["p_values"], dtype=float)[-1])
            if trial_p < best_p:
                best_p = trial_p
                best_feature = feat
        if best_feature is not None and best_p < p_enter:
            included.append(best_feature)
            excluded.remove(best_feature)
            changed = True

        # Backward step.
        if included:
            pvals = _stepwise_p_values(x, y, included)
            worst_feature = max(pvals, key=pvals.get)
            if pvals[worst_feature] > p_remove:
                included.remove(worst_feature)
                excluded.add(worst_feature)
                changed = True
        if not changed:
            break

    w = np.zeros(n_features, dtype=float)
    if included:
        w[np.array(included)] = lscov(x[:, included], y)
    return w


def filter_masks(masks: np.ndarray, max_items: int) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    idx = list(range(masks.shape[1]))
    i_mask1 = 0
    while i_mask1 < len(idx):
        count = int(np.sum(masks[:, idx[i_mask1]]))
        if count < 1 or count > max_items:
            idx.pop(i_mask1)
            continue
        i_mask2 = i_mask1 + 1
        while i_mask2 < len(idx):
            if np.array_equal(masks[:, idx[i_mask1]], masks[:, idx[i_mask2]]):
                idx.pop(i_mask2)
            else:
                i_mask2 += 1
        i_mask1 += 1
    return np.array(idx, dtype=int)


def test_mask(
    mask: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    param: Mapping[str, object],
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask.reshape(-1, 1)
    errors = []
    crit_names = list(param.get("crit", []))
    for col in range(mask.shape[1]):
        active = mask[:, col]
        x_sel = features[:, active]
        if x_sel.size == 0:
            errors.append(float("inf"))
            continue
        w = lscov(x_sel, target)
        total = 0.0
        for name in crit_names:
            total += float(run_criterion(name, x_sel, target, w, param))
        errors.append(total)
    return np.asarray(errors, dtype=float)


def genetic_algorithm(
    features: np.ndarray,
    target: np.ndarray,
    params: Mapping[str, object],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    n_features = features.shape[1]
    max_n_features = n_features
    n_generations = int(params.get("Genetic", {}).get("nGenerations", 10))
    n_individuals = int(params.get("Genetic", {}).get("nIndividuals", 10))
    mutation_prob = float(params.get("Genetic", {}).get("mutationProb", 0.5))

    generation = rng.random((n_features, n_individuals)) > 0.5
    q = test_mask(generation, features, target, params)

    for _ in range(n_generations):
        new_generation = np.empty((n_features, 0), dtype=bool)
        for i_ind1 in range(n_individuals):
            ind1 = generation[:, i_ind1]
            for i_ind2 in range(i_ind1 + 1, n_individuals):
                ind2 = generation[:, i_ind2]
                part1 = rng.random(n_features)
                part2 = 1.0 - part1
                child = (ind1 * part1 + ind2 * part2) >= 0.5
                new_generation = np.column_stack([new_generation, child])
            mutated = np.logical_xor(ind1, rng.random(n_features) < mutation_prob)
            new_generation = np.column_stack([new_generation, mutated])
        if new_generation.shape[1] > 0:
            idx_new = filter_masks(new_generation, max_n_features)
            new_generation = new_generation[:, idx_new]
            new_q = test_mask(new_generation, features, target, params)
            generation = np.column_stack([generation, new_generation])
            q = np.concatenate([q, new_q])

        idx = filter_masks(generation, max_n_features)
        generation = generation[:, idx]
        q = q[idx]

        sort_idx = np.argsort(q)
        keep = sort_idx[: min(n_individuals, sort_idx.size)]
        generation = generation[:, keep]
        q = q[keep]

    informative = generation[:, 0]
    weights = np.zeros(n_features, dtype=float)
    if np.any(informative):
        weights[informative] = lscov(features[:, informative], target)
    return weights


ALGORITHMS: dict[str, Callable[[np.ndarray, np.ndarray, Mapping[str, object]], np.ndarray]] = {
    "Lasso": lasso_algorithm,
    "ElasticNet": elastic_net_algorithm,
    "Ridge": ridge_like_algorithm,
    "LARS": lars_algorithm,
    "Stepwise": stepwise_algorithm,
    "Genetic": genetic_algorithm,
}


def run_algorithm(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    params: Mapping[str, object],
) -> np.ndarray:
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm: {name}")
    w = ALGORITHMS[name](x, y, params)
    return np.asarray(w, dtype=float).reshape(-1)
