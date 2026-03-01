"""High-level orchestration for the QP feature-selection workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ._linear import lscov
from .criteria import vif as vif_criterion
from .data import create_data, load_real_data, normalize_design_and_target, split_train_test
from .optimization import create_opt_problem, solve_opt_problem


@dataclass(slots=True)
class FeatureSpec:
    rand_features: int = 50
    ortfeat_features: int = 0
    coltarget_features: int = 0
    colfeat_features: int = 0
    ortcol_features: int = 0


@dataclass(slots=True)
class GeneticParams:
    n_generations: int = 10
    n_individuals: int = 20
    mutation_prob: float = 0.5


@dataclass(slots=True)
class MainLikeParams:
    objects: int = 1000
    multpar: float = 0.8
    algorithms: Sequence[str] = field(
        default_factory=lambda: ["Lasso", "LARS", "Stepwise", "ElasticNet", "Ridge", "Genetic"]
    )
    criteria: Sequence[str] = field(
        default_factory=lambda: ["complexity", "Cp", "RSS", "CondNumber", "Vif", "bic"]
    )
    iter: int = 1
    s_0: float = 0.5
    threshold: float = 1e-10
    data: str = "artificial"
    real_data_filename: str = "data/BP50GATEST.mat"
    real_data_x: str = "bp50_s1d_ll_a"
    real_data_y: str = "bp50_y1_ll_a"
    max_features: int | None = None
    test_set_ratio: float = 0.7
    sim: str = "correl"
    rel: str = "correl"
    normalize: bool = True
    random_seed: int = 0
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    genetic: GeneticParams = field(default_factory=GeneticParams)

    def to_algorithm_params(self, target: np.ndarray | None = None) -> dict[str, object]:
        par: dict[str, object] = {
            "multpar": self.multpar,
            "iter": self.iter,
            "s_0": self.s_0,
            "threshold": self.threshold,
            "data": self.data,
            "real_data_filename": self.real_data_filename,
            "real_data_X": self.real_data_x,
            "real_data_y": self.real_data_y,
            "Genetic": {
                "nGenerations": self.genetic.n_generations,
                "nIndividuals": self.genetic.n_individuals,
                "mutationProb": self.genetic.mutation_prob,
            },
        }
        if target is not None:
            par["target"] = target
        return par


@dataclass(slots=True)
class ThresholdSweepResult:
    x: np.ndarray
    threshold: np.ndarray
    rss: np.ndarray
    rss_test: np.ndarray
    vif: np.ndarray
    complexity: np.ndarray
    active_masks: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    q: np.ndarray
    b: np.ndarray


def run_main_like_pipeline(params: MainLikeParams | None = None) -> ThresholdSweepResult:
    """Run a Python-native equivalent of mcode/main.m."""
    params = MainLikeParams() if params is None else params
    rng = np.random.default_rng(params.random_seed)

    if params.data == "real":
        x, y = load_real_data(params.real_data_filename, params.real_data_x, params.real_data_y)
        y = y.reshape(-1)
    else:
        target = rng.integers(1, int(1.5 * params.objects) + 1, size=params.objects)
        par = params.to_algorithm_params(target=target)
        feat = {
            "rand_features": params.feature_spec.rand_features,
            "ortfeat_features": params.feature_spec.ortfeat_features,
            "coltarget_features": params.feature_spec.coltarget_features,
            "colfeat_features": params.feature_spec.colfeat_features,
            "ortcol_features": params.feature_spec.ortcol_features,
        }
        x = create_data(params.objects, feat, par, rng=rng)
        y = target.astype(float)

    if params.max_features is not None:
        x = x[:, : int(params.max_features)]

    if params.normalize:
        x, y = normalize_design_and_target(x, y)

    x_train, y_train, x_test, y_test = split_train_test(x, y, params.test_set_ratio)
    q, b = create_opt_problem(x_train, y_train, sim=params.sim, rel=params.rel)
    x_weights = solve_opt_problem(q, b)

    threshold = np.sort(x_weights)
    n_thresh = threshold.size
    rss = np.zeros(n_thresh, dtype=float)
    rss_test = np.zeros(n_thresh, dtype=float)
    vif = np.zeros(n_thresh, dtype=float)
    complexity = np.zeros(n_thresh, dtype=float)
    active_masks = np.zeros((n_thresh, x_train.shape[1]), dtype=bool)

    for i, th in enumerate(threshold):
        active_idx = x_weights >= th
        active_masks[i, :] = active_idx
        if np.sum(active_idx) > 0:
            w = lscov(x_train[:, active_idx], y_train)
            rss[i] = float(np.sum((x_train[:, active_idx] @ w - y_train) ** 2))
            rss_test[i] = float(np.sum((x_test[:, active_idx] @ w - y_test) ** 2))
            vif[i] = float(vif_criterion(x_train[:, active_idx], y_train, w, {}))
            complexity[i] = float(np.sum(active_idx))
        else:
            break

    return ThresholdSweepResult(
        x=x_weights,
        threshold=threshold,
        rss=rss,
        rss_test=rss_test,
        vif=vif,
        complexity=complexity,
        active_masks=active_masks,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        q=q,
        b=b,
    )
