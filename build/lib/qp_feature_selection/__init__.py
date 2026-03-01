"""Python-native QPFeatureSelection package."""

from .algorithms import run_algorithm
from .criteria import run_criterion
from .data import (
    create_data,
    create_data2,
    load_real_data,
    normalize_design_and_target,
    split_train_test,
)
from .evaluation import alg_crit, test_alg_crit
from .optimization import create_opt_problem, solve_opt_problem
from .pipeline import (
    FeatureSpec,
    GeneticParams,
    MainLikeParams,
    ThresholdSweepResult,
    run_main_like_pipeline,
)

__all__ = [
    "FeatureSpec",
    "GeneticParams",
    "MainLikeParams",
    "ThresholdSweepResult",
    "alg_crit",
    "create_data",
    "create_data2",
    "create_opt_problem",
    "load_real_data",
    "normalize_design_and_target",
    "run_algorithm",
    "run_criterion",
    "run_main_like_pipeline",
    "solve_opt_problem",
    "split_train_test",
    "test_alg_crit",
]
