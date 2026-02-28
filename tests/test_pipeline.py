import numpy as np

from qp_feature_selection import (
    FeatureSpec,
    MainLikeParams,
    alg_crit,
    create_opt_problem,
    load_real_data,
    run_main_like_pipeline,
    solve_opt_problem,
)


def test_qp_solution_satisfies_constraints() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((80, 12))
    y = rng.standard_normal(80)
    q, b = create_opt_problem(x, y, sim="correl", rel="correl")
    w = solve_opt_problem(q, b)
    assert w.shape == (12,)
    assert np.all(w >= -1e-8)
    assert np.sum(w) <= 1.0 + 1e-8


def test_main_like_pipeline_runs_on_small_synthetic_dataset() -> None:
    params = MainLikeParams(
        data="artificial",
        objects=120,
        feature_spec=FeatureSpec(
            rand_features=12,
            ortfeat_features=0,
            coltarget_features=0,
            colfeat_features=0,
            ortcol_features=0,
        ),
        random_seed=0,
    )
    result = run_main_like_pipeline(params)
    assert result.x.shape[0] == result.x_train.shape[1]
    assert np.all(result.x >= -1e-8)
    assert np.sum(result.x) <= 1.0 + 1e-8
    assert np.all(np.diff(result.threshold) >= -1e-12)
    assert result.active_masks.shape[0] == result.threshold.shape[0]


def test_load_real_data() -> None:
    x, y = load_real_data("data/BP50GATEST.mat", "bp50_s1d_ll_a", "bp50_y1_ll_a")
    assert x.shape == (113, 401)
    assert y.shape == (113,)


def test_alg_crit_smoke() -> None:
    x, y = load_real_data("data/BP50GATEST.mat", "bp50_s1d_ll_a", "bp50_y1_ll_a")
    y = y.reshape(-1)
    x = x[:70, :20]
    y = y[:70]
    params = {
        "iter": 1,
        "threshold": 1e-10,
        "s_0": 0.5,
        "Genetic": {"nGenerations": 2, "nIndividuals": 4, "mutationProb": 0.5},
    }
    mat, w = alg_crit(
        ["Lasso", "Ridge"],
        ["RSS", "CondNumber", "Vif", "complexity"],
        x,
        y,
        params,
    )
    assert mat.shape == (2, 4)
    assert w.shape == (20, 2)
    assert np.all(np.isfinite(mat) | np.isinf(mat))
