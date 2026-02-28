# Python Port Brief: QPFeatureSelection

## Purpose
This repository now contains a Python-native package that mirrors the MATLAB `mcode` workflow for quadratic-programming-based feature selection under multicollinearity.

Primary goals:
- Preserve the original algorithmic intent and defaults from MATLAB.
- Provide a maintainable Python package API.
- Provide a reproducible parity workflow (package pipeline vs MATLAB-style execution path, with optional direct MATLAB Engine comparison).

## Package Layout
- `src/qp_feature_selection/data.py`
  - MATLAB equivalents: `CreateData.m`, `CreateData2.m`
  - Responsibilities: synthetic data generation, real `.mat` loading, normalization, train/test split.
- `src/qp_feature_selection/optimization.py`
  - MATLAB equivalents: `CreateOptProblem.m`, `SolveOptProblem.m`
  - Responsibilities: build `Q` and `b`, solve constrained QP (`x >= 0`, `||x||_1 <= 1`).
- `src/qp_feature_selection/criteria.py`
  - MATLAB equivalents: `criteria/*.m`
  - Responsibilities: RSS, Cp, BIC, VIF, condition number, adjusted R^2, stability diagnostics.
- `src/qp_feature_selection/algorithms.py`
  - MATLAB equivalents: `alg/*.m`
  - Responsibilities: Lasso, LARS, ElasticNet, Ridge-like path, Stepwise, Genetic feature selection utilities.
- `src/qp_feature_selection/evaluation.py`
  - MATLAB equivalents: `AlgCrit.m`, `TestAlgCrit.m`
  - Responsibilities: algorithm/criterion scoring loops and cross-method evaluation matrices.
- `src/qp_feature_selection/pipeline.py`
  - MATLAB equivalent: `main.m` high-level run path.
  - Responsibilities: end-to-end orchestration and threshold sweep outputs.
- `src/qp_feature_selection/matlab_parity.py`
  - Optional direct MATLAB Engine runner for cross-language parity checks.

## Behavioral Notes
- The Python code intentionally keeps several MATLAB quirks where practical, because parity matters more than idealized cleanup.
- Example: `Cp` uses MATLAB-style dimensional behavior (column-vector interpretation can make `p=1` unless a 2D row/column shape is passed).
- Some MATLAB internals (CVX/MOSEK, `stepwisefit`, MI mex binaries) do not have exact Python equivalents in base scientific stack; these are approximated with standard Python tools while keeping interfaces and intent aligned.

## Main API
- High-level entry:
  - `run_main_like_pipeline(MainLikeParams(...))`
- Lower-level building blocks:
  - `create_data(...)`, `create_opt_problem(...)`, `solve_opt_problem(...)`
  - `run_algorithm(...)`, `run_criterion(...)`
  - `alg_crit(...)`, `test_alg_crit(...)`

## Notebook and Parity Workflow
- Notebook: `notebooks/qp_feature_selection_demo.ipynb`
- It provides:
  1. Python package run on real data.
  2. MATLAB-style procedural reference run in pure Python.
  3. Numeric comparisons (`np.allclose`) for key outputs.
  4. Optional direct MATLAB Engine comparison if MATLAB is installed.

## Contributor Guidance (Human + AI)
- Keep MATLAB mapping explicit when changing behavior:
  - Mention the source MATLAB file/function in PR notes or commit messages.
- Prefer additive changes over silent semantic shifts:
  - If you improve an algorithm, keep a compatibility mode or explain parity impact.
- Verify with tests:
  - Run `python -m pytest -q`.
- If changing notebook logic:
  - Keep at least one deterministic parity check table in the notebook.
- If adding dependencies:
  - Keep core runtime lean; prefer optional extras for heavy tooling (notebooks/visualization/dev).

## Quick Start
```bash
python -m pip install -e .[dev]
python -m pytest -q
```

Optional notebook support:
```bash
python -m pip install -e .[notebook]
```

## Known Limits
- Direct MATLAB execution is optional and requires local MATLAB + MATLAB Engine for Python.
- Exact floating-point equality across MATLAB/Python solvers is not guaranteed; tolerance-based checks are used.
- The MI path is provided in Python but is not a binary-identical substitute for MATLAB mex implementations.
