# QPFeatureSelection Contributor Brief

## Purpose
This codebase provides a MATLAB reference implementation and a Python-native implementation of the same quadratic-programming feature-selection method for multicollinearity.

The Python package was built to support apples-to-apples comparisons with other Python modeling architectures while keeping behavior aligned with the MATLAB baseline.

## Scope Of The Python Package
The Python package (`src/qp_feature_selection`) ports the MATLAB workflow end-to-end:
- data creation/loading
- optimization problem setup (`Q`, `b`)
- constrained QP solve for feature weights `x`
- threshold sweep diagnostics (`rss`, `rss_test`, `vif`, `complexity`)
- algorithm/criterion evaluation helpers (`AlgCrit`/`TestAlgCrit` equivalents)

## Current Architecture
- `src/qp_feature_selection/data.py`
  - MATLAB mapping: `CreateData.m`, `CreateData2.m`
  - Handles synthetic data generation, `.mat` loading, normalization, split logic.
- `src/qp_feature_selection/optimization.py`
  - MATLAB mapping: `CreateOptProblem.m`, `SolveOptProblem.m`
  - Builds `Q`, `b`, solves constrained QP (`x >= 0`, `||x||_1 <= 1`).
- `src/qp_feature_selection/criteria.py`
  - MATLAB mapping: `criteria/*.m`
  - Implements criteria used in selection and diagnostics, including `Vif`.
- `src/qp_feature_selection/algorithms.py`
  - MATLAB mapping: `alg/*.m`
  - Wraps algorithmic feature-selection variants used by evaluation utilities.
- `src/qp_feature_selection/evaluation.py`
  - MATLAB mapping: `AlgCrit.m`, `TestAlgCrit.m`
  - Produces algorithm-vs-criteria result matrices.
- `src/qp_feature_selection/pipeline.py`
  - MATLAB mapping: `main.m`
  - Provides a high-level run path (`run_main_like_pipeline`) for reproducible outputs.
- `src/qp_feature_selection/matlab_parity.py`
  - Optional direct MATLAB Engine integration for runtime parity checks.
- `notebooks/qp_feature_selection_demo.ipynb`
  - Demonstrates Python run, MATLAB-style reference run, direct MATLAB run, and metric-level comparisons.

## Usage
### Python
```bash
python -m pip install -e .[dev]
python -m pytest -q
```

Notebook support:
```bash
python -m pip install -e .[notebook]
```

Open and run:
- `notebooks/qp_feature_selection_demo.ipynb`

### MATLAB
Use `mcode/main.m` as reference entry point.  
If CVX/MOSEK are unavailable, `mcode/SolveOptProblem.m` now includes a fallback projected-gradient path.

## Parity And Comparison Strategy
Parity checks focus on these outputs:
- optimization artifacts: `Q`, `b`, `x`, `threshold`
- sweep diagnostics: `rss`, `rss_test`, `vif`, `complexity`

Notes:
- Exact floating-point equality is not always expected; tolerance-based checks are required.
- `VIF` can be numerically unstable in near-singular regimes; comparison logic should account for this.

## Rules For Future Changes (Human + AI)
- Preserve MATLAB mapping and document any intentional divergence.
- Do not silently change default behavior in ways that break comparison studies.
- Prefer compatibility flags when introducing improvements.
- Keep dependencies minimal in runtime; use optional extras for notebook/dev tooling.
- Maintain or improve parity checks when modifying pipeline, solver, or threshold logic.

## Why This Matters
This repository is not just a port. It is a controlled benchmark implementation:
- MATLAB acts as legacy/reference behavior.
- Python acts as integration and experimentation surface.
- Together they enable fair comparison against other Python-first model pipelines under the same feature-selection algorithm.
