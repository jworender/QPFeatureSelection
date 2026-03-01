# QPFeatureSelection

## Purpose
This repository implements quadratic-programming feature selection to address multicollinearity in regression [[A. Katrutsa, V. Srijov, 2017]](http://www.sciencedirect.com/science/article/pii/S0957417417300635).

The project now contains:
- Original MATLAB implementation in `mcode/`.
- A Python-native package in `src/qp_feature_selection/` that reproduces the same core algorithm and workflow.

The Python port exists specifically to enable apples-to-apples comparisons against other Python-based modeling architectures while preserving the original MATLAB logic and outputs as a reference.

## What The Algorithm Produces
Given design matrix `X` and target `y`, the workflow builds and solves:
- feature-similarity matrix `Q`
- feature-relevance vector `b`
- constrained QP weights `x` such that `x >= 0` and `||x||_1 <= 1`

Then it sweeps thresholds over `x` to produce model-selection diagnostics such as:
- `rss`, `rss_test`
- `vif`
- `complexity`

## Architecture
- MATLAB reference code: `mcode/`
  - `CreateData*.m`, `CreateOptProblem.m`, `SolveOptProblem.m`, criteria and algorithm modules
- Python package: `src/qp_feature_selection/`
  - `data.py`: data generation/loading and preprocessing
  - `optimization.py`: `Q`, `b`, and constrained QP solve
  - `criteria.py`: evaluation criteria (RSS, Cp, VIF, etc.)
  - `algorithms.py`: feature-selection algorithm wrappers
  - `evaluation.py`: MATLAB-like algorithm/criteria matrix evaluation
  - `pipeline.py`: high-level `main.m`-style run path
  - `matlab_parity.py`: optional direct MATLAB Engine parity utilities
- Demo notebook: `notebooks/qp_feature_selection_demo.ipynb`
  - Runs Python pipeline
  - Runs MATLAB-style and direct MATLAB parity checks
  - Reports metric-level comparison tables

## Usage
### MATLAB
From repository root in MATLAB:
```matlab
addpath('./mcode');
addpath('./mcode/criteria');
addpath('./mcode/alg');
addpath('./data');
main
```

### Python
Install:
```bash
python -m pip install -e .[dev]
```

Optional notebook extras:
```bash
python -m pip install -e .[notebook]
```

Run tests:
```bash
python -m pytest -q
```

Run the demo notebook:
- `notebooks/qp_feature_selection_demo.ipynb`

## Parity And Comparison Workflow
The notebook and package are designed to validate equivalence at key outputs:
- `x`, `threshold`
- `Q`, `b`
- sweep metrics (`rss`, `rss_test`, `vif`, `complexity`)

This parity-first design is what enables fair apples-to-apples evaluation versus other Python modeling pipelines.
