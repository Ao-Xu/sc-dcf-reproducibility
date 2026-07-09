# Reproducibility Notes

This folder contains the code used to generate the computational results in the CSDA manuscript.

## Environment

The reported run used:

- Python 3.7.0
- NumPy 1.21.6
- SciPy 1.1.0
- scikit-learn 1.0.2
- pandas 1.1.5
- Intel Core Ultra 9 285H workstation, 16 logical processors, 31.4 GB RAM

Install the Python dependencies listed in `requirements.txt`.

## Rerun command

From this directory, run:

```powershell
.\run_all_experiments.ps1
```

or equivalently:

```powershell
python .\run_experiments.py
```

The script downloads public UCI and Mulan data sets as needed, caches them under `data_cache/`, and writes CSV summaries, LaTeX tables, and figures under `outputs/`.

## Random seeds

All simulation seeds, cross-validation splits, random center baselines, k-means initializations, and random forest baselines are fixed inside `run_experiments.py`.

## Main output files

- `outputs/raw_results.csv`: all fold-level and replicate-level results.
- `outputs/rate_summary.csv`: low-dimensional rate simulation summary.
- `outputs/scaling_summary.csv`: actual output-scaling timing summary.
- `outputs/large_scale_stress_summary.csv`: large-scale stress-test timing summary.
- `outputs/direct_max_sanity_summary.csv`: small direct max-feature sanity summary.
- `outputs/real_standardized_summary.csv`: full real-data six-method summary.
- `outputs/table_*.tex`: LaTeX tables used or referenced by the manuscript.
- `outputs/fig_*.pdf`: manuscript figures.
