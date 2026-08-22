# Data-Driven Decision Intelligence with Uncertainty-Calibrated AI

## Learning When to Predict and When to Defer

This repository contains a complete, reproducible predictive-maintenance experiment for studying whether calibrated AI should make a prediction automatically or defer an uncertain case to a human reviewer.

The code downloads the official [UCI AI4I 2020 Predictive Maintenance dataset](https://doi.org/10.24432/C5HS5C), validates it, runs leakage-safe experiments, and creates publication-ready tables, figures, fitted artifacts, and a Markdown report.

## Fastest RunPod procedure

Use a RunPod image with Python 3.10, 3.11, or 3.12. A GPU is not required; the dataset has only 10,000 rows.

```bash
cd uncertainty-calibrated-ddi
bash runpod_start.sh
```

The script creates an isolated environment, installs pinned packages, runs the unit tests, downloads the 510 KB dataset, and executes the five-seed paper experiment. On ordinary RunPod hardware it should finish well inside one hour. Network and image startup time can vary.

If the pod already has a suitable environment:

```bash
cd uncertainty-calibrated-ddi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python run_experiment.py --mode paper
```

For a one-seed smoke test before the complete run:

```bash
python run_experiment.py --mode quick --output-dir quick_results
```

An optional `Dockerfile` is included for a completely isolated Python 3.11/Linux run with the XGBoost OpenMP runtime installed.

## What the experiment tests

The proposed framework has four components:

1. **Data-driven prediction:** XGBoost estimates machine-failure probability.
2. **Knowledge engineering:** sensor interactions represent thermal gap, mechanical power, wear-load interaction, and deviation from nominal speed.
3. **Uncertainty calibration:** a separate held-out split fits Platt probability calibration.
4. **Decision intelligence:** class-conditional split conformal prediction automates a decision only when a singleton prediction set agrees with the calibrated model; ambiguous, empty, or contradictory cases are deferred.

The primary comparison is:

- calibrated knowledge-feature XGBoost that always predicts; versus
- the same model with a Mondrian conformal predict-or-defer policy.

This isolates the contribution of the decision policy instead of attributing differences to a different predictive model.

## Leakage-safe split design

Every seed creates four disjoint stratified partitions:

| Partition | Fraction | Purpose |
|---|---:|---|
| Model training | 55% | Fit logistic regression and XGBoost models |
| Probability calibration | 15% | Fit the Platt calibrator |
| Conformal calibration | 15% | Learn class-specific nonconformity thresholds |
| Test | 15% | Final evaluation only |

The subtype labels `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`, along with `UDI` and `Product ID`, are explicitly excluded from every feature matrix. Using failure-subtype labels as inputs would leak the target.

## Models and ablations

The pipeline evaluates:

- class-balanced logistic regression on raw sensors;
- XGBoost on raw sensors;
- XGBoost on raw sensors plus knowledge features;
- knowledge-feature XGBoost plus Platt calibration;
- calibrated knowledge-feature XGBoost plus Mondrian conformal deferral.

Five fixed seeds are used in paper mode. Conformal sensitivity is measured for `alpha = 0.02, 0.05, 0.10, 0.15, 0.20`; `alpha = 0.10` is the predeclared primary setting.

## Metrics

Prediction and calibration:

- ROC-AUC and PR-AUC;
- recall, precision, specificity, F1 and balanced accuracy;
- Brier score and expected calibration error.

Decision quality:

- automatic coverage and deferral rate;
- selective risk and selective accuracy;
- system accuracy assuming deferred cases are correctly reviewed;
- automatic failure recall;
- fraction of baseline errors captured by deferral;
- mean decision cost.

The default cost scenario is:

```text
false positive = 1
false negative = 20
human deferral = 0.5
```

These are transparent scenario weights, not measured dollars. Change them in `src/ddi/config.py` and report a sensitivity analysis if the paper makes cost claims.

## Results produced

After a successful run, `results/` contains:

```text
results/
├── report.md
├── run_config.json
├── split_counts.csv
├── predictive_metrics_by_seed.csv
├── predictive_metrics_summary.csv
├── selective_metrics_by_seed.csv
├── selective_metrics_summary.csv
├── alpha_tradeoff.csv
├── conformal_thresholds.csv
├── bootstrap_ci_primary_seed.csv
├── primary_seed_test_predictions.csv
├── models/
│   ├── boosting_knowledge_model.joblib
│   ├── platt_calibrator.joblib
│   ├── conformal_policy.json
│   └── feature_schema.json
└── plots/
    ├── reliability_diagram.png
    ├── risk_coverage_curve.png
    ├── alpha_tradeoff.png
    └── feature_importance.png
```

## Apply the fitted policy to a CSV

The input must contain the six raw fields used by AI4I: `Type`, air temperature, process temperature, rotational speed, torque, and tool wear. Extra columns are preserved.

```bash
python predict.py \
  --input data/ai4i2020.csv \
  --model-dir results/models \
  --output new_decisions.csv
```

The decision encoding is:

- `0`: automatically predict no failure;
- `1`: automatically predict failure;
- `-1`: defer to a human reviewer.

## Repository layout

```text
run_experiment.py       Main experiment command
predict.py              Apply a fitted decision policy
runpod_start.sh         One-command RunPod setup and run
src/ddi/data.py         Download, validation, and four-way splitting
src/ddi/features.py     Raw and knowledge-guided feature construction
src/ddi/models.py       Logistic-regression and boosted-tree training
src/ddi/calibration.py  Independent Platt calibration
src/ddi/conformal.py    Mondrian conformal prediction and deferral
src/ddi/metrics.py      Predictive, calibration, decision, and bootstrap metrics
src/ddi/plots.py        Publication-oriented plots
src/ddi/experiment.py   End-to-end orchestration and report generation
tests/                  Unit tests for leakage, splits, conformal logic, and costs
```

## Methodological limits

- AI4I is synthetic. It is appropriate for a fast reproducible first study, but a strong journal submission should add a real maintenance dataset.
- A deferred case is assumed to be correctly resolved by a human. If reviewer error data are available, replace this assumption with an empirical human-performance model.
- Conformal validity depends on exchangeability between conformal-calibration and test examples.
- The experiment measures associative prediction and decision utility; it does not establish causal effects.
- Knowledge features use only sensor inputs, but AI4I's synthetic failure process is documented. Clearly disclose the engineered features and repeat the analysis without them, as this repository does.

## Reproducibility checklist

- Pinned package versions
- Fixed random seeds
- Cached official dataset and logged dataset DOI
- No target or failure-subtype leakage
- Separate probability- and conformal-calibration partitions
- Predeclared primary alpha and cost scenario
- Multi-seed results and bootstrap confidence intervals
- Saved predictions, fitted artifacts, plots, and run configuration

This software is a research prototype, not a safety-certified maintenance system.
