# Experiment report

## Study

**Title:** Data-Driven Decision Intelligence with Uncertainty-Calibrated AI: Learning When to Predict and When to Defer

The experiment used the UCI AI4I 2020 Predictive Maintenance dataset (https://doi.org/10.24432/C5HS5C). It compared raw-feature baselines, a knowledge-feature XGBoost model, independent Platt calibration, and a class-conditional (Mondrian) split-conformal defer policy.

Runtime: **0.08 minutes**  
Seeds: **11, 23, 42, 67, 101**  
Primary conformal error level: **0.10**  
Boosting backend: **XGBoost**

## Predictive and calibration results

| Model | PR-AUC | ROC-AUC | Brier | ECE | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression — raw sensors | 0.4192 ± 0.0205 | 0.9088 ± 0.0174 | 0.1253 ± 0.0039 | 0.2397 ± 0.0077 | 0.8235 ± 0.0650 | 0.2372 ± 0.0102 |
| XGBoost — raw sensors | 0.7761 ± 0.0465 | 0.9793 ± 0.0071 | 0.0265 ± 0.0013 | 0.0414 ± 0.0024 | 0.8549 ± 0.0382 | 0.6150 ± 0.0252 |
| XGBoost — knowledge features | 0.8757 ± 0.0199 | 0.9837 ± 0.0074 | 0.0150 ± 0.0015 | 0.0238 ± 0.0019 | 0.8706 ± 0.0328 | 0.7501 ± 0.0293 |
| XGBoost — knowledge + Platt calibration | 0.8757 ± 0.0199 | 0.9837 ± 0.0074 | 0.0087 ± 0.0016 | 0.0059 ± 0.0013 | 0.8196 ± 0.0578 | 0.8388 ± 0.0357 |

Lower Brier score and ECE are better. PR-AUC is particularly important because machine failure is rare.

## Decision results

| Policy | Coverage | Selective risk | System accuracy* | Mean cost |
| --- | --- | --- | --- | --- |
| Calibrated XGBoost — no defer | 1.0000 ± 0.0000 | 0.0107 ± 0.0022 | 0.9893 ± 0.0022 | 0.1272 ± 0.0390 |
| Proposed Mondrian conformal defer | 0.8945 ± 0.0187 | 0.0064 ± 0.0019 | 0.9943 ± 0.0018 | 0.0813 ± 0.0062 |

*System accuracy assumes every deferred case is resolved correctly by a human reviewer. The corresponding review cost is included in mean decision cost.

The cost scenario is predeclared as false positive=1, false negative=20, and defer=0.5. These are scenario weights, not measured currency values.

## Primary-seed uncertainty intervals

| Primary-seed quantity | Estimate | 95% bootstrap CI |
| --- | --- | --- |
| coverage | 0.8923 | [0.8767, 0.9077] |
| selective_risk | 0.0045 | [0.0015, 0.0082] |
| system_accuracy_assuming_correct_review | 0.9960 | [0.9927, 0.9987] |
| proposed_mean_cost | 0.0707 | [0.0512, 0.1060] |
| baseline_mean_cost | 0.0969 | [0.0427, 0.1777] |
| cost_reduction_vs_baseline | 0.0261 | [-0.0273, 0.0939] |

A positive `cost_reduction_vs_baseline` means the proposed defer policy costs less than the calibrated no-defer policy under the declared cost scenario.

## Interpretation rules

- Do not claim that calibration improves ranking performance; calibration targets probability reliability.
- Do not claim causal effects or real-world financial savings from this synthetic dataset.
- Treat conformal guarantees as marginal, class-conditional coverage under exchangeability, not a guarantee for every individual case.
- Report coverage together with selective risk: a model can reduce risk trivially by deferring nearly everything.
- Validate on an additional real predictive-maintenance dataset before targeting a selective Q1 journal.

## Generated files

- `predictive_metrics_by_seed.csv`: accuracy, ranking, and calibration results.
- `selective_metrics_by_seed.csv`: no-defer versus conformal-defer policies.
- `alpha_tradeoff.csv`: sensitivity to the conformal error level.
- `bootstrap_ci_primary_seed.csv`: paired uncertainty intervals.
- `plots/`: reliability, risk–coverage, alpha trade-off, and feature-importance figures.
- `models/`: primary-seed fitted model, calibrator, conformal thresholds, and feature schema.
