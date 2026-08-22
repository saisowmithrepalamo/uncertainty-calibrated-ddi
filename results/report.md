# Experiment report

## Study

**Title:** Data-Driven Decision Intelligence with Uncertainty-Calibrated AI: Learning When to Predict and When to Defer

The experiment used the UCI AI4I 2020 Predictive Maintenance dataset (https://doi.org/10.24432/C5HS5C). It compared raw-feature baselines, a knowledge-feature XGBoost model, independent Platt calibration, and a class-conditional (Mondrian) split-conformal defer policy.

Runtime: **0.09 minutes**  
Seeds: **11, 23, 42, 67, 101**  
Primary conformal error level: **0.10**  
Boosting backend: **XGBoost**

## Dataset integrity audit

- Dataset SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`
- Rows: **10000**
- Target failures: **339** (3.390%)
- Missing required cells: **0**
- Complete duplicate rows: **0**

The official file contains **27** target/failure-mode inconsistencies relative to the dataset documentation's OR rule: **9** target-positive rows have no positive failure-mode indicator, and **18** target-zero rows have at least one positive failure-mode indicator. Failure-mode columns are excluded from all model features. Labels are preserved exactly as published; the audit does not silently alter them.

## Predictive and calibration results

| Model | PR-AUC | ROC-AUC | Brier | ECE | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression — raw sensors | 0.4192 ± 0.0205 | 0.9088 ± 0.0174 | 0.1253 ± 0.0039 | 0.2397 ± 0.0077 | 0.8235 ± 0.0650 | 0.2372 ± 0.0102 |
| XGBoost — raw sensors | 0.7761 ± 0.0465 | 0.9793 ± 0.0071 | 0.0265 ± 0.0013 | 0.0414 ± 0.0024 | 0.8549 ± 0.0382 | 0.6150 ± 0.0252 |
| XGBoost — knowledge features | 0.8757 ± 0.0199 | 0.9837 ± 0.0074 | 0.0150 ± 0.0015 | 0.0238 ± 0.0019 | 0.8706 ± 0.0328 | 0.7501 ± 0.0293 |
| XGBoost — knowledge + Platt calibration | 0.8757 ± 0.0199 | 0.9837 ± 0.0074 | 0.0087 ± 0.0016 | 0.0059 ± 0.0013 | 0.8196 ± 0.0578 | 0.8388 ± 0.0357 |

Lower Brier score and ECE are better. PR-AUC is particularly important because machine failure is rare.

## Decision results

| Policy | Automation coverage | Deferral rate | Selective risk | System accuracy* | Mean cost |
| --- | --- | --- | --- | --- | --- |
| Calibrated XGBoost — no defer | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0107 ± 0.0022 | 0.9893 ± 0.0022 | 0.1272 ± 0.0390 |
| Proposed Mondrian conformal defer | 0.8945 ± 0.0187 | 0.1055 ± 0.0187 | 0.0064 ± 0.0019 | 0.9943 ± 0.0018 | 0.0813 ± 0.0062 |

*System accuracy assumes every deferred case is resolved correctly by a human reviewer. The corresponding review cost is included in mean decision cost.

The cost scenario is predeclared as false positive=1, false negative=20, and defer=0.5. These are scenario weights, not measured currency values.

`Automation coverage` is the fraction of cases receiving an automatic decision; it is exactly one minus the deferral rate. It is not conformal prediction-set coverage.

## Empirical conformal-set validation

| Setting | Empirical set coverage | Class-0 coverage | Class-1 coverage | Mean set size | Empty-set rate | Ambiguous-set rate |
| --- | --- | --- | --- | --- | --- | --- |
| alpha=0.10 | 0.8925 ± 0.0169 | 0.8912 ± 0.0182 | 0.9294 ± 0.0472 | 0.9521 ± 0.0263 | 0.0479 ± 0.0263 | 0.0000 ± 0.0000 |

Empirical set coverage is the fraction of test examples whose conformal prediction set contains the true label. Class-0 and class-1 coverage are reported separately because the method is Mondrian/class-conditional. These observed rates are diagnostics, not guarantees for individual cases.

## Primary-seed uncertainty intervals

| Primary-seed quantity | Estimate | 95% bootstrap CI |
| --- | --- | --- |
| automation_coverage | 0.8920 | [0.8767, 0.9077] |
| empirical_set_coverage | 0.8920 | [0.8770, 0.9073] |
| empirical_class_zero_coverage | 0.8889 | [0.8740, 0.9042] |
| empirical_class_one_coverage | 0.9804 | [0.9325, 1.0000] |
| selective_risk | 0.0045 | [0.0015, 0.0082] |
| system_accuracy_assuming_correct_review | 0.9960 | [0.9927, 0.9987] |
| proposed_mean_cost | 0.0707 | [0.0512, 0.1060] |
| baseline_mean_cost | 0.0967 | [0.0427, 0.1777] |
| cost_reduction_vs_baseline | 0.0260 | [-0.0273, 0.0939] |

A positive `cost_reduction_vs_baseline` means the proposed defer policy costs less than the calibrated no-defer policy under the declared cost scenario.

The cost-reduction interval includes zero, so these results do not establish a consistently positive reduction under the declared 95% bootstrap interval.

## Interpretation rules

- Do not claim that calibration improves ranking performance; calibration targets probability reliability.
- Do not claim causal effects or real-world financial savings from this synthetic dataset.
- Distinguish empirical conformal-set coverage from automation coverage; they measure different quantities.
- Treat conformal validity as class-conditional under exchangeability, not a guarantee for every individual case.
- Report automation coverage together with selective risk: a model can reduce risk trivially by deferring nearly everything.
- The engineered thermal, power, and wear-load features closely match documented rules used to generate this synthetic dataset. Their performance gain must not be generalized without external validation.
- Five seeds measure split sensitivity on one dataset; they are not five independent datasets.
- The bootstrap intervals use only the primary seed's test partition.
- Exact XGBoost values can vary across operating systems and CPU architectures even with pinned versions and seeds. Use the saved predictions and run configuration to identify the reported run.
- Validate on an additional real predictive-maintenance dataset before targeting a selective Q1 journal.

## Generated files

- `predictive_metrics_by_seed.csv`: accuracy, ranking, and calibration results.
- `selective_metrics_by_seed.csv`: no-defer versus conformal-defer policies.
- `conformal_metrics_by_seed.csv`: empirical prediction-set validity and set-size diagnostics.
- `conformal_metrics_summary.csv`: five-seed summary by conformal alpha.
- `alpha_tradeoff.csv`: sensitivity to the conformal error level.
- `bootstrap_ci_primary_seed.csv`: paired uncertainty intervals.
- `data_quality_audit.json`: dataset integrity and documented-rule consistency checks.
- `plots/`: reliability, risk–coverage, alpha trade-off, and feature-importance figures.
- `models/`: primary-seed fitted model, calibrator, conformal thresholds, and feature schema.
