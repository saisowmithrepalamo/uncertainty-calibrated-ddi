from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from ddi.calibration import PlattCalibrator
from ddi.config import ExperimentConfig
from ddi.conformal import MondrianConformalClassifier
from ddi.data import (
    CSV_NAME,
    DATASET_DOI,
    TARGET,
    dataset_quality_audit,
    file_sha256,
    load_ai4i,
    make_four_way_split,
    split_summary,
)
from ddi.features import build_feature_matrix
from ddi.metrics import (
    bootstrap_metric_intervals,
    conformal_set_metrics,
    predictive_metrics,
    selective_metrics,
    summarize_across_seeds,
)
from ddi.models import (
    BOOSTING_BACKEND,
    XGBOOST_PACKAGE_VERSION,
    model_feature_importance,
    positive_probability,
    train_boosted_tree,
    train_logistic_baseline,
)
from ddi.plots import (
    plot_alpha_tradeoff,
    plot_feature_importance,
    plot_reliability,
    plot_risk_coverage,
)


LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mean_std_table(
    frame: pd.DataFrame, label_column: str, metrics: list[str]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for label, group in frame.groupby(label_column, sort=False):
        row = [str(label)]
        for metric in metrics:
            mean = float(group[metric].mean())
            std = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row.append(f"{mean:.4f} ± {std:.4f}")
        rows.append(row)
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _write_report(
    output_dir: Path,
    config: ExperimentConfig,
    predictive_frame: pd.DataFrame,
    selective_frame: pd.DataFrame,
    conformal_frame: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    data_audit: dict[str, int | float | bool],
    dataset_sha256: str,
    elapsed_seconds: float,
) -> None:
    predictive_metrics_to_show = ["pr_auc", "roc_auc", "brier", "ece", "recall", "f1"]
    selective_metrics_to_show = [
        "automation_coverage",
        "deferral_rate",
        "selective_risk",
        "system_accuracy_assuming_correct_review",
        "mean_decision_cost",
    ]
    conformal_metrics_to_show = [
        "empirical_set_coverage",
        "empirical_class_zero_coverage",
        "empirical_class_one_coverage",
        "mean_prediction_set_size",
        "empty_set_rate",
        "ambiguous_set_rate",
    ]
    predictive_table = _markdown_table(
        ["Model", "PR-AUC", "ROC-AUC", "Brier", "ECE", "Recall", "F1"],
        _mean_std_table(predictive_frame, "model", predictive_metrics_to_show),
    )
    selective_table = _markdown_table(
        [
            "Policy",
            "Automation coverage",
            "Deferral rate",
            "Selective risk",
            "System accuracy*",
            "Mean cost",
        ],
        _mean_std_table(selective_frame, "policy", selective_metrics_to_show),
    )
    primary_conformal = conformal_frame[
        np.isclose(conformal_frame["alpha"], config.primary_alpha)
    ].assign(setting=f"alpha={config.primary_alpha:.2f}")
    conformal_table = _markdown_table(
        [
            "Setting",
            "Empirical set coverage",
            "Class-0 coverage",
            "Class-1 coverage",
            "Mean set size",
            "Empty-set rate",
            "Ambiguous-set rate",
        ],
        _mean_std_table(primary_conformal, "setting", conformal_metrics_to_show),
    )
    bootstrap_rows = [
        [
            str(row.metric),
            f"{row.estimate:.4f}",
            f"[{row.ci_lower_95:.4f}, {row.ci_upper_95:.4f}]",
        ]
        for row in bootstrap_frame.itertuples(index=False)
    ]
    bootstrap_table = _markdown_table(
        ["Primary-seed quantity", "Estimate", "95% bootstrap CI"], bootstrap_rows
    )
    cost_interval = bootstrap_frame[
        bootstrap_frame["metric"] == "cost_reduction_vs_baseline"
    ].iloc[0]
    cost_interval_statement = (
        "The cost-reduction interval includes zero, so these results do not "
        "establish a consistently positive reduction under the declared 95% "
        "bootstrap interval."
        if float(cost_interval.ci_lower_95) <= 0.0 <= float(cost_interval.ci_upper_95)
        else "The cost-reduction interval does not include zero under this "
        "primary-seed bootstrap analysis."
    )
    report = f"""# Experiment report

## Study

**Title:** Data-Driven Decision Intelligence with Uncertainty-Calibrated AI: Learning When to Predict and When to Defer

The experiment used the UCI AI4I 2020 Predictive Maintenance dataset ({DATASET_DOI}). It compared raw-feature baselines, a knowledge-feature {BOOSTING_BACKEND} model, independent Platt calibration, and a class-conditional (Mondrian) split-conformal defer policy.

Runtime: **{elapsed_seconds / 60.0:.2f} minutes**  
Seeds: **{', '.join(map(str, config.seeds))}**  
Primary conformal error level: **{config.primary_alpha:.2f}**  
Boosting backend: **{BOOSTING_BACKEND}**

## Dataset integrity audit

- Dataset SHA-256: `{dataset_sha256}`
- Rows: **{data_audit['rows']}**
- Target failures: **{data_audit['target_failures']}** ({100.0 * float(data_audit['target_failure_rate']):.3f}%)
- Missing required cells: **{data_audit['missing_required_cells']}**
- Complete duplicate rows: **{data_audit['duplicate_full_rows']}**

The official file contains **{data_audit['target_failure_mode_or_rule_mismatches']}** target/failure-mode inconsistencies relative to the dataset documentation's OR rule: **{data_audit['target_positive_without_failure_mode_rows']}** target-positive rows have no positive failure-mode indicator, and **{data_audit['failure_mode_positive_with_target_zero_rows']}** target-zero rows have at least one positive failure-mode indicator. Failure-mode columns are excluded from all model features. Labels are preserved exactly as published; the audit does not silently alter them.

## Predictive and calibration results

{predictive_table}

Lower Brier score and ECE are better. PR-AUC is particularly important because machine failure is rare.

## Decision results

{selective_table}

*System accuracy assumes every deferred case is resolved correctly by a human reviewer. The corresponding review cost is included in mean decision cost.

The cost scenario is predeclared as false positive={config.false_positive_cost:g}, false negative={config.false_negative_cost:g}, and defer={config.deferral_cost:g}. These are scenario weights, not measured currency values.

`Automation coverage` is the fraction of cases receiving an automatic decision; it is exactly one minus the deferral rate. It is not conformal prediction-set coverage.

## Empirical conformal-set validation

{conformal_table}

Empirical set coverage is the fraction of test examples whose conformal prediction set contains the true label. Class-0 and class-1 coverage are reported separately because the method is Mondrian/class-conditional. These observed rates are diagnostics, not guarantees for individual cases.

## Primary-seed uncertainty intervals

{bootstrap_table}

A positive `cost_reduction_vs_baseline` means the proposed defer policy costs less than the calibrated no-defer policy under the declared cost scenario.

{cost_interval_statement}

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
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_experiment(
    config: ExperimentConfig, force_download: bool = False
) -> dict[str, Path | float]:
    start_time = time.perf_counter()
    output_dir = config.output_path.resolve()
    data_dir = config.data_path.resolve()
    plot_dir = output_dir / "plots"
    model_dir = output_dir / "models"
    for directory in [output_dir, plot_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    frame = load_ai4i(data_dir, force_download=force_download)
    data_path = data_dir / CSV_NAME
    data_audit = dataset_quality_audit(frame)
    dataset_sha256 = file_sha256(data_path)
    data_audit["dataset_sha256"] = dataset_sha256
    _write_json(output_dir / "data_quality_audit.json", data_audit)
    _write_json(
        output_dir / "run_config.json",
        {
            **config.as_dict(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "boosting_backend": BOOSTING_BACKEND,
            "xgboost_package_version": XGBOOST_PACKAGE_VERSION,
            "dataset_doi": DATASET_DOI,
            "dataset_bytes": data_path.stat().st_size,
            "dataset_sha256": dataset_sha256,
        },
    )
    if int(data_audit["target_failure_mode_or_rule_mismatches"]) > 0:
        LOGGER.warning(
            "Dataset audit found %d target/failure-mode OR-rule mismatches; "
            "labels are preserved and failure-mode columns remain excluded",
            data_audit["target_failure_mode_or_rule_mismatches"],
        )
    y = frame[TARGET].to_numpy(dtype=int)
    X_raw = build_feature_matrix(frame, include_knowledge=False)
    X_knowledge = build_feature_matrix(frame, include_knowledge=True)
    LOGGER.info(
        "Loaded %s rows; failure rate %.3f%%; %d knowledge-model features",
        f"{len(frame):,}",
        100.0 * y.mean(),
        X_knowledge.shape[1],
    )

    predictive_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    split_rows: list[pd.DataFrame] = []
    primary_artifacts: dict[str, Any] = {}

    for seed_index, seed in enumerate(config.seeds):
        LOGGER.info("Running seed %d (%d/%d)", seed, seed_index + 1, len(config.seeds))
        split = make_four_way_split(
            y,
            seed=seed,
            train_fraction=config.train_fraction,
            probability_calibration_fraction=config.probability_calibration_fraction,
            conformal_calibration_fraction=config.conformal_calibration_fraction,
            test_fraction=config.test_fraction,
        )
        split_frame = split_summary(split, y)
        split_frame.insert(0, "seed", seed)
        split_rows.append(split_frame)

        y_train = y[split.train]
        y_probability = y[split.probability_calibration]
        y_conformal = y[split.conformal_calibration]
        y_test = y[split.test]

        logistic = train_logistic_baseline(X_raw.iloc[split.train], y_train, seed)
        raw_xgb = train_boosted_tree(
            X_raw.iloc[split.train],
            y_train,
            seed,
            config.xgb_estimators,
            config.xgb_n_jobs,
        )
        knowledge_xgb = train_boosted_tree(
            X_knowledge.iloc[split.train],
            y_train,
            seed,
            config.xgb_estimators,
            config.xgb_n_jobs,
        )

        p_logistic = positive_probability(logistic, X_raw.iloc[split.test])
        p_raw_xgb = positive_probability(raw_xgb, X_raw.iloc[split.test])
        p_knowledge_uncalibrated = positive_probability(
            knowledge_xgb, X_knowledge.iloc[split.test]
        )

        calibrator = PlattCalibrator().fit(
            positive_probability(
                knowledge_xgb, X_knowledge.iloc[split.probability_calibration]
            ),
            y_probability,
        )
        p_conformal_calibration = calibrator.predict(
            positive_probability(
                knowledge_xgb, X_knowledge.iloc[split.conformal_calibration]
            )
        )
        p_calibrated_test = calibrator.predict(p_knowledge_uncalibrated)

        for model_name, probabilities in [
            ("Logistic regression — raw sensors", p_logistic),
            (f"{BOOSTING_BACKEND} — raw sensors", p_raw_xgb),
            (f"{BOOSTING_BACKEND} — knowledge features", p_knowledge_uncalibrated),
            (f"{BOOSTING_BACKEND} — knowledge + Platt calibration", p_calibrated_test),
        ]:
            predictive_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    **predictive_metrics(y_test, probabilities),
                }
            )

        baseline_decisions = (p_calibrated_test >= 0.5).astype(int)
        baseline_policy_metrics = selective_metrics(
            y_test,
            baseline_decisions,
            p_calibrated_test,
            config.false_positive_cost,
            config.false_negative_cost,
            config.deferral_cost,
        )
        selective_rows.append(
            {
                "seed": seed,
                "policy": f"Calibrated {BOOSTING_BACKEND} — no defer",
                **baseline_policy_metrics,
            }
        )

        primary_conformal: MondrianConformalClassifier | None = None
        primary_decisions: np.ndarray | None = None
        primary_prediction_sets: np.ndarray | None = None
        for alpha in config.alpha_grid:
            conformal = MondrianConformalClassifier(alpha=alpha).fit(
                p_conformal_calibration, y_conformal
            )
            prediction_sets = conformal.prediction_sets(p_calibrated_test)
            decisions = conformal.predict_or_defer(p_calibrated_test)
            decision_metrics = selective_metrics(
                y_test,
                decisions,
                p_calibrated_test,
                config.false_positive_cost,
                config.false_negative_cost,
                config.deferral_cost,
            )
            set_metrics = conformal_set_metrics(y_test, prediction_sets)
            conformal_rows.append({"seed": seed, "alpha": alpha, **set_metrics})
            alpha_rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    **decision_metrics,
                    **set_metrics,
                }
            )
            if abs(alpha - config.primary_alpha) < 1e-12:
                primary_conformal = conformal
                primary_decisions = decisions
                primary_prediction_sets = prediction_sets
                selective_rows.append(
                    {
                        "seed": seed,
                        "policy": "Proposed Mondrian conformal defer",
                        **decision_metrics,
                    }
                )
                assert conformal.thresholds is not None
                threshold_rows.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "class_zero_nonconformity_quantile": conformal.thresholds.class_zero,
                        "class_one_nonconformity_quantile": conformal.thresholds.class_one,
                    }
                )

        if (
            primary_conformal is None
            or primary_decisions is None
            or primary_prediction_sets is None
        ):
            raise RuntimeError("primary_alpha must be present in alpha_grid")

        if seed_index == 0:
            primary_artifacts = {
                "seed": seed,
                "y_test": y_test,
                "p_uncalibrated": p_knowledge_uncalibrated,
                "p_calibrated": p_calibrated_test,
                "baseline_decisions": baseline_decisions,
                "proposed_decisions": primary_decisions,
                "prediction_sets": primary_prediction_sets,
                "knowledge_model": knowledge_xgb,
                "calibrator": calibrator,
                "conformal": primary_conformal,
                "feature_names": X_knowledge.columns.tolist(),
                "split": split,
            }

    predictive_frame = pd.DataFrame(predictive_rows)
    selective_frame = pd.DataFrame(selective_rows)
    conformal_frame = pd.DataFrame(conformal_rows)
    alpha_frame = pd.DataFrame(alpha_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    split_frame_all = pd.concat(split_rows, ignore_index=True)

    predictive_frame.to_csv(output_dir / "predictive_metrics_by_seed.csv", index=False)
    selective_frame.to_csv(output_dir / "selective_metrics_by_seed.csv", index=False)
    conformal_frame.to_csv(output_dir / "conformal_metrics_by_seed.csv", index=False)
    alpha_frame.to_csv(output_dir / "alpha_tradeoff.csv", index=False)
    threshold_frame.to_csv(output_dir / "conformal_thresholds.csv", index=False)
    split_frame_all.to_csv(output_dir / "split_counts.csv", index=False)
    summarize_across_seeds(predictive_frame, ["model"]).to_csv(
        output_dir / "predictive_metrics_summary.csv", index=False
    )
    summarize_across_seeds(selective_frame, ["policy"]).to_csv(
        output_dir / "selective_metrics_summary.csv", index=False
    )
    summarize_across_seeds(conformal_frame, ["alpha"]).to_csv(
        output_dir / "conformal_metrics_summary.csv", index=False
    )

    bootstrap_frame = bootstrap_metric_intervals(
        primary_artifacts["y_test"],
        primary_artifacts["proposed_decisions"],
        primary_artifacts["prediction_sets"],
        primary_artifacts["baseline_decisions"],
        primary_artifacts["p_calibrated"],
        config.false_positive_cost,
        config.false_negative_cost,
        config.deferral_cost,
        repetitions=config.bootstrap_repetitions,
        seed=int(primary_artifacts["seed"]) + 10_000,
    )
    bootstrap_frame.to_csv(output_dir / "bootstrap_ci_primary_seed.csv", index=False)

    prediction_export = pd.DataFrame(
        {
            "row_index": primary_artifacts["split"].test,
            "true_failure": primary_artifacts["y_test"],
            "uncalibrated_failure_probability": primary_artifacts["p_uncalibrated"],
            "calibrated_failure_probability": primary_artifacts["p_calibrated"],
            "baseline_decision": primary_artifacts["baseline_decisions"],
            "proposed_decision": primary_artifacts["proposed_decisions"],
            "conformal_includes_no_failure": primary_artifacts["prediction_sets"][
                :, 0
            ],
            "conformal_includes_failure": primary_artifacts["prediction_sets"][:, 1],
        }
    )
    prediction_export["conformal_set_size"] = (
        prediction_export["conformal_includes_no_failure"].astype(int)
        + prediction_export["conformal_includes_failure"].astype(int)
    )
    prediction_export["conformal_contains_true_label"] = np.where(
        prediction_export["true_failure"].to_numpy(dtype=int) == 0,
        prediction_export["conformal_includes_no_failure"],
        prediction_export["conformal_includes_failure"],
    )
    prediction_export["proposed_action"] = prediction_export["proposed_decision"].map(
        {-1: "defer", 0: "predict_no_failure", 1: "predict_failure"}
    )
    prediction_export.to_csv(output_dir / "primary_seed_test_predictions.csv", index=False)

    joblib.dump(
        primary_artifacts["knowledge_model"],
        model_dir / "boosting_knowledge_model.joblib",
    )
    joblib.dump(primary_artifacts["calibrator"], model_dir / "platt_calibrator.joblib")
    _write_json(model_dir / "feature_schema.json", primary_artifacts["feature_names"])
    conformal_thresholds = primary_artifacts["conformal"].thresholds
    _write_json(
        model_dir / "conformal_policy.json",
        {
            "alpha": conformal_thresholds.alpha,
            "class_zero_nonconformity_quantile": conformal_thresholds.class_zero,
            "class_one_nonconformity_quantile": conformal_thresholds.class_one,
            "decision_encoding": {
                "-1": "defer",
                "0": "predict_no_failure",
                "1": "predict_failure",
            },
        },
    )

    plot_reliability(
        primary_artifacts["y_test"],
        primary_artifacts["p_uncalibrated"],
        primary_artifacts["p_calibrated"],
        plot_dir / "reliability_diagram.png",
    )
    plot_risk_coverage(
        primary_artifacts["y_test"],
        primary_artifacts["p_calibrated"],
        plot_dir / "risk_coverage_curve.png",
    )
    plot_alpha_tradeoff(alpha_frame, plot_dir / "alpha_tradeoff.png")
    feature_importances = model_feature_importance(
        primary_artifacts["knowledge_model"],
        X_knowledge.iloc[primary_artifacts["split"].test],
        primary_artifacts["y_test"],
        seed=int(primary_artifacts["seed"]),
    )
    plot_feature_importance(
        primary_artifacts["feature_names"],
        feature_importances,
        plot_dir / "feature_importance.png",
    )

    elapsed = time.perf_counter() - start_time
    _write_report(
        output_dir,
        config,
        predictive_frame,
        selective_frame,
        conformal_frame,
        bootstrap_frame,
        data_audit,
        dataset_sha256,
        elapsed,
    )
    LOGGER.info("Experiment finished in %.2f minutes", elapsed / 60.0)
    LOGGER.info("Report: %s", output_dir / "report.md")
    return {
        "output_dir": output_dir,
        "report": output_dir / "report.md",
        "elapsed_seconds": elapsed,
    }
