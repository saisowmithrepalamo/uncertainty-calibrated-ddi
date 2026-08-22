#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(os.cpu_count() or 1, 8)))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ddi.features import build_feature_matrix
from ddi.models import positive_probability


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a fitted AI4I defer policy to a CSV.")
    parser.add_argument("--input", required=True, help="CSV containing the six raw sensor fields")
    parser.add_argument("--model-dir", default="results/models")
    parser.add_argument("--output", default="predictions.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    model_dir = Path(args.model_dir)
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    features = build_feature_matrix(frame, include_knowledge=True)
    expected_features = json.loads((model_dir / "feature_schema.json").read_text())
    missing = set(expected_features).difference(features.columns)
    if missing:
        raise ValueError(f"Input is missing encoded features: {sorted(missing)}")
    features = features.reindex(columns=expected_features, fill_value=0.0)

    model = joblib.load(model_dir / "boosting_knowledge_model.joblib")
    calibrator = joblib.load(model_dir / "platt_calibrator.joblib")
    policy = json.loads((model_dir / "conformal_policy.json").read_text())

    raw_probability = positive_probability(model, features)
    probability = calibrator.predict(raw_probability)
    include_zero = probability <= policy["class_zero_nonconformity_quantile"]
    include_one = (1.0 - probability) <= policy["class_one_nonconformity_quantile"]
    base_prediction = (probability >= 0.5).astype(int)
    decision = np.full(len(frame), -1, dtype=int)
    decision[include_zero & ~include_one & (base_prediction == 0)] = 0
    decision[include_one & ~include_zero & (base_prediction == 1)] = 1

    output = frame.copy()
    output["failure_probability_uncalibrated"] = raw_probability
    output["failure_probability_calibrated"] = probability
    output["conformal_includes_no_failure"] = include_zero
    output["conformal_includes_failure"] = include_one
    output["decision"] = decision
    output["action"] = pd.Series(decision).map(
        {-1: "defer", 0: "predict_no_failure", 1: "predict_failure"}
    )
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} decisions to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
