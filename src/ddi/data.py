from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)
DATASET_DOI = "https://doi.org/10.24432/C5HS5C"
CSV_NAME = "ai4i2020.csv"

TARGET = "Machine failure"
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]

REQUIRED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET,
    *LEAKAGE_COLUMNS,
}


@dataclass(frozen=True)
class DataSplits:
    train: np.ndarray
    probability_calibration: np.ndarray
    conformal_calibration: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train,
            "probability_calibration": self.probability_calibration,
            "conformal_calibration": self.conformal_calibration,
            "test": self.test,
        }


def _download_with_retries(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            LOGGER.info("Downloading AI4I data (attempt %d/%d)", attempt, retries)
            response = requests.get(
                url,
                headers={"User-Agent": "uncertainty-calibrated-ddi/1.0"},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.content
            if len(payload) < 100_000:
                raise RuntimeError(
                    f"Downloaded file is unexpectedly small: {len(payload)} bytes"
                )
            temporary_path = destination.with_suffix(".tmp")
            temporary_path.write_bytes(payload)
            temporary_path.replace(destination)
            LOGGER.info(
                "Downloaded %s bytes; SHA256=%s",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
            return
        except (requests.RequestException, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def validate_dataset(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(frame) != 10_000:
        raise ValueError(f"Expected 10,000 rows, found {len(frame):,}")
    target_values = set(frame[TARGET].dropna().unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(f"Target must be binary 0/1, found {target_values}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required dataset columns contain missing values")


def load_ai4i(data_dir: str | Path, force_download: bool = False) -> pd.DataFrame:
    """Download, cache, validate, and return the official UCI AI4I dataset."""

    data_path = Path(data_dir) / CSV_NAME
    if force_download or not data_path.exists():
        _download_with_retries(DATA_URL, data_path)
    frame = pd.read_csv(data_path, encoding="utf-8-sig")
    validate_dataset(frame)
    return frame


def make_four_way_split(
    y: pd.Series | np.ndarray,
    seed: int,
    train_fraction: float,
    probability_calibration_fraction: float,
    conformal_calibration_fraction: float,
    test_fraction: float,
) -> DataSplits:
    """Create four disjoint stratified splits using only row indices."""

    labels = np.asarray(y, dtype=int)
    indices = np.arange(len(labels))

    remaining, test = train_test_split(
        indices,
        test_size=test_fraction,
        stratify=labels,
        random_state=seed,
    )
    conformal_relative = conformal_calibration_fraction / (1.0 - test_fraction)
    remaining, conformal = train_test_split(
        remaining,
        test_size=conformal_relative,
        stratify=labels[remaining],
        random_state=seed + 1,
    )
    probability_relative = probability_calibration_fraction / (
        train_fraction + probability_calibration_fraction
    )
    train, probability = train_test_split(
        remaining,
        test_size=probability_relative,
        stratify=labels[remaining],
        random_state=seed + 2,
    )

    splits = DataSplits(
        train=np.sort(train),
        probability_calibration=np.sort(probability),
        conformal_calibration=np.sort(conformal),
        test=np.sort(test),
    )
    joined = np.concatenate(list(splits.as_dict().values()))
    if len(joined) != len(np.unique(joined)) or set(joined) != set(indices):
        raise AssertionError("Four-way split is not a complete disjoint partition")
    return splits


def split_summary(splits: DataSplits, y: pd.Series | np.ndarray) -> pd.DataFrame:
    labels = np.asarray(y, dtype=int)
    rows: list[dict[str, float | int | str]] = []
    for name, idx in splits.as_dict().items():
        rows.append(
            {
                "split": name,
                "rows": len(idx),
                "failures": int(labels[idx].sum()),
                "failure_rate": float(labels[idx].mean()),
            }
        )
    return pd.DataFrame(rows)
