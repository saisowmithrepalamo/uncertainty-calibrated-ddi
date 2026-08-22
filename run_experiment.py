#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPOSITORY_ROOT / ".matplotlib-cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(os.cpu_count() or 1, 8)))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ddi.config import ExperimentConfig
from ddi.experiment import run_experiment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the uncertainty-calibrated decision-intelligence experiment on AI4I."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "paper"],
        default="paper",
        help="quick: one seed; paper: five seeds and larger bootstrap (default)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for reports, tables, models, and plots",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory used to cache the UCI CSV"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the dataset again even if it is cached",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = ExperimentConfig.for_mode(
        args.mode,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    try:
        result = run_experiment(config, force_download=args.force_download)
    except Exception:
        logging.exception("Experiment failed")
        return 1
    print(f"\nCompleted in {float(result['elapsed_seconds']) / 60.0:.2f} minutes")
    print(f"Open: {result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
