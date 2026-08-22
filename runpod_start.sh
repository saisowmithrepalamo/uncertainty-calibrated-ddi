#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13)' 2>/dev/null; then
  echo "Python 3.10, 3.11, or 3.12 is required. Current interpreter: $($PYTHON_BIN --version 2>&1)"
  exit 2
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pytest -q
python run_experiment.py --mode paper --output-dir results --data-dir data

echo
echo "Finished. Read results/report.md and inspect results/plots/."
