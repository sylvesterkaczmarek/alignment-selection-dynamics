#!/usr/bin/env bash
set -euo pipefail
python -m experiments.run_all --config configs/reference.yaml --out results
