#!/usr/bin/env bash
set -euo pipefail
python -m experiments.run_all --seeds 7 17 29 41 53 --out results
