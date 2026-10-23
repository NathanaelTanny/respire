#!/usr/bin/env bash
# Fit the encoder and run the improvement rounds.
#
#   scripts/launch_train.sh configs/round/primary.yaml
#   scripts/launch_train.sh configs/round/_smoke.yaml --output runs/smoke
#
# The primary document is the reported configuration. It assumes one GPU and roughly two days of
# wall clock for the full protocol; the smoke document finishes in about two minutes on a laptop.
set -euo pipefail

CONFIG="${1:-configs/round/primary.yaml}"
shift || true

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

python3 -m respire.cli.train --config "${CONFIG}" --log-level INFO "$@"
