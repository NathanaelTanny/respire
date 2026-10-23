#!/usr/bin/env bash
# Evaluate a saved bundle on the held-out split and write the comparison table.
#
#   scripts/launch_eval.sh runs/rounds/bundle.pt
set -euo pipefail

BUNDLE="${1:-runs/rounds/bundle.pt}"
shift || true

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

python3 -m respire.cli.evaluate --bundle "${BUNDLE}" --config configs/round/primary.yaml "$@"
python3 -m respire.cli.compare --bundle "${BUNDLE}" --config configs/round/primary.yaml "$@"
