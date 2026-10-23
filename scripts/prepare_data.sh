#!/usr/bin/env bash
# Build the cohort and, when the credentialed exports are present, check their layout.
#
#   scripts/prepare_data.sh                    # cohort only
#   scripts/prepare_data.sh --records /data    # also inspect a record export
#
# The corpora listed in dataset_urls.txt are credentialed and are never copied into the release.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

RECORDS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --records) RECORDS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

python3 -m respire.cli.cohort --config configs/round/primary.yaml

if [[ -n "${RECORDS}" ]]; then
  python3 - "${RECORDS}" <<'PY'
import sys
from pathlib import Path

from respire.data.exports import describe_export

report = describe_export(Path(sys.argv[1]))
for key, value in report.items():
    print(f"{key}: {value}")
PY
fi
