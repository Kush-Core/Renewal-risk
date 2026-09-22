#!/usr/bin/env bash
# Rebuild everything from the raw CSVs, in module order, then test and lint.
# Prints the wall time of every step so slow steps are visible.
# FROM=<step> starts at that step and reuses the tables built before it.
set -euo pipefail
PY="${PY:-uv run python}"
STEPS="load audit clean decisions validate_label features km baseline model evaluate horizon explain score monitor"
FROM="${FROM:-load}"
[[ " ${STEPS} " == *" ${FROM} "* ]] || { echo "unknown FROM step: ${FROM}"; exit 1; }
total=$SECONDS
started=0
for step in ${STEPS}; do
  [[ "${step}" == "${FROM}" ]] && started=1
  [[ ${started} -eq 1 ]] || continue
  start=$SECONDS
  echo "== ${step}"
  $PY -m "renewal_risk.${step}"
  echo "-- ${step}: $((SECONDS - start)) s"
done
$PY -m pytest -q
$PY -m ruff check renewal_risk tests scripts
echo "-- total: $((SECONDS - total)) s"