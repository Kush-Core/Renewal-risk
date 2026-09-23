#!/usr/bin/env bash
# Rebuild everything from the raw CSVs, in pipeline order, then test and lint.
# Prints the wall time of every step so slow steps are visible.
#
#   FROM=<step>      start at that step and reuse the tables built before it
#   ONLY="a b c"     run just these steps, in the order given
#   SKIP_CHECKS=1    skip pytest and ruff
#   PY=...           interpreter to use
set -euo pipefail

# The project interpreter directly: `uv run` re-resolves the environment on each
# of the fourteen steps. Fall back to uv when the venv is not there.
PY="${PY:-.venv/bin/python}"
[[ -x "${PY%% *}" ]] || PY="uv run python"

STEPS="load audit clean decisions validate_label features km baseline model evaluate horizon explain score monitor"

# How much of the machine DuckDB may take. The SQL steps have it to themselves.
# The modelling steps have to leave room for pandas frames and design matrices,
# and DuckDB sitting on cached blocks it no longer needs is exactly what pushes
# them into swap; over its limit it spills to disk instead, which is far cheaper.
SQL_STEPS="load audit clean decisions validate_label features"
SQL_FRACTION="${SQL_FRACTION:-0.60}"
ML_FRACTION="${ML_FRACTION:-0.20}"

run="${ONLY:-}"
if [[ -z "${run}" ]]; then
  FROM="${FROM:-load}"
  [[ " ${STEPS} " == *" ${FROM} "* ]] || { echo "unknown FROM step: ${FROM}"; exit 1; }
  started=0
  for step in ${STEPS}; do
    [[ "${step}" == "${FROM}" ]] && started=1
    if [[ ${started} -eq 1 ]]; then run="${run} ${step}"; fi
  done
fi

total=$SECONDS
for step in ${run}; do
  [[ " ${STEPS} " == *" ${step} "* ]] || { echo "unknown step: ${step}"; exit 1; }
  if [[ " ${SQL_STEPS} " == *" ${step} "* ]]; then f="${SQL_FRACTION}"; else f="${ML_FRACTION}"; fi
  start=$SECONDS
  echo "== ${step}"
  DUCKDB_MEMORY_FRACTION="${f}" $PY -m "renewal_risk.${step}"
  echo "-- ${step}: $((SECONDS - start)) s"
done

if [[ -z "${SKIP_CHECKS:-}" ]]; then
  $PY -m pytest -q
  $PY -m ruff check renewal_risk tests scripts
fi
echo "-- total: $((SECONDS - total)) s"
