#!/usr/bin/env bash
# run_defense.sh -- Attack against the defenses, Figures 14-15.
#
#   Fig. 14 CLRZ:  clrzTest.py -> clrzPlot.py
#   Fig. 15 OSSE:  osseTest.py -> ossePlot.py
#
# Default combinations (as in README): {Enron,Lucene} x {S1,S2,S3}
# (6 runs per attack, ~1.5 h each).
# Estimated total runtime: ~18 h on the reference machine.
#
# Execution order: all experiment scripts first (write pic_pkl/),
# then all plotting scripts (write pic/pictures/).
#
# Usage: bash scripts/run_defense.sh [-d <dataset>] [-s <scenario>]
#   -d <dataset>   restrict the dataset dimension (default: Enron and Lucene)
#   -s <scenario>  restrict the scenario dimension (default: S1, S2 and S3)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

data=(Enron Lucene)
scen=(S1 S2 S3)

opt_d=""
opt_s=""
while (($#)); do
  case "$1" in
    -d) opt_d="${2:?}"; shift 2 ;;
    -s) opt_s="${2:?}"; shift 2 ;;
    *)  echo "unknown option: $1"; exit 2 ;;
  esac
done
if [[ -n "$opt_d" ]]; then data=("$opt_d"); fi
if [[ -n "$opt_s" ]]; then scen=("$opt_s"); fi

echo "== Defense evaluation (Figs. 14-15): ${data[*]} x ${scen[*]} =="

# Phase A: run the experiments (pkls into pic_pkl/)
for d in "${data[@]}"; do
  for s in "${scen[@]}"; do
    echo "-- clrzTest ($d, $s) [Fig. 14] --"
    python clrzTest.py -d "$d" -s "$s"
  done
done
for d in "${data[@]}"; do
  for s in "${scen[@]}"; do
    echo "-- osseTest ($d, $s) [Fig. 15] --"
    python osseTest.py -d "$d" -s "$s"
  done
done

# Phase B: generate the figures from pic/
mkdir -p pic/pictures
for d in "${data[@]}"; do
  for s in "${scen[@]}"; do
    ( cd pic && python clrzPlot.py -d "$d" -s "$s" )
  done
done
for d in "${data[@]}"; do
  for s in "${scen[@]}"; do
    ( cd pic && python ossePlot.py -d "$d" -s "$s" )
  done
done

echo "== done; figures in pic/pictures/ =="