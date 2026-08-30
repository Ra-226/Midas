#!/usr/bin/env bash
# run_ablation.sh -- Ablation studies, Figures 6-8.
#
#   Fig. 6 co-absence leakage:   ablationLeakageTest.py -> ablationLeakagePlot.py
#   Fig. 7 iterative refinement: ablationRRTest.py      -> ablationRRPlot.py
#   Fig. 8 incremental compute:  ablationOptimizeM/NTest -> ablationOptimizeM/NPlot
#
# Dataset: Enron (fixed).  Scenarios: S1 and S2 for Figs. 6-7;
# Fig. 8 takes no scenario argument.
# Estimated total runtime (reference machine): ~40 min
# (<2 min per leakage run, ~5 min per RR run, ~10/15 min for the optimize tests).
#
# Execution order: all experiment scripts first (write pic_pkl/),
# then all plotting scripts (write pic/pictures/).
#
# Usage: bash scripts/run_ablation.sh [-s <scenario>]
#   -s <scenario>   only run the given scenario (default: S1 and S2)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

scenarios=(S1 S2)
while (($#)); do
  case "$1" in
    -s) scenarios=("${2:?}"); shift 2 ;;
    *)  echo "unknown option: $1"; exit 2 ;;
  esac
done

echo "== Ablation studies (Figs. 6-8) on Enron, scenarios: ${scenarios[*]} =="

# Phase A: run the experiments (pkls into pic_pkl/)
for s in "${scenarios[@]}"; do
  echo "-- ablationLeakageTest ($s) [Fig. 6] --"
  python ablationLeakageTest.py -s "$s"
  echo "-- ablationRRTest ($s) [Fig. 7] --"
  python ablationRRTest.py -s "$s"
done
echo "-- ablationOptimizeMTest / ablationOptimizeNTest [Fig. 8] --"
python ablationOptimizeMTest.py
python ablationOptimizeNTest.py

# Phase B: generate the figures from pic/
mkdir -p pic/pictures
for s in "${scenarios[@]}"; do
  ( cd pic && python ablationLeakagePlot.py -s "$s" )
  ( cd pic && python ablationRRPlot.py -s "$s" )
done
( cd pic && python ablationOptimizeMPlot.py && python ablationOptimizeNPlot.py )

echo "== done; figures in pic/pictures/ =="