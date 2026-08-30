#!/usr/bin/env bash
# run_parameter.sh -- Parameter studies, Figures 2-5.
#
#   PR   (Fig. 2): python PRTest.py    -> PRaPlot.py + PRbPlot.py
#   RR   (Fig. 3): python RRTest.py    -> RRPlot.py
#   CR   (Fig. 4): python CRTest.py    -> CRPlot.py
#   IHOP^M (Fig. 5): python ihopMTest.py -> ihopMPlot.py
#
# Datasets: Enron and Lucene (as in README; the Lucene runs draw the
# corresponding figures of the paper's Appendix B).
# Estimated total runtime (reference machine): PR/RR/CR not recorded;
# IHOP^M ~30 min (Enron), ~1 h (Lucene).
#
# Execution order: all experiment scripts first (write pic_pkl/),
# then all plotting scripts (write pic/pictures/).
#
# Usage: bash scripts/run_parameter.sh [-d <dataset>]
#   -d <dataset>   only run the given dataset (default: Enron and Lucene)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

datasets=(Enron Lucene)
while (($#)); do
  case "$1" in
    -d) datasets=("${2:?}"); shift 2 ;;
    *)  echo "unknown option: $1"; exit 2 ;;
  esac
done

echo "== Parameter studies (Figs. 2-5) on: ${datasets[*]} =="

# Phase A: run the experiments (pkls into pic_pkl/)
for d in "${datasets[@]}"; do
  echo "-- PRTest ($d) [Fig. 2] --"
  python PRTest.py -d "$d"
  echo "-- RRTest ($d) [Fig. 3] --"
  python RRTest.py -d "$d"
  echo "-- CRTest ($d) [Fig. 4] --"
  python CRTest.py -d "$d"
  echo "-- ihopMTest ($d) [Fig. 5] --"
  python ihopMTest.py -d "$d"
done

# Phase B: generate the figures from pic/
mkdir -p pic/pictures
for d in "${datasets[@]}"; do
  ( cd pic && python PRaPlot.py -d "$d" && python PRbPlot.py -d "$d" )
  ( cd pic && python RRPlot.py -d "$d" )
  ( cd pic && python CRPlot.py -d "$d" )
  ( cd pic && python ihopMPlot.py -d "$d" )
done

echo "== done; figures in pic/pictures/ =="