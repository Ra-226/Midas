#!/usr/bin/env bash
# run_comparison.sh -- Main comparisons, Figures 9-13.
#
#   Fig. 9  auxiliary knowledge |D_sim|: auxTest        -> auxPlot
#   Fig. 10 keyword space n:             nTest          -> nPlot + nTimePlot
#   Fig. 11 similar runtime on n:        nTestLimitedTime -> limitedTimeNPlot
#   Fig. 12 number of observed queries m: mTest         -> mPlot + mTimePlot
#   Fig. 13 similar runtime on m:        mTestLimitedTime -> limitedTimeMPlot
#
# Default combinations (as in README):
#   Fig. 9 : {Enron,Lucene} x {S1,S2,S3}   (6 runs, ~2.5 h each)
#   Fig. 10: Enron x {S1,S2,S3}            (~2.5 h each)
#   Fig. 11: {Enron,Lucene} x {S1,S2}      (~25 min Enron / ~40 min Lucene)
#   Fig. 12: Enron x {S1,S2,S3}            (~3.5 h each)
#   Fig. 13: {Enron,Lucene} x {S1,S2}      (~10 min each)
# Estimated total runtime: roughly 35 compute-hours for a full run.
#
# Execution order: all experiment scripts first (write pic_pkl/),
# then all plotting scripts (write pic/pictures/).
#
# Usage: bash scripts/run_comparison.sh [-d <dataset>] [-s <scenario>]
#   -d <dataset>   restrict the dataset dimension (default: Enron and Lucene)
#   -s <scenario>  restrict the scenario dimension (default: S1, S2 and S3)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

aux_data=(Enron Lucene)
lim_data=(Enron Lucene)
aux_scen=(S1 S2 S3)
n_scen=(S1 S2 S3)
m_scen=(S1 S2 S3)
lim_scen=(S1 S2)

opt_d=""
opt_s=""
while (($#)); do
  case "$1" in
    -d) opt_d="${2:?}"; shift 2 ;;
    -s) opt_s="${2:?}"; shift 2 ;;
    *)  echo "unknown option: $1"; exit 2 ;;
  esac
done
if [[ -n "$opt_d" ]]; then aux_data=("$opt_d"); lim_data=("$opt_d"); fi
if [[ -n "$opt_s" ]]; then aux_scen=("$opt_s"); n_scen=("$opt_s"); m_scen=("$opt_s"); lim_scen=("$opt_s"); fi

echo "== Main comparisons (Figs. 9-13) =="
echo "   aux (Fig. 9):      ${aux_data[*]} x ${aux_scen[*]}"
echo "   n   (Fig. 10):     Enron x ${n_scen[*]}"
echo "   n-t (Fig. 11):     ${lim_data[*]} x ${lim_scen[*]}"
echo "   m   (Fig. 12):     Enron x ${m_scen[*]}"
echo "   m-t (Fig. 13):     ${lim_data[*]} x ${lim_scen[*]}"

# Phase A: run the experiments (pkls into pic_pkl/)
for d in "${aux_data[@]}"; do
  for s in "${aux_scen[@]}"; do
    echo "-- auxTest ($d, $s) [Fig. 9] --"
    python auxTest.py -d "$d" -s "$s"
  done
done
for s in "${n_scen[@]}"; do
  echo "-- nTest (Enron, $s) [Fig. 10] --"
  python nTest.py -d Enron -s "$s"
done
for d in "${lim_data[@]}"; do
  for s in "${lim_scen[@]}"; do
    echo "-- nTestLimitedTime ($d, $s) [Fig. 11] --"
    python nTestLimitedTime.py -d "$d" -s "$s"
  done
done
for s in "${m_scen[@]}"; do
  echo "-- mTest (Enron, $s) [Fig. 12] --"
  python mTest.py -d Enron -s "$s"
done
for d in "${lim_data[@]}"; do
  for s in "${lim_scen[@]}"; do
    echo "-- mTestLimitedTime ($d, $s) [Fig. 13] --"
    python mTestLimitedTime.py -d "$d" -s "$s"
  done
done

# Phase B: generate the figures from pic/
mkdir -p pic/pictures
for d in "${aux_data[@]}"; do
  for s in "${aux_scen[@]}"; do
    ( cd pic && python auxPlot.py -d "$d" -s "$s" )
  done
done
for s in "${n_scen[@]}"; do
  ( cd pic && python nPlot.py -d Enron -s "$s" && python nTimePlot.py -d Enron -s "$s" )
done
for d in "${lim_data[@]}"; do
  for s in "${lim_scen[@]}"; do
    ( cd pic && python limitedTimeNPlot.py -d "$d" -s "$s" )
  done
done
for s in "${m_scen[@]}"; do
  ( cd pic && python mPlot.py -d Enron -s "$s" && python mTimePlot.py -d Enron -s "$s" )
done
for d in "${lim_data[@]}"; do
  for s in "${lim_scen[@]}"; do
    ( cd pic && python limitedTimeMPlot.py -d "$d" -s "$s" )
  done
done

echo "== done; figures in pic/pictures/ =="