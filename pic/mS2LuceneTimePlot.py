# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd

with open("../pic_pkl/mS2Lucene.pkl", "rb") as f:
    pkl = pickle.load(f)
fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='m', y='time', hue='attack',
             hue_order=['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"],
             palette=["C1", "C2", 'C3', 'C4', 'c', 'C6', "C0", "C5"], style="attack", linewidth=3,
             markers=["*", "o", "<", "v", "^", 'H', ">", "d"], markeredgecolor='none', markersize=16, legend=False,
             errorbar=('ci', 95))

ax1.lines[0].set_linestyle("-")
ax1.lines[1].set_linestyle("-")
ax1.lines[2].set_linestyle("-")
ax1.lines[3].set_linestyle("-")
ax1.lines[4].set_linestyle("-")
ax1.lines[5].set_linestyle("--")
ax1.lines[6].set_linestyle("-")
ax1.lines[7].set_linestyle("-")
plt.grid(False)
plt.grid(axis='y', ls='--')

m = [0.25, 0.5, 0.75, 1]
plt.xticks([0.25, 0.5, 0.75, 1], m, fontsize=16)
ax1.set_yscale('symlog', linthresh=50)
ax1.set_ylim(bottom=-5)

max_t = pkl['time'].max()
log_ticks = []
v = 50
while v < max_t:
    log_ticks.append(v)
    v = v * 2 if v >= 100 else 100
log_ticks.append(v)

yticks = [0, 30, 50] + log_ticks
plt.yticks(yticks, [str(v) for v in yticks], fontsize=16)

ax1.set_ylabel('Running Time (s)', fontsize=20)
ax1.set_xlabel('Number of queries ($m$)', fontsize=20)

plt.savefig("./pictures/mS2Lucenetime.pdf", bbox_inches='tight')
