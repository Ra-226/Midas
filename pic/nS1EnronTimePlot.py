# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd
import copy
import numpy as np

with open("../pic_pkl/nS1Enron.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))
sns.lineplot(pkl, x='n', y='time', hue='attack',
             hue_order=['score', 'ikk', 'sap', 'ihop', 'ihopM', 'midas_1', 'midas', "jigsaw"],
             palette=["C1", "C2", 'C3', 'C4', 'c', 'C6', 'C0', "C5"], style="attack", linewidth=3,
             markers=["*", "o", "<", "v", "d", "^", "H", ">"], markeredgecolor='none', markersize=16, legend=False,
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

legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
                   Line2D([0], [0], color='C1', linestyle='-', marker='v'),
                   Line2D([0], [0], color='C2', linestyle='-', marker='d'),
                   Line2D([0], [0], color='C3', linestyle='-', marker='^'),
                   Line2D([0], [0], color='C4', linestyle='-', marker='H'),
                   Line2D([0], [0], color='c', linestyle='-', marker='>'),
                   Line2D([0], [0], color='C5', linestyle='-', marker='<'),
                   Line2D([0], [0], color='C6', linestyle='--', marker='*'),
                   ]
legend_labels = ['{}'.format(nkw) for nkw in
                 ['Midas', 'Score', 'IKK', 'SAP', 'IHOP', 'IHOP$^M$', "Jigsaw", "Midas $\gamma=1$"]]
legend2 = Legend(ax1, legend_elements, legend_labels, ncol=2, loc='upper left',
                 fontsize=12, markerscale=2)
ax1.add_artist(legend2)

m = [500, 1000, 1500, 2000]
plt.xticks([500, 1000, 1500, 2000], m, fontsize=16)
yy = [0, 50, 100, 150, 200, 250]
plt.yticks(yy, yy, fontsize=16)
ax1.set_ylabel('Running Time (s)', fontsize=20)
ax1.set_xlabel('Number of keywords ($n$)', fontsize=20)

plt.savefig("./pictures/nS1EnronTime.pdf", bbox_inches='tight')
plt.show()
