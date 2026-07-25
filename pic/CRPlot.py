# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd
import copy
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Enron', choices=['Enron', 'Lucene'])
args = parser.parse_args()
dataset = args.dataset

with open(f"../pic_pkl/CRTest_Dsim_5000_count_30_{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='mu', y='recovery', hue='gamma', palette=["C0", "C1", "C2", 'C3'], style="gamma",
             markers=['o', 'v', 'd', '^'], markeredgecolor='none', markersize=12, legend=False, errorbar=('ci', 95))

ax1.lines[0].set_linestyle("-")
ax1.lines[1].set_linestyle("-")
ax1.lines[2].set_linestyle("-")
ax1.lines[3].set_linestyle("-")

plt.grid(False)
plt.grid(axis='y', ls='--')

legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
                   Line2D([0], [0], color='C1', linestyle='-', marker='v'),
                   Line2D([0], [0], color='C2', linestyle='-', marker='d'),
                   Line2D([0], [0], color='C3', linestyle='-', marker='^')]
legend_labels = ['$\gamma$={}'.format(nkw) for nkw in [1, 2, 3, 4]]
legend2 = Legend(ax1, legend_elements, legend_labels, loc='lower right', fontsize=16, markerscale=2)
ax1.add_artist(legend2)

u = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
yaxis = [0, 0.25, 0.5, 0.75, 1]

xtick_labels = ['$\mu$={}'.format(pp) for pp in u]
plt.xticks(u, u, fontsize=16)
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=16)
ax1.set_xlabel('$\mu$', fontsize=16)

plt.savefig(f"./pictures/CR{dataset}.pdf", bbox_inches='tight')

plt.show()
