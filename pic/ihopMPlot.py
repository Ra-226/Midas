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
parser.add_argument('-d', '--dataset', default='Enron', choices=['Enron', 'Lucene'])
args = parser.parse_args()
dataset = args.dataset

with open(f"../pic_pkl/ihopM{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))

ax = sns.lineplot(pkl, x='x', y='recovery', hue='theta',
                  palette=['C0', "C1", "C2", 'C3', 'C4'], style="theta",
                  markers=['o', 'v', 'd', '^', 'H'], markeredgecolor='none', linewidth=2, markersize=12, legend=False,
                  errorbar=('ci', 95))

ax1.lines[0].set_linestyle("-")
ax1.lines[1].set_linestyle("-")
ax1.lines[2].set_linestyle("-")
ax1.lines[3].set_linestyle("-")
ax1.lines[4].set_linestyle("-")

plt.grid(False)
plt.grid(axis='y', ls='--')

legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
                   Line2D([0], [0], color='C1', linestyle='-', marker='v'),
                   Line2D([0], [0], color='C2', linestyle='-', marker='d'),
                   Line2D([0], [0], color='C3', linestyle='-', marker='^'),
                   Line2D([0], [0], color='C4', linestyle='-', marker='H')]
legend_labels = ['IHOP'] + ['IHOP$^M$ $\\theta$={}'.format(nkw) for nkw in [1, 5, 10, 25]]
legend2 = Legend(ax1, legend_elements, legend_labels, loc='lower right', fontsize=16, markerscale=1.5)
ax1.add_artist(legend2)

niter = [10, 20, 50, 100, 200, 500, 1000, 2000]
yaxis = [0.00, 0.25, 0.50, 0.75, 1.00]
ax.set_xticks(range(8), niter, fontsize=16)
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=16)
ax1.set_xlabel('$n_{iters}$', fontsize=16)

plt.savefig(f"./pictures/ihopM{dataset}.pdf", bbox_inches='tight')
plt.show()
