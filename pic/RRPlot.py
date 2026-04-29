# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd
import copy
import numpy as np

with open("../pic_pkl/RR.pkl", "rb") as f:
    df = pickle.load(f)
fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(df, x='e_n', y='recovery', hue='m', palette=["C0", "C1", "C2"], style="m", markers=['o', 'v', 'd'],
             markeredgecolor='none', markersize=12, legend=False)
ax1.lines[0].set_linestyle("-")
ax1.lines[1].set_linestyle("-")
ax1.lines[2].set_linestyle("-")

plt.grid(False)
plt.grid(axis='y', ls='--')

legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
                   Line2D([0], [0], color='C1', linestyle='-', marker='v'),
                   Line2D([0], [0], color='C2', linestyle='-', marker='d')]
legend_labels = ['$m$={}$\cdot n$'.format(nkw) for nkw in [0.25, 0.5, 0.75]]
legend2 = Legend(ax1, legend_elements, legend_labels, fontsize=16, markerscale=2)
ax1.add_artist(legend2)

e_n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yaxis = [0.00, 0.25, 0.50, 0.75, 1.00]

xtick_labels = ['$\eta$={}'.format(pp) for pp in e_n]
plt.yticks(yaxis, yaxis, fontsize=16)
plt.xticks(e_n, e_n, fontsize=16)
ax1.set_ylabel('Prior query accuracy', fontsize=16)
ax1.set_xlabel('$\eta$', fontsize=16)

plt.savefig("./pictures/RR.pdf", bbox_inches='tight')
