# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='Enron', choices=['Enron', 'Lucene'])
args = parser.parse_args()
dataset = args.dataset

with open(f"../pic_pkl/PR{dataset}.pkl", "rb") as f:
    df = pickle.load(f)

count = 20
p = [10, 15, 20, 25]
m = [0.25, 0.5, 0.75]

colors = []
xvalues = []
yrate = []
yvalues = []

for i_p, v_p in enumerate(p):
    for i_m, v_m in enumerate(m):
        if i_m == 3:
            continue
        candidate_y_vals = list(df.loc[(i_p * 4 + i_m) * count: (i_p * 4 + i_m + 1) * count - 1]['len'])

        yrate.append(np.mean(df.loc[(i_p * 4 + i_m) * count: (i_p * 4 + i_m + 1) * count - 1]['inclusionRate']))
        xvalues.append(i_p * (len(m) + 1) + i_m)
        colors.append('C{:d}'.format(i_m))
        yvalues.append(candidate_y_vals)

fig, ax1 = plt.subplots(figsize=(6, 4))
plt.grid(axis='y', ls='--')

box = ax1.boxplot(yvalues, positions=xvalues, patch_artist=True)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for item in ['whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(box[item], color='k', linewidth=1.3)

all_vals = [v for sublist in yvalues for v in sublist]
max_val = max(all_vals)
step = 10 ** (len(str(int(max_val))) - 1)
if max_val / step < 2:
    step //= 2
yticks = list(range(0, int(max_val) + step, step))
plt.yticks(yticks, yticks, fontsize=16)

legend_elements = box['boxes'][:len(m)] + [Line2D([0], [0], color='C3', linestyle=':', marker='o')]
legend_labels = ['$m$={}$n$'.format(nkw) for nkw in m] + ['existence']
legend1 = Legend(ax1, legend_elements, legend_labels, frameon=True,
                 ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.98),
                 fontsize=16, markerscale=2)
ax1.add_artist(legend1)

ax1.set_ylabel('Prior query number', fontsize=20)
ax2 = ax1.twinx()
ax2.set_ylabel('Existence rate', color='r', fontsize=20)

xtick_positions = []
for i in range(len(p)):
    mid_pos = 0.5 * (xvalues[i * len(m)] + xvalues[(i + 1) * len(m) - 1])
    xtick_positions.append(mid_pos)
    ax2.plot(xvalues[i * len(m):(i + 1) * len(m)], yrate[i * len(m):(i + 1) * len(m)], 'C3o:', markersize=9)

xtick_labels = ['$\\rho$={}'.format(pp) for pp in p]
plt.xticks(xtick_positions, xtick_labels, fontsize=18)
plt.setp(ax1.get_xticklabels(), fontsize=16)

plt.yticks([0, 0.25, 0.5, 0.75, 1], [0.0, 0.25, 0.5, 0.75, 1.0], fontsize=16)
plt.tick_params(axis='y', colors='red')

ax1.set_ylim(-0.01 * yticks[-1], yticks[-1])
ax2.set_ylim(-0.01, 1.0)

plt.savefig(f"./pictures/PRa{dataset}.pdf", bbox_inches='tight', bbox_extra_artists=(legend1,))
