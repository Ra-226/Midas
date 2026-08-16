# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse(default_scenarios='S1')
scenarios = args.scenarios

with open(f"../pic_pkl/ablationLeakage{scenarios}Enron.pkl", "rb") as f:
    pkl = pickle.load(f)

x_order = sorted(pkl['m'].unique())
hue_order = sorted(pkl['use_co_absence'].unique())
colors = ["C0", "C1"]
labels = ['PR with co-absence pattern', 'PR without co-absence pattern']
xtick_labels = ["0.25", "0.5", "0.75"]

bar_width = 0.3
step = 0.34
offsets = (np.arange(len(hue_order)) - (len(hue_order) - 1) / 2) * step
alpha_fill = 0.25


def boot_ci(a, n_boot=1000, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(a, size=len(a), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


fig, ax1 = plt.subplots(figsize=(6, 4))

for i, m_val in enumerate(x_order):
    for h, uca in enumerate(hue_order):
        vals = pkl[(pkl['m'] == m_val) & (pkl['use_co_absence'] == uca)]['recovery'].values
        mean = vals.mean()
        lo, hi = boot_ci(vals)
        color = colors[h]
        x = i + offsets[h]
        half = bar_width / 2
        ax1.bar(x, mean, width=bar_width,
                facecolor=mcolors.to_rgba(color, alpha_fill),
                edgecolor='none', zorder=2)
        ax1.plot([x-half, x-half, x+half, x+half], [0, mean, mean, 0],
                 color=color, linewidth=1.5, zorder=3)
        ax1.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]],
                     fmt='none', ecolor=color, elinewidth=1.5, capsize=3, zorder=3)

plt.grid(False)
plt.grid(axis='y', ls='--')
plt.xticks(range(len(x_order)), xtick_labels, fontsize=16)
yaxis = [0.00, .1, 0.2, 0.4, 0.6]
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_xlim(-0.5, len(x_order) - 0.5)
ax1.set_ylim(0, 0.65)
ax1.set_ylabel('Prior query accuracy', fontsize=20)
ax1.set_xlabel('Number of queries ($m$)', fontsize=20)

handles = [mpatches.Patch(facecolor=mcolors.to_rgba(c, alpha_fill), edgecolor=c, linewidth=1.5)
           for c in colors]
plt.legend(handles=handles, labels=labels, fontsize=12, loc='upper left')

plt.savefig(f"./pictures/ablationLeakage{scenarios}Enron.pdf", bbox_inches='tight')
plt.show()