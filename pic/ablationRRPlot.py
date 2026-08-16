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

with open(f"../pic_pkl/ablationRR{scenarios}Enron.pkl", "rb") as f:
    pkl = pickle.load(f)

m_order = sorted(pkl['m'].unique())
gamma_order = sorted(pkl['gamma'].unique())
colors = ["C0", "C1", "C2", "C3", "C4"]

bar_width = 0.11
step = 0.16
offsets = (np.arange(len(gamma_order)) - (len(gamma_order) - 1) / 2) * step
alpha_fill = 0.3


def boot_ci(a, n_boot=1000, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(a, size=len(a), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


fig, ax1 = plt.subplots(figsize=(8, 5))

for i, m_val in enumerate(m_order):
    for g, gamma_val in enumerate(gamma_order):
        vals = pkl[(pkl['m'] == m_val) & (pkl['gamma'] == gamma_val)]['recovery'].values
        mean = vals.mean()
        lo, hi = boot_ci(vals)
        color = colors[g]
        x = i + offsets[g]
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
plt.xticks(ticks=range(len(m_order)), labels=[str(v) for v in m_order], fontsize=16)
yaxis = [.0, .25, .5, .75, 1.0]
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_xlim(-0.5, len(m_order) - 0.5)
ax1.set_ylim(-0.02, 1.08)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('Number of queries ($m$)', fontsize=20)

labels = ['Midas with RR ($\\gamma$ = 4)', 'Midas without RR ($\\gamma$ = 1)',
          'Midas without RR ($\\gamma$ = 2)', 'Midas without RR ($\\gamma$ = 3)',
          'Midas without RR ($\\gamma$ = 4)']
handles = [mpatches.Patch(facecolor=mcolors.to_rgba(c, alpha_fill), edgecolor=c, linewidth=1.5)
           for c in colors]
plt.legend(handles=handles, labels=labels, fontsize=10)

plt.savefig(f"./pictures/ablationRR{scenarios}Enron.pdf", bbox_inches='tight')
plt.show()