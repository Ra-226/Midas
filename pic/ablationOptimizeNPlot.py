# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

scenarios = "S1"
dataset = "Enron"

with open(f"../pic_pkl/ablationOptimize{scenarios}N{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

x_order = sorted(pkl['n'].unique())
hue_order = ['midas', 'midasSlow']
colors = ['C0', 'C1']
labels = ['Midas with incremental computation', 'Midas without incremental computation']
xtick_labels = ['500', '1000', '1500', '2000']

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

for i, n_val in enumerate(x_order):
    for h, atk in enumerate(hue_order):
        vals = pkl[(pkl['n'] == n_val) & (pkl['attack'] == atk)]['time'].values
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
yy = [0, 6, 12, 18, 24, 30]
plt.yticks(yy, yy, fontsize=16)
ax1.set_xlim(-0.5, len(x_order) - 0.5)
ax1.set_ylim(0, 32)
ax1.set_ylabel('Running Time (s)', fontsize=20)
ax1.set_xlabel('Number of keywords ($n$)', fontsize=20)

handles = [mpatches.Patch(facecolor=mcolors.to_rgba(c, alpha_fill), edgecolor=c, linewidth=1.5)
           for c in colors]
ax1.legend(handles, labels, loc='upper left', ncol=1, fontsize=12, markerscale=1.5)

plt.savefig(f"./pictures/ablationOptimize{scenarios}N{dataset}.pdf", bbox_inches='tight')
plt.show()