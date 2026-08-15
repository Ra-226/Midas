# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse('Enron', 'S1')
dataset, scenario = args.dataset, args.scenarios

with open(f"../pic_pkl/m{scenario}{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

ATTACK_ORDER = ['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"]
marker_map = dict(zip(ATTACK_ORDER, ["v", 'd', "^", 'H', ">", "*", 'o', "<"]))
m_order = sorted(pkl['m'].unique())

def boot_ci(a, n_boot=1000, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(a, size=len(a), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi

group_width = 0.8
offsets = np.linspace(-group_width / 2 + group_width / (2 * len(ATTACK_ORDER)),
                      group_width / 2 - group_width / (2 * len(ATTACK_ORDER)),
                      len(ATTACK_ORDER))
colors = ["C4", "C6", 'c', 'C2', 'C1', 'C3', "C0", "C5"]

fig, ax1 = plt.subplots(figsize=(6, 4))
plt.grid(axis='y', ls='--')

for j, atk in enumerate(ATTACK_ORDER):
    sub = pkl[pkl['attack'] == atk]
    x_pos, y_mean, yerr_lo, yerr_hi = [], [], [], []
    for i, m_val in enumerate(m_order):
        vals = sub[sub['m'] == m_val]['recovery'].values
        lo, hi = boot_ci(vals)
        x_pos.append(i + offsets[j])
        y_mean.append(vals.mean())
        yerr_lo.append(vals.mean() - lo)
        yerr_hi.append(hi - vals.mean())
    ax1.errorbar(x_pos, y_mean, yerr=[yerr_lo, yerr_hi],
                 fmt=marker_map[atk], color=colors[j], ecolor=colors[j],
                 ms=8,  # marker size in points (default 6)
                 elinewidth=1,  # error bar line width in points (default 1)
                 capsize=3,  # length of the caps on the error bars (points)
                 mew=0.5, mec='k',  # marker edge width / edge color (optional)
                 zorder=3)

for xc in np.arange(0.5, len(m_order) - 0.5):
    ax1.axvline(xc, color='black', linestyle='--',
                linewidth=0.8, alpha=0.5, zorder=2)

ax1.set_ylim(-0.05, 1.05)
m = [0.25, 0.5, 0.75, 1]
ax1.set_xticks(range(len(m_order)))
ax1.set_xticklabels([str(v) for v in m], fontsize=16)
ax1.set_xlim(-0.5, len(m_order) - 0.5)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('Number of queries ($m$)', fontsize=20)

plt.savefig(f"./pictures/m{scenario}{dataset}.pdf", bbox_inches='tight')
plt.show()