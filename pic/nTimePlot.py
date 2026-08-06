# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse('Enron', 'S1')
dataset, scenario = args.dataset, args.scenarios

with open(f"../pic_pkl/n{scenario}{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

ATTACK_ORDER = ['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"]

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='n', y='time', hue='attack',
             hue_order=ATTACK_ORDER,
             style_order=ATTACK_ORDER,
             palette=["C4", "C6", 'c', 'C2', 'C1', 'C3', "C0", "C5"], style="attack", linewidth=3,
             markers=["v", "d", "^", "H", ">", "*", "o", "<"], markeredgecolor='none', markersize=16,
             legend=False,
             errorbar=('ci', 95))

for col in ax1.collections:
    if isinstance(col, PolyCollection):
        fc = mcolors.to_rgba(col.get_facecolor()[0])
        edge = tuple(0.7 * fc[i] for i in range(3)) + (1.0,)
        edge = sns.utils.set_hls_values(fc, l=0.45)
        col.set_edgecolor(edge + (0.35,))
        col.set_linewidth(0.7)
        col.set_facecolor(mcolors.to_rgba(col.get_facecolor()[0], 0.15))
        col.set_alpha(None)

for i, atk in enumerate(ATTACK_ORDER):
    ax1.lines[i].set_linestyle("--" if atk == "midas_1" else "-")

plt.grid(False)
plt.grid(axis='y', ls='--')

m = [500, 1000, 1500, 2000]
plt.xticks([500, 1000, 1500, 2000], m, fontsize=16)
yy = [0, 50, 100, 150, 200, 250]
plt.yticks(yy, yy, fontsize=16)
ax1.set_ylabel('Running Time (s)', fontsize=20)
ax1.set_xlabel('Number of keywords ($n$)', fontsize=20)

# legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
#                    Line2D([0], [0], color='C4', linestyle='-', marker='v'),
#                    Line2D([0], [0], color='C6', linestyle='-', marker='d'),
#                    Line2D([0], [0], color='c', linestyle='-', marker='^'),
#                    Line2D([0], [0], color='C2', linestyle='-', marker='H'),
#                    Line2D([0], [0], color='C1', linestyle='-', marker='>'),
#                    Line2D([0], [0], color='C5', linestyle='-', marker='<'),
#                    Line2D([0], [0], color='C3', linestyle='--', marker='*')]
# legend_labels = ['Midas', 'Score', 'IKK', 'SAP', 'IHOP', 'IHOP$^M$', "Jigsaw", "Midas $\\gamma=1$"]
# legend2 = Legend(ax1, legend_elements, legend_labels, ncol=2, loc='upper left',
#                  fontsize=12, markerscale=2)
# ax1.add_artist(legend2)

plt.savefig(f"./pictures/n{scenario}{dataset}Time.pdf", bbox_inches='tight')
plt.show()
