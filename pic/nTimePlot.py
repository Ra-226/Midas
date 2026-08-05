# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse('Enron', 'S1')
dataset, scenario = args.dataset, args.scenarios

with open(f"../pic_pkl/n{scenario}{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='n', y='time', hue='attack',
             hue_order=['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"],
             palette=["C1", "C2", 'C3', 'C4', 'c', 'C6', "C0", "C5"], style="attack", linewidth=3,
             markers=["*", "o", "<", "v", "d", "^", "H", ">"], markeredgecolor='none', markersize=16,
             legend=False,
             errorbar=('ci', 95))

for i in [0, 1, 2, 3, 4, 6, 7]:
    ax1.lines[i].set_linestyle("-")
ax1.lines[5].set_linestyle("--")

plt.grid(False)
plt.grid(axis='y', ls='--')

m = [500, 1000, 1500, 2000]
plt.xticks([500, 1000, 1500, 2000], m, fontsize=16)
yy = [0, 50, 100, 150, 200, 250]
plt.yticks(yy, yy, fontsize=16)
ax1.set_ylabel('Running Time (s)', fontsize=20)
ax1.set_xlabel('Number of keywords ($n$)', fontsize=20)

# legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
#                    Line2D([0], [0], color='C1', linestyle='-', marker='v'),
#                    Line2D([0], [0], color='C2', linestyle='-', marker='d'),
#                    Line2D([0], [0], color='C3', linestyle='-', marker='^'),
#                    Line2D([0], [0], color='C4', linestyle='-', marker='H'),
#                    Line2D([0], [0], color='c', linestyle='-', marker='>'),
#                    Line2D([0], [0], color='C5', linestyle='-', marker='<'),
#                    Line2D([0], [0], color='C6', linestyle='--', marker='*')]
# legend_labels = ['Midas', 'Score', 'IKK', 'SAP', 'IHOP', 'IHOP$^M$', "Jigsaw", "Midas $\\gamma=1$"]
# legend2 = Legend(ax1, legend_elements, legend_labels, ncol=2, loc='upper left',
#                  fontsize=12, markerscale=2)
# ax1.add_artist(legend2)

plt.savefig(f"./pictures/n{scenario}{dataset}Time.pdf", bbox_inches='tight')
plt.show()
