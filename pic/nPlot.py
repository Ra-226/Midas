# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse('Enron', 'S1')
dataset, scenario = args.dataset, args.scenarios

with open(f"../pic_pkl/n{scenario}{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='n', y='recovery', hue='attack',
             hue_order=['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"],
             palette=["C1", "C2", 'C3', 'C4', 'c', 'C6', "C0", "C5"], style="attack", linewidth=3,
             markers=["*", 'o', "<", 'v', 'd', '^', 'H', ">"], markeredgecolor='none', markersize=16,
             legend=False,
             errorbar=('ci', 95))

for i in [0, 1, 2, 3, 4, 6, 7]:
    ax1.lines[i].set_linestyle("-")
ax1.lines[5].set_linestyle("--")

plt.grid(False)
plt.grid(axis='y', ls='--')

m = [500, 1000, 1500, 2000]
yaxis = [0.00, 0.25, 0.50, 0.75, 1.00]
plt.xticks([500, 1000, 1500, 2000], m, fontsize=16)
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('Number of keywords ($n$)', fontsize=20)

plt.savefig(f"./pictures/n{scenario}{dataset}.pdf", bbox_inches='tight')
plt.show()
