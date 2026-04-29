# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

with open("../pic_pkl/AuxS1Enron.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='m', y='recovery', hue='attack',
             hue_order=['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"],
             palette=["C1", "C2", 'C3', 'C4', 'c', 'C6', "C0", "C5"], style="attack", linewidth=3,
             markers=["o", '*', "<", 'v', 'd', '^', 'H', ">"], markeredgecolor='none', markersize=16, legend=False,
             errorbar=('ci', 95))

ax1.lines[0].set_linestyle("-")
ax1.lines[1].set_linestyle("-")
ax1.lines[2].set_linestyle("-")
ax1.lines[3].set_linestyle("-")
ax1.lines[4].set_linestyle("-")
ax1.lines[5].set_linestyle("--")
ax1.lines[6].set_linestyle("-")
ax1.lines[7].set_linestyle("-")

plt.grid(False)
plt.grid(axis='y', ls='--')

legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
                   Line2D([0], [0], color='C1', linestyle='-', marker='v'),
                   Line2D([0], [0], color='C2', linestyle='-', marker='d'),
                   Line2D([0], [0], color='C3', linestyle='-', marker='^'),
                   Line2D([0], [0], color='C4', linestyle='-', marker='H'),
                   Line2D([0], [0], color='c', linestyle='-', marker='>'),
                   Line2D([0], [0], color='C5', linestyle='-', marker='<'),
                   Line2D([0], [0], color='C6', linestyle='--', marker='*'),
                   ]
legend_labels = ['{}'.format(nkw) for nkw in
                 ['Midas', 'Score', 'IKK', 'SAP', 'IHOP', 'IHOP$^M$', "Jigsaw", "Midas $\gamma=1$"]]
legend2 = Legend(ax1, legend_elements, legend_labels, ncol=2, loc='best',
                 bbox_to_anchor=(0.4, 0.7), fontsize=12, markerscale=2)

m = ['2k', '3k', '5k', '7k', '10k', '12k', '15k']
plt.xticks([2000, 3000, 5000, 7000, 10000, 12000, 15000], m, fontsize=16)
plt.yticks([0.0, 0.25, 0.50, 0.75, 1.00], [0.0, 0.25, 0.50, 0.75, 1.00], fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('Number of files ($|\mathsf{D}_{sim}|$)', fontsize=20)

average_time_by_n = pkl.groupby(['attack'])['time'].mean().reset_index()
print(average_time_by_n)

plt.savefig("./pictures/AuxS1Enron.pdf", bbox_inches='tight')
plt.show()
