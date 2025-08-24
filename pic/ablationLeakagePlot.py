import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd
import copy
import numpy as np

scenarios = "S2"

with open(f"../pic_pkl/ablationLeakage{scenarios}Enron.pkl", "rb") as f:
    pkl = pickle.load(f)
hue_order = pkl['use_non_co_occurrence'].unique()
custom_colors = ["C0", "#1fb4a7"]
m = [0.25, 0.5, 0.75]
fig, ax1 = plt.subplots(figsize=(6, 4))

ax = sns.barplot(
    pkl, x='m', y='recovery', hue='use_non_co_occurrence',
    hue_order=hue_order,
    palette=dict(zip(hue_order, custom_colors)),
)

handles, _ = ax.get_legend_handles_labels()
plt.legend(
    handles=handles,
    labels=['PR with non-co-occurrence leakage', 'PR without non-co-occurrence leakage'],
    fontsize=12,
)

plt.grid(False)
plt.grid(axis='y', ls='--')

plt.xticks(ticks=[0, 1, 2], labels=["$0.25$", "$0.5$", "$0.75$"], fontsize=16)
yaxis = [0.00, .1, 0.2, 0.4, 0.6]
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_ylabel('Prior query accuracy', fontsize=20)
ax1.set_xlabel('Number of queries ($m$)', fontsize=20)

plt.savefig(f"./pictures/ablationLeakage{scenarios}Enron.pdf", bbox_inches='tight')
plt.show()
