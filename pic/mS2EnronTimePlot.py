# -*- coding: utf-8 -*-
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd
import copy
import numpy as np

with open("../pic_pkl/mS2Enron.0.pkl", "rb") as f:
    pkl = pickle.load(f)

pkl1 = pkl[pkl['attack'] != 'ikk']
pkl2 = pkl[pkl['attack'] == 'ikk']
pkl2 = pkl2.copy()
max_value = max(pkl2['time'])
min_value = min(pkl2['time'])
for i in pkl2.index:
    pkl2.loc[i, 'time'] = (pkl2.loc[i]['time'] - min_value) * (140 - min_value) / (max_value - min_value) + min_value

fig, ax1 = plt.subplots(figsize=(6, 4))

pkl3 = pd.concat([pkl1, pkl2])
sns.lineplot(pkl3, x='m', y='time', hue='attack',
             hue_order=['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"],
             palette=["C1", "C2", 'C3', 'C4', 'c', 'C6', "C0", "C5"], style="attack", linewidth=3,
             markers=["*", "o", "<", "v", "^", 'H', ">", "d"], markeredgecolor='none', markersize=16, legend=False,
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

m = [0.25, 0.5, 0.75, 1]
plt.xticks([0.25, 0.5, 0.75, 1], m, fontsize=16)
plt.yticks([0, 30, 50, 80, 110, 140], [0, 30, 50, 80, 500, 900], fontsize=16)

ax1.set_ylabel('Running Time (s)', fontsize=20)
ax1.set_xlabel('Number of queries ($m$)', fontsize=20)

plt.savefig("./pictures/mS2EnronTime.pdf", bbox_inches='tight')
