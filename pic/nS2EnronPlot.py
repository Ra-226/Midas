# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:25:41 2024

@author: Ra
"""

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd
import copy
import numpy as np

with open("../pic_pkl/nS2Enron.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))
sns.lineplot(pkl, x='n', y='recovery', hue='attack',
             hue_order=['score', 'ikk', 'sap', 'ihop', 'ihopM', "midas_1", 'midas', "jigsaw"],
             palette=["C1", "C2", 'C3', 'C4', 'c', 'C6', "C0", "C5"], style="attack", linewidth=3,
             markers=["*", 'o', 'v', 'd', '^', 'H', ">", "<"], markeredgecolor='none', markersize=16, legend=False,
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

m = [500, 1000, 1500, 2000]
plt.xticks([500, 1000, 1500, 2000], m, fontsize=16)
plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00], [0.00, 0.25, 0.50, 0.75, 1.00], fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('Number of keywords ($n$)', fontsize=20)

plt.savefig("./pictures/nS2Enron.pdf", bbox_inches='tight')
plt.show()
