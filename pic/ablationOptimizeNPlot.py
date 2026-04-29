# -*- coding: utf-8 -*-

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

scenarios = "S1"
dataset = "Enron"

with open(f"../pic_pkl/ablationOptimize{scenarios}N{dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.barplot(pkl, x='n', y='time', hue='attack',
            hue_order=['midas', "midasSlow"],
            palette=['C0', "#1fb4a7"],
            errorbar=None)

plt.grid(False)
plt.grid(axis='y', ls='--')

handles, labels = ax1.get_legend_handles_labels()
custom_labels = ['Midas with incremental computation', 'Midas without incremental computation']
ax1.legend(handles, custom_labels,
           loc='upper left',
           ncol=1,
           fontsize=12,
           markerscale=1.5)

m = [500, 1000, 1500, 2000]
plt.xticks([0, 1, 2, 3], m, fontsize=16)
yy = [0, 6, 12, 18, 24, 30]
plt.yticks(yy, yy, fontsize=16)
ax1.set_ylabel('Running Time (s)', fontsize=20)
ax1.set_xlabel('Number of keywords ($n$)', fontsize=20)

plt.savefig(f"./pictures/ablationOptimize{scenarios}N{dataset}.pdf", bbox_inches='tight')
plt.show()
