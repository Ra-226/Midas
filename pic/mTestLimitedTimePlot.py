import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pandas as pd
import copy
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse(default_scenarios='S1')
scenarios = args.scenarios
dataset = 'Enron'

with open(f"../pic_pkl/m{scenarios}{dataset}LimitedTime.pkl", "rb") as f:
    pkl = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(6, 4))
# plt.subplots_adjust()
sns.lineplot(pkl, x='m', y='recovery', hue='attack',
             hue_order=['ihop', 'ihopM', 'midas', "jigsaw"],
             palette=['C4', 'c', "C0", "C5"], style="attack", linewidth=3,
             markers=['o', "<", 'H', ">"], markeredgecolor='none', markersize=16, legend=False,
             errorbar=('ci', 95))

ax1.lines[0].set_linestyle("-")
ax1.lines[1].set_linestyle("-")
ax1.lines[2].set_linestyle("-")
ax1.lines[3].set_linestyle("-")

plt.grid(False)
plt.grid(axis='y', ls='--')

m = [0.25, 0.5, 0.75, 1]
yaxis = [0.00, 0.25, 0.50, 0.75, 1.00]
plt.xticks([0.25, 0.5, 0.75, 1], m, fontsize=16)
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('Number of queries ($m$)', fontsize=20)

plt.savefig(f"./pictures/m{scenarios}{dataset}LimitedTime.pdf", bbox_inches='tight')
plt.show()
