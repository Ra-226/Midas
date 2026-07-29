import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
args = utils.parameter_parse(default_scenarios='S2')
scenarios = args.scenarios
m = 500
file = f"m{scenarios}EnronLimitedTime"
with open(f"../pic_pkl/{file}.pkl", "rb") as f:
    pkl = pickle.load(f)

df = pkl[pkl['attack'] != 'score']

order = sorted(df['m'].unique())
attacks = ['midas', 'jigsaw', 'ihop', 'ihopM']
attackexample = ['Midas', 'Jigsaw', 'IHOP', 'IHOP$^M$']

common_colors_acc = ['C0', 'C5', 'C4', 'c']
common_colors_time = sns.color_palette("Set2")
palette = dict(zip(attacks, common_colors_acc))
palette2 = dict(zip(attacks, common_colors_time))
markers = ['o', '<', 'H', '>']
marker_map = dict(zip(attacks, markers))

group_width = 0.8
offsets = np.linspace(-group_width / 2 + group_width / (2 * len(attacks)),
                      group_width / 2 - group_width / (2 * len(attacks)),
                      len(attacks))
offset_map = dict(zip(attacks, offsets))

int_order = []
for item in order:
    int_order.append(item)

n_to_x = {n_val: i for i, n_val in enumerate(int_order)}

fig, ax1 = plt.subplots(figsize=(10, 5))

sns.boxplot(
    data=df, x='m', y='recovery',
    order=order, hue='attack', hue_order=attacks,
    palette=palette, ax=ax1
)
ax1.set_ylabel('Accuracy', fontsize=22)

for i in range(len(order) - 1):
    for ax in (ax1,):
        ax.axvline(i + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=0)

ax2 = ax1.twinx()
for atk in attacks:
    sub = df[df['attack'] == atk].copy()
    x_base = sub['m'].map(n_to_x).astype(float)
    x_pos = x_base + offset_map[atk]
    ax2.scatter(
        x_pos, sub['time'],
        s=144, marker=marker_map[atk],
        edgecolors='black',
        alpha=0.85,
        c=[palette2[atk]] * len(sub),
        zorder=3
    )
ax2.set_ylabel('Running Time (s)', fontsize=22, color='red')

handles_box, labels_box = ax1.get_legend_handles_labels()
ax1.legend(handles_box, ['Midas', 'Jigsaw', 'IHOP', 'IHOP$^M$'],
           loc='upper center', bbox_to_anchor=(0.5, 1.18),
           ncol=len(attacks), frameon=True, framealpha=1, fontsize=20,
           columnspacing=2)

scatter_handles = [Line2D([0], [0], marker=marker_map[atk], color='w',
                          markerfacecolor=palette2[atk], markersize=12,
                          markeredgecolor='black', label=atk) for atk in attacks]
ax2.legend(scatter_handles, ['Midas', 'Jigsaw', 'IHOP', 'IHOP$^M$'], title='Time', title_fontsize=18,
           loc='lower right', ncol=2,
           frameon=True,
           framealpha=0.5,
           edgecolor='black',
           fontsize=18,
           columnspacing=.8,
           handletextpad=0.3,
           handlelength=1.5,
           borderaxespad=0.5,
           labelspacing=0.8
           )

ax1.set_xticks(range(len(order)))
ax1.set_xticklabels([str(v) for v in int_order], fontsize=20)

yaxis = [0.0, 0.25, 0.5, 0.75, 1.0]
ax1.set_yticks(yaxis)
ax1.set_yticklabels([str(v) for v in yaxis], fontsize=20)
ax1.tick_params(axis='y')

ax2.tick_params(axis='y', labelsize=20, colors='red')

ax1.set_xlabel('Number of queries (m)', fontsize=22)

ax2.grid(False)
ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
plt.subplots_adjust(left=0.1, right=0.95)
ax1.margins(x=0)

ax1.set_xlim(-0.5, len(order) - 0.5)
ax2.margins(x=0)

plt.savefig(f"./pictures/limited_time_{scenarios}_Enron_n_500_m_[0.25, 0.5, 0.75, 1].pdf", bbox_inches='tight')
plt.show()
