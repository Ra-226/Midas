import pickle
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerTuple
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse('Enron', 'S2')
scenarios = args.scenarios
dataset = args.dataset
m = 500
file = f"n{scenarios}EnronLimitedTime"
with open(f"../pic_pkl/{file}.pkl", "rb") as f:
    pkl = pickle.load(f)

df = pkl[pkl['attack'] != 'score']

order = sorted(df['n'].unique())
attacks = ['midas', 'jigsaw', 'ihop', 'ihopM']
attackexample = ['Midas', 'Jigsaw', 'IHOP', 'IHOP$^M$']

common_colors_acc = ['C0', 'C5', 'C2', 'C1']
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
    int_item = int(item)
    int_order.append(int_item)

n_to_x = {n_val: i for i, n_val in enumerate(int_order)}

fig, ax1 = plt.subplots(figsize=(6, 8))

sns.boxplot(
    data=df, x='n', y='recovery',
    order=order, hue='attack', hue_order=attacks,
    palette=palette, ax=ax1
)
ax1.set_ylabel('Accuracy', fontsize=22)

for i in range(len(order) - 1):
    for ax in (ax1,):
        ax.axvline(i + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=0)

ax2 = ax1.twinx()
mean_times = df.groupby(['n', 'attack'], as_index=False)['time'].mean()
for atk in attacks:
    sub = mean_times[mean_times['attack'] == atk].sort_values('n')
    x_pos = [n_to_x[n] + offset_map[atk] for n in sub['n']]
    ax2.scatter(x_pos, sub['time'], s=80,
                color='red', marker=marker_map[atk],
                zorder=3)
ax2.set_ylabel('Running Time (s)', fontsize=22, color='red')

handles_box, labels_box = ax1.get_legend_handles_labels()
red_handles = [Line2D([0], [0], marker=marker_map[atk], color='red',
                      linestyle='None', markersize=7, label=atk) for atk in attacks]
combined_handles = list(zip(handles_box, red_handles))
ax1.legend(combined_handles, ['Midas', 'Jigsaw', 'IHOP', 'IHOP$^M$'],
           loc='upper center', bbox_to_anchor=(0.523, 0.88), bbox_transform=fig.transFigure,
           ncol=2, frameon=True, framealpha=1, fontsize=16,
           columnspacing=3, handlelength=5.0,
           handler_map={tuple: HandlerTuple(ndivide=2)})

ax1.set_xticks(range(len(order)))
ax1.set_xticklabels([str(v) for v in int_order], fontsize=20)

yaxis = [0.0, 0.25, 0.5, 0.75, 1.0]
ax1.set_yticks(yaxis)
ax1.set_yticklabels([str(v) for v in yaxis], fontsize=20)
ax1.tick_params(axis='y')

ax2.tick_params(axis='y', labelsize=20, colors='red')

ax1.set_xlabel('Number of keywords(n)', fontsize=22)

ax2.grid(False)
ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
plt.subplots_adjust(left=0.1, right=0.95, top=0.75)
ax1.margins(x=0)

ax1.set_xlim(-0.5, len(order) - 0.5)
ax2.margins(x=0)

plt.savefig(f"./pictures/{file}.pdf", bbox_inches='tight')
plt.show()
