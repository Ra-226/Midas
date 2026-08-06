import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='Enron', choices=['Enron', 'Lucene'])
parser.add_argument('-s', '--scenarios', default='S3', choices=['S1', 'S2', 'S3'])
args = parser.parse_args()

with open(f"../pic_pkl/osse{args.scenarios}{args.dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

pkl.replace({"fpr": {0.01: 1, 0.02: 2, 0.05: 3}}, inplace=True)

full_order = ['score', 'ikk', 'sap', 'ihop', 'ihopM', 'midas_1', 'midas', 'jigsaw']
palette_full = ['C4', 'C6', 'c', 'C2', 'C1', 'C3', 'C0', 'C5']
markers_full = ['v', 'd', '^', 'H', '>', '*', 'o', '<']

present = pkl['attack'].unique()
hue_order = [a for a in full_order if a in present]
palette = [palette_full[i] for i, a in enumerate(full_order) if a in present]
markers = [markers_full[i] for i, a in enumerate(full_order) if a in present]


fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='fpr', y='recovery', hue='attack',
              hue_order=hue_order,
              style_order=hue_order,
              palette=palette, style="attack", linewidth=3,
              markers=markers, markeredgecolor='none', markersize=16, legend=False,
              errorbar=('ci', 95))

for col in ax1.collections:
    if isinstance(col, PolyCollection):
        fc = mcolors.to_rgba(col.get_facecolor()[0])
        edge = tuple(0.7 * fc[i] for i in range(3)) + (1.0,)
        edge = sns.utils.set_hls_values(fc, l=0.45)
        col.set_edgecolor(edge + (0.35,))
        col.set_linewidth(0.7)
        col.set_facecolor(mcolors.to_rgba(col.get_facecolor()[0], 0.15))
        col.set_alpha(None)

for i, atk in enumerate(hue_order):
    ax1.lines[i].set_linestyle("--" if atk == "midas_1" else "-")

plt.grid(False)
plt.grid(axis='y', ls='--')

m = [0, 1, 2, 3]
plt.xticks(m, ['No def', 0.01, 0.02, 0.05], fontsize=16)
plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00],
           [0.00, 0.25, 0.50, 0.75, 1.00], fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('False Positive Rate (FPR)', fontsize=20)

legend_elements = [Line2D([0], [0], color='C0', linestyle='-', marker='o'),
                   Line2D([0], [0], color='C4', linestyle='-', marker='v'),
                   Line2D([0], [0], color='C6', linestyle='-', marker='d'),
                   Line2D([0], [0], color='c', linestyle='-', marker='^'),
                   Line2D([0], [0], color='C2', linestyle='-', marker='H'),
                   Line2D([0], [0], color='C1', linestyle='-', marker='>'),
                   Line2D([0], [0], color='C5', linestyle='-', marker='<'),
                   Line2D([0], [0], color='C3', linestyle='--', marker='*')]
legend_labels = ['Midas', 'Score', 'IKK', 'SAP', 'IHOP', 'IHOP$^M$', "Jigsaw", "Midas $\\gamma=1$"]
legend2 = Legend(ax1, legend_elements, legend_labels, ncol=2, loc='best',
                 fontsize=12, markerscale=2)
ax1.add_artist(legend2)

plt.savefig(f"./pictures/osse{args.scenarios}{args.dataset}.pdf",
            bbox_inches='tight')

