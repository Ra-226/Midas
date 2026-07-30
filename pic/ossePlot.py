import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='Enron', choices=['Enron', 'Lucene'])
parser.add_argument('-s', '--scenarios', default='S3', choices=['S1', 'S2', 'S3'])
args = parser.parse_args()

with open(f"../pic_pkl/osse{args.scenarios}{args.dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

pkl.replace({"fpr": {0.01: 1, 0.02: 2, 0.05: 3}}, inplace=True)

full_order = ['score', 'ikk', 'sap', 'ihop', 'ihopM', 'midas_1', 'midas', 'jigsaw']
palette_full = ['C1', 'C2', 'C3', 'C4', 'c', 'C6', 'C0', 'C5']
markers_full = ['o', '*', '<', 'v', 'd', '^', 'H', '>']

present = pkl['attack'].unique()
hue_order = [a for a in full_order if a in present]
palette = [palette_full[i] for i, a in enumerate(full_order) if a in present]
markers = [markers_full[i] for i, a in enumerate(full_order) if a in present]


fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='fpr', y='recovery', hue='attack',
             hue_order=hue_order,
             palette=palette, style="attack", linewidth=3,
             markers=markers, markeredgecolor='none', markersize=16, legend=False,
             errorbar=('ci', 95))

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

plt.savefig(f"./pictures/osse{args.scenarios}{args.dataset}.pdf",
            bbox_inches='tight')
plt.show()
