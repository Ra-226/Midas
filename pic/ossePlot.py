import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Enron', choices=['Enron', 'Lucene'])
parser.add_argument('--scenarios', default='S3', choices=['S1', 'S2', 'S3'])
args = parser.parse_args()

attack_config = {
    'midas':  {'color': 'C0', 'marker': 'H', 'label': 'Midas'},
    'midas_1':{'color': 'C1', 'marker': 'X', 'label': 'Midas-1'},
    'jigsaw': {'color': 'C5', 'marker': '>', 'label': 'Jigsaw'},
    'score':  {'color': 'C2', 'marker': 's', 'label': 'Score'},
    'ikk':    {'color': 'C3', 'marker': 'D', 'label': 'IKK'},
    'sap':    {'color': 'C6', 'marker': 'o', 'label': 'SAP'},
    'ihop':   {'color': 'C4', 'marker': 'v', 'label': 'IHOP'},
    'ihopM':  {'color': 'C7', 'marker': '^', 'label': 'IHOP^M'},
}

with open(f"../pic_pkl/osse{args.scenarios}{args.dataset}.pkl", "rb") as f:
    pkl = pickle.load(f)

pkl.replace({"fpr": {0.01: 1, 0.02: 2, 0.05: 3}}, inplace=True)

present = pkl['attack'].unique()
hue_order = [a for a in attack_config if a in present]
palette = [attack_config[a]['color'] for a in hue_order]
markers = [attack_config[a]['marker'] for a in hue_order]
custom_labels = [attack_config[a]['label'] for a in hue_order]

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(pkl, x='fpr', y='recovery', hue='attack',
             hue_order=hue_order,
             palette=palette,
             style="attack", linewidth=3,
             markers=markers, markeredgecolor='none',
             markersize=16, legend=False,
             errorbar=('ci', 95))

plt.grid(False)
plt.grid(axis='y', ls='--')

m = [0, 1, 2, 3]
plt.xticks(m, ['No def', 0.01, 0.02, 0.05], fontsize=16)
plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00],
           [0.00, 0.25, 0.50, 0.75, 1.00], fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=20)
ax1.set_xlabel('False Positive Rate (FPR)', fontsize=20)

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, custom_labels, loc='lower left',
           ncol=1, fontsize=14, markerscale=1.5)

plt.savefig(f"./pictures/osse{args.scenarios}{args.dataset}.pdf",
            bbox_inches='tight')
plt.show()
