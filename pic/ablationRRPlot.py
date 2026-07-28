import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

args = utils.parameter_parse(default_scenarios='S2')
scenarios = args.scenarios
word_len = 500

with open(f"../pic_pkl/ablationRR{scenarios}Enron.pkl", "rb") as f:
    pkl = pickle.load(f)

gamma_count = pkl['gamma'].nunique()

colors = ["C0", "#b8f3ee",
          "#75e7dd",
          "#32dccd",
          "#1fb4a7"]

fig, ax1 = plt.subplots(figsize=(8, 5))

sns.barplot(data=pkl, x='m', y='recovery', hue='gamma', ax=ax1, palette=colors)
ax1.legend_.remove()

plt.grid(False)
plt.grid(axis='y', ls='--')
plt.xticks(ticks=[0, 1, 2], labels=["0.25", "0.5", "0.75"], fontsize=16)
yaxis = [.0, .25, .5, .75, 1.0]
plt.yticks(yaxis, yaxis, fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=22)
ax1.set_xlabel('Number of queries ($m$)', fontsize=22)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(
    handles=handles,
    labels=['Midas with RR ($\gamma$ = 4)', 'Midas without RR ($\gamma$ = 1)', 'Midas without RR ($\gamma$ = 2)',
            'Midas without RR ($\gamma$ = 3)', 'Midas without RR ($\gamma$ = 4)'],
    fontsize=10,
)

plt.savefig(f"./pictures/ablationRR{scenarios}Enron.pdf", bbox_inches='tight')
plt.show()
