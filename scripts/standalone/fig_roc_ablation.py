import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
P = pd.read_csv(os.path.expanduser("~/model_results/roc_preds_v4.csv"))
A = pd.read_csv(os.path.expanduser("~/model_results_ablation_v4.csv"))
lab = {"all": "all features (deployed)", "shear": "shear only", "force": "force only", "time": "time since loosening"}
sty = {"all": ("k", "-"), "shear": ("tab:blue", "--"), "force": ("tab:orange", "--"), "time": ("0.5", ":")}
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5))
for j, split in enumerate(["LOBO", "LOSO"]):
    for f in ["all", "shear", "force", "time"]:
        q = P[(P.features == f) & (P.split == split)]
        fpr, tpr, _ = roc_curve(q.y, q.p)
        ax[j].plot(fpr, tpr, color=sty[f][0], ls=sty[f][1], lw=1.2, label=f"{lab[f]} ({roc_auc_score(q.y, q.p):.3f})")
    ax[j].plot([0, 1], [0, 1], color="0.8", lw=0.8)
    ax[j].set_title(f"({'ab'[j]}) ROC, leave-one-{'berry' if split == 'LOBO' else 'size'}-out", fontsize=8)
    ax[j].set_xlabel("false positive rate", fontsize=8); ax[j].set_ylabel("true positive rate", fontsize=8)
    ax[j].legend(fontsize=5.5, loc="lower right")
names = ["all (deployed)", "shear only", "force only", "force + time"]
x = np.arange(len(names)); w = 0.38
for k, split in enumerate(["LOBO", "LOSO"]):
    v = [int(A[(A.features == n) & (A.split == split)].caught.iloc[0].split("/")[0]) for n in names]
    b = ax[2].bar(x + (k - 0.5) * w, v, w, label=split)
    ax[2].bar_label(b, fontsize=6)
ax[2].set_xticks(x); ax[2].set_xticklabels(["all", "shear", "force", "force\n+ time"], fontsize=7)
ax[2].set_ylim(40, 77); ax[2].set_yticks(range(40, 71, 5)); ax[2].axhline(70, color="0.6", lw=0.8, ls=":")
ax[2].set_ylabel("drops detected", fontsize=8); ax[2].set_title("(c) drops detected (of 70)", fontsize=8)
ax[2].legend(fontsize=5.5, loc="upper center", ncol=2, framealpha=0.9, handlelength=1.2, columnspacing=1.0)
for a in ax: a.tick_params(labelsize=7); a.grid(alpha=0.3)
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.expanduser(f"~/Desktop/fig_roc_ablation.{e}"), dpi=300)
print("saved ~/Desktop/fig_roc_ablation.png/.pdf")
