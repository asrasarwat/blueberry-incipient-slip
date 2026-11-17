import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
GI = pd.read_csv(os.path.expanduser("~/model_results/perm_importance_groups.csv"))
L = pd.read_csv(os.path.expanduser("~/model_results/learning_curve.csv"))
g = GI.groupby("group").auc_drop.agg(["mean", "std"]).sort_values("mean")
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4))
col = ["tab:orange" if "force" in i else "tab:blue" for i in g.index]
ax[0].barh(g.index, g["mean"], xerr=g["std"], color=col, capsize=2)
ax[0].set_xlabel("AUC drop when permuted", fontsize=8); ax[0].set_title("(a) signal-group importance (LOSO)", fontsize=8)
s = L.groupby("n_train").agg(auc=("auc", "mean"), lo=("auc", lambda x: x.quantile(.25)), hi=("auc", lambda x: x.quantile(.75)), det=("caught_frac", "mean"))
ax[1].plot(s.index, s.auc, "k-o", ms=3, label="AUC"); ax[1].fill_between(s.index, s.lo, s.hi, color="k", alpha=0.12)
ax[1].plot(s.index, s.det, "--s", color="tab:green", ms=3, label="fraction of drops detected")
ax[1].set_xlabel("training trials", fontsize=8); ax[1].set_ylim(0.5, 1.02)
ax[1].set_title("(b) learning curve (held-out size)", fontsize=8); ax[1].legend(fontsize=6.5, loc="lower right")
for a in ax: a.tick_params(labelsize=7); a.grid(alpha=0.3)
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.expanduser(f"~/Desktop/fig_importance_learning.{e}"), dpi=300)
print("saved ~/Desktop/fig_importance_learning.png/.pdf")
