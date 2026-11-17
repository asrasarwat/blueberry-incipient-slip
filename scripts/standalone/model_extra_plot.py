import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8}); H = os.path.expanduser("~")
P = pd.read_csv(f"{H}/model_results/extra_P.csv"); R = pd.read_csv(f"{H}/model_results/extra_R.csv")
R["alarm"] = pd.to_numeric(R["alarm"], errors="coerce"); R["warn"] = R["drop"] - R["alarm"] - 0.25; R["hold_fa"] = R["alarm"] < R["loosen"]
tp = int((R.alarm.notna() & ~R.hold_fa).sum()); fn = len(R) - tp; fp = int(R.hold_fa.sum()); tn = len(R) - fp
print(f"event level: drops alarmed {tp}/{len(R)}, holds with false alarm {fp}/{len(R)}")
fig, ax = plt.subplots(1, 4, figsize=(11, 2.7))
M = np.array([[tp, fn], [fp, tn]]); ax[0].imshow(M, cmap="Blues", vmin=0, vmax=len(R))
for i in range(2):
    for j in range(2): ax[0].text(j, i, M[i, j], ha="center", va="center", fontsize=12, color="white" if M[i, j] > 35 else "black")
ax[0].set(xticks=[0, 1], xticklabels=["alarm", "no alarm"], yticks=[0, 1], yticklabels=["drop (70)", "stable hold (70)"], title="(a) event level, LOSO")
sizes = sorted(R["size"].unique()); data = [R[R["size"] == s].warn.dropna().values for s in sizes]
ax[1].boxplot(data, tick_labels=[f"{int(s)}" for s in sizes], widths=0.6)
ax[1].axhline(0, color="0.5", ls=":", lw=0.8); ax[1].set(xlabel="berry size (mm)", ylabel="causal warning (s)", title="(b) warning before the drop")
bins = np.linspace(0, 1, 11); P["bin"] = np.clip(np.digitize(P.p, bins) - 1, 0, 9)
c = P.groupby("bin").agg(pm=("p", "mean"), ym=("y", "mean"), n=("y", "size")); c = c[c.n >= 20]
ax[2].plot([0, 1], [0, 1], ":", color="0.5"); ax[2].plot(c.pm, c.ym, "o-", ms=4)
ax[2].set(xlabel="predicted probability", ylabel="observed fraction", title="(c) calibration", xlim=(0, 1), ylim=(0, 1))
P["ttd"] = P["drop"] - P.t; grid = np.arange(-15, 0.01, 0.25); rows = []
for tid, g in P.groupby("trial"):
    g = g.sort_values("ttd"); rows.append(np.interp(-grid, g.ttd.values, g.p.values, left=np.nan, right=np.nan))
A = np.array(rows); med, q1, q3 = np.nanmedian(A, 0), np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0)
ax[3].fill_between(grid, q1, q3, alpha=0.3); ax[3].plot(grid, med, lw=1.5); ax[3].axhline(P.thr.median(), color="r", ls=":", lw=1, label="median threshold")
ax[3].set(xlabel="time to drop (s)", ylabel="p(t)", title="(d) p(t) aligned to the drop"); ax[3].legend(frameon=False, fontsize=7)
for a in ax[1:]: a.grid(alpha=0.3)
fig.tight_layout(); out = f"{H}/paper_figures/figS_model_extra"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200); print("saved", out + ".png")
print("causal warning by size (median s):", R.groupby("size").warn.median().round(2).to_dict())
print("Brier score:", round(float(((P.p - P.y) ** 2).mean()), 4))
