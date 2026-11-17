import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binomtest
H = os.path.expanduser("~"); norm = lambda v: f"{float(v):.1f}"
G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv"); G["size"] = G["size"].map(norm)
B = pd.read_csv(f"{H}/claim1_results/dong_baseline_per_trial.csv")[["trial", "onset"]].rename(columns={"onset": "rigid_onset"})
D = G.merge(B, on="trial", how="left")
E = pd.read_csv(f"{H}/claim1_results/entropy_baseline_per_trial.csv")[["trial", "onset"]].rename(columns={"onset": "entropy_onset"})
D = D.merge(E, on="trial", how="left")
MM = [0.1, 0.2, 0.3, 0.5]
meth = {"stick ratio (pre-registered)": ("stick_onset", "k", "o"), "displacement entropy": ("entropy_onset", "tab:blue", "^"), "rigid-body residual": ("rigid_onset", "tab:red", "s")}
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5))
print(f"{'method':22s} {'mm':>4s} {'first':>8s} {'p':>8s} {'median lead (s)':>16s}")
for k, (name, (col, c, mk)) in enumerate(meth.items()):
    frac, lo, hi, med, q1, q3 = [], [], [], [], [], []
    for mm in MM:
        lead = (D[f"cam_{mm}mm"] - D[col]).dropna()
        n, kk = len(lead), int((lead >= 0).sum())
        ci = binomtest(kk, n, 0.5).proportion_ci(0.95)
        frac.append(kk / n); lo.append(kk / n - ci.low); hi.append(ci.high - kk / n)
        med.append(lead.median()); q1.append(lead.quantile(.25)); q3.append(lead.quantile(.75))
        print(f"{name:22s} {mm:4.1f} {kk:3d}/{n:<4d} {binomtest(kk, n, 0.5).pvalue:8.4f} {lead.median():+16.2f}")
    xs = np.array(MM) + (k - 1) * 0.012
    ax[0].errorbar(xs, frac, yerr=[lo, hi], color=c, marker=mk, ms=4, lw=1.2, capsize=2, label=name)
    ax[1].plot(MM, med, color=c, marker=mk, ms=4, lw=1.2, label=name)
    ax[1].fill_between(MM, q1, q3, color=c, alpha=0.12)
ax[0].axhline(0.5, color="0.6", ls=":", lw=0.8)
ax[0].set_ylim(0, 1); ax[0].set_ylabel("fraction tactile first", fontsize=8)
ax[0].set_title("(a) onset before gross slip", fontsize=8)
ax[1].axhline(0, color="0.6", ls=":", lw=0.8); ax[1].set_ylabel("lead over gross slip (s)", fontsize=8)
ax[1].set_title("(b) median lead (IQR shaded)", fontsize=8)
for a in ax[:2]:
    a.set_xticks(MM); a.set_xlabel("camera gross-slip threshold (mm)", fontsize=8)
    a.axvline(0.3, color="tab:blue", lw=0.8, alpha=0.4)

SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]; x = np.arange(len(SIZES)); w = 0.27
for k, (name, (col, c, mk)) in enumerate(meth.items()):
    v = []
    for s in SIZES:
        lead = (D[D["size"] == s]["cam_0.3mm"] - D[D["size"] == s][col]).dropna()
        v.append((lead >= 0).mean())
    ax[2].bar(x + (k - 1) * w, v, w, color=c, alpha=0.85, label=name)
ax[2].axhline(0.5, color="0.6", ls=":", lw=0.8)
ax[2].set_xticks(x); ax[2].set_xticklabels([s.replace(".0", "") for s in SIZES], fontsize=7)
ax[2].set_ylim(0, 1); ax[2].set_xlabel("berry size (mm)", fontsize=8); ax[2].set_ylabel("fraction tactile first", fontsize=8)
ax[2].set_title("(c) per size, 0.3 mm threshold", fontsize=8); 
for a in ax: a.tick_params(labelsize=7); a.grid(alpha=0.3)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.9))
for e in ("png", "pdf"): fig.savefig(f"{H}/Desktop/fig_lead_thresholds_v2.{e}", dpi=300)
print("saved ~/Desktop/fig_lead_thresholds_v2.png/.pdf")
