import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import claim1_check_7 as C
H = os.path.expanduser("~"); norm = lambda v: f"{float(v):.1f}"
G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv"); G["size"] = G["size"].map(norm)
R = pd.read_csv(f"{H}/claim1_results/rotation_per_trial.csv");  R["size"] = R["size"].map(norm)
rows = []
for _, g in G.iterrows():
    s, tr = g["size"], g.trial
    ev = json.load(open(f"{H}/shear_{s}mm/{tr}_events.json")); hold = float(ev["hold"])
    z = np.load(f"{H}/grasp_data/blueberry/size_{s}mm/{tr}/taxels.npz")
    area, fz = [], []
    for k, key in enumerate(("index", "thumb")):
        m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0); t = z["t"][m]
        F = np.abs(z[key][m][..., 2].astype(float)); ref = (t >= hold) & (t < hold + 1.0)
        if ref.sum() < 3: continue
        f = F[ref].mean(0)
        M = z[f"{key}_flow"]
        cm = C.resize_mask((f > C.CONTACT_FRAC * f.max()) & (f > 1e-3), M.shape[1:3])
        area.append(int(cm.sum())); fz.append(float(f.sum()))
    rows.append(dict(trial=tr, size=s, contact_markers=np.mean(area) if area else np.nan,
                     hold_force=np.sum(fz) if fz else np.nan, lead=-g["stick_minus_cam_0.3mm"]))
    print(f"--- {tr} ({s} mm): contact markers {rows[-1]['contact_markers']:.0f}", flush=True)
D = pd.DataFrame(rows).merge(R[["trial", "max_abs_rot_deg"]], on="trial", how="left")
D.to_csv(f"{H}/claim1_results/failure_analysis.csv", index=False)
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
print("\nper size (medians):"); print(D.groupby("size")[["contact_markers", "hold_force", "max_abs_rot_deg", "lead"]].median().round(2))
d = D.dropna(subset=["contact_markers", "lead"])
rho, p = spearmanr(d.contact_markers, d.lead); print(f"\nlead vs contact patch: Spearman rho = {rho:.2f}, p = {p:.4f}, n = {len(d)}")
d2 = D.dropna(subset=["max_abs_rot_deg", "lead"])
rho2, p2 = spearmanr(d2.max_abs_rot_deg, d2.lead); print(f"lead vs rotation:      Spearman rho = {rho2:.2f}, p = {p2:.4f}, n = {len(d2)}")
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.8))
ax[0].boxplot([D[D["size"] == s].contact_markers.dropna() for s in SIZES], tick_labels=[s.replace(".0", "") for s in SIZES])
ax[0].set_xlabel("berry size (mm)", fontsize=8); ax[0].set_ylabel("contact patch (markers)", fontsize=8); ax[0].set_title("(a) contact patch size", fontsize=8)
cmap = plt.get_cmap("viridis")
for i, s in enumerate(SIZES):
    q = d[d["size"] == s]; ax[1].scatter(q.contact_markers, q.lead, s=12, color=cmap(i / 6), label=s.replace(".0", ""))
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("contact patch (markers)", fontsize=8); ax[1].set_ylabel("lead over 0.3 mm gross slip (s)", fontsize=8)
ax[1].set_title(f"(b) lead vs contact patch (rho = {rho:.2f}, p = {p:.3f})", fontsize=8); ax[1].legend(title="mm", fontsize=6, title_fontsize=6, ncol=2)
for a in ax: a.tick_params(labelsize=7); a.grid(alpha=0.3)
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(f"{H}/Desktop/fig_failure_analysis.{e}", dpi=300)
print("saved ~/Desktop/fig_failure_analysis.png/.pdf")
