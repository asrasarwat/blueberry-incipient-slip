import os
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~"); SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
rows = []
for s in SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        try:
            R = pd.read_csv(f"{H}/rot_{s}mm/{r.folder}_rot.csv")
            lo = float(json.load(open(f"{H}/shear_{s}mm/{r.folder}_events.json"))["loosen"])
        except FileNotFoundError:
            print("skip", r.folder); continue
        w = R[(R.t_s >= lo) & (R.t_s <= float(r.drop_camera_s) - 0.3) & (R.score >= 0.5)]
        rows.append(dict(size=s, trial=r.folder, max_abs_rot_deg=w.rot_deg.abs().max() if len(w) else np.nan))
Rt = pd.DataFrame(rows); Rt.to_csv(f"{H}/claim1_results/rotation_per_trial.csv", index=False)
W = pd.read_csv(f"{H}/model_results/warn_per_trial_loso.csv"); W["size"] = W["size"].map(lambda v: f"{float(v):.1f}")
print("rotation during loosening (deg):"); print(Rt.groupby("size").max_abs_rot_deg.describe()[["count", "50%", "min", "max"]].round(1))
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.5))
lab = [s.replace(".0", "") for s in SIZES]
ax[0].boxplot([Rt[Rt["size"] == s].max_abs_rot_deg.dropna() for s in SIZES], tick_labels=lab, flierprops=dict(markersize=3))
ax[0].set_xlabel("berry size (mm)", fontsize=8); ax[0].set_ylabel("max |rotation| (deg)", fontsize=8); ax[0].set_title("(a) berry rotation before the drop", fontsize=8)
ax[1].boxplot([W[W["size"] == s].warn_s.dropna() for s in SIZES], tick_labels=lab, flierprops=dict(markersize=3))
ax[1].set_xlabel("held-out berry size (mm)", fontsize=8); ax[1].set_ylabel("warning before drop (s)", fontsize=8); ax[1].set_title("(b) model warning, leave-one-size-out", fontsize=8)
for a in ax: a.grid(alpha=0.3); a.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(f"{H}/Desktop/fig_rotation_warning.png", dpi=300); fig.savefig(f"{H}/Desktop/fig_rotation_warning.pdf"); print("saved ~/Desktop/fig_rotation_warning.png")
