import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr, mannwhitneyu
import claim1_check_7 as C
from dong_baseline import kabsch
H = os.path.expanduser("~"); norm = lambda v: f"{float(v):.1f}"
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
R = pd.read_csv(f"{H}/claim1_results/rotation_per_trial.csv"); R["size"] = R["size"].map(norm)
rows = []
for s in SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        ev = json.load(open(f"{H}/shear_{s}mm/{r.folder}_events.json")); hold, loosen = float(ev["hold"]), float(ev["loosen"])
        drop = float(r.drop_camera_s)
        z = np.load(f"{H}/grasp_data/blueberry/size_{s}mm/{r.folder}/taxels.npz")
        twist, dvec = [], []
        for k, key in enumerate(("index", "thumb")):
            m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0); t = z["t"][m]
            M = z[f"{key}_flow"][m].astype(float)[..., :2]; F = np.abs(z[key][m][..., 2].astype(float))
            ref = (t >= hold) & (t < hold + 1.0)
            if ref.sum() < 3: continue
            fz = F[ref].mean(0); cm = C.resize_mask((fz > C.CONTACT_FRAC * fz.max()) & (fz > 1e-3), M.shape[1:3])
            if cm.sum() < 8: continue
            inner = binary_erosion(cm, iterations=1)
            if inner.sum() < 3: inner = cm.copy()
            D = C.causal_mean(M - M[ref].mean(0), C.SMOOTH_N)
            yy, xx = np.mgrid[0:M.shape[1], 0:M.shape[2]]; P = np.stack([xx, yy], -1).astype(float)
            w = np.where((t > loosen) & (t < drop - 0.3))[0][::3]
            ang = [np.degrees(np.arctan2(*(lambda Rm: (Rm[1, 0], Rm[0, 0]))(kabsch(P[inner], P[inner] + D[i][inner])[0]))) for i in w]
            twist.append(np.max(np.abs(ang)) if ang else np.nan)
            late = (t >= drop - 1.0) & (t < drop - 0.3)
            dvec.append(D[late][:, cm].mean((0, 1)) if late.any() else np.full(2, np.nan))
        cosang = np.nan
        if len(dvec) == 2 and np.all(np.isfinite(dvec)):
            a, b = dvec; cosang = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        rows.append(dict(trial=r.folder, size=s, twist_deg=np.nanmax(twist) if twist else np.nan, cos_index_thumb=cosang))
        print(f"--- {r.folder} ({s} mm): twist {rows[-1]['twist_deg']:.2f} deg, cos {cosang:+.2f}", flush=True)
T = pd.DataFrame(rows).merge(R[["trial", "max_abs_rot_deg"]], on="trial")
T.to_csv(f"{H}/claim1_results/tactile_rotation.csv", index=False)
print("\nper size (medians):"); print(T.groupby("size")[["max_abs_rot_deg", "twist_deg", "cos_index_thumb"]].median().round(2))
for c in ("twist_deg", "cos_index_thumb"):
    q = T.dropna(subset=[c, "max_abs_rot_deg"]); rho, p = spearmanr(q[c], q.max_abs_rot_deg)
    print(f"camera rotation vs {c}: Spearman rho = {rho:+.2f}, p = {p:.4f}, n = {len(q)}")
roll = T.max_abs_rot_deg >= 10
for c in ("twist_deg", "cos_index_thumb"):
    a, b = T[roll][c].dropna(), T[~roll][c].dropna()
    print(f"{c}: rolling (>=10 deg, n={len(a)}) median {a.median():+.2f} vs sliding (n={len(b)}) median {b.median():+.2f}, Mann-Whitney p = {mannwhitneyu(a, b).pvalue:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6)); cmap = plt.get_cmap("viridis")
for j, (c, lab) in enumerate([("twist_deg", "tactile in-plane twist (deg)"), ("cos_index_thumb", "cosine, index vs thumb shear direction")]):
    for i, s in enumerate(SIZES):
        q = T[T["size"] == s]; ax[j].scatter(q.max_abs_rot_deg, q[c], s=12, color=cmap(i / 6), label=s.replace(".0", ""))
    q = T.dropna(subset=[c]); rho, p = spearmanr(q[c], q.max_abs_rot_deg)
    ax[j].set_xlabel("camera berry rotation (deg)", fontsize=8); ax[j].set_ylabel(lab, fontsize=8)
    ax[j].set_title(f"({'ab'[j]}) rho = {rho:+.2f}, p = {p:.3g}", fontsize=8); ax[j].tick_params(labelsize=7); ax[j].grid(alpha=0.3)
ax[1].legend(title="mm", fontsize=6, title_fontsize=6, ncol=2)
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(f"{H}/Desktop/fig_tactile_rotation.{e}", dpi=300)
print("saved ~/Desktop/fig_tactile_rotation.png/.pdf")

W = pd.read_csv(f"{H}/model_results/warn_per_trial_loso.csv")
caught = int(W.warn_s.notna().sum()); fa_hold = int((W.alarm_s.notna() & W.warn_s.isna()).sum())
nop = pd.read_csv(f"{H}/model_results/nopull_fa_v4.csv"); nop_ok = int((~nop.false_alarm).sum())
correct = caught + (len(W) - fa_hold) + nop_ok; total = len(W) + len(W) + len(nop)
print(f"\nDong-style accuracy (learned model, LOSO): drops detected {caught}/{len(W)}, holds without false alarm {len(W) - fa_hold}/{len(W)}, "
      f"no-pull controls without alarm {nop_ok}/{len(nop)} -> {correct}/{total} = {100 * correct / total:.1f}%")
