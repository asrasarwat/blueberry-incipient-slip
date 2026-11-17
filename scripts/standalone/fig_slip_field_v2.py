import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import claim1_check_7 as C
H = os.path.expanduser("~")
G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv")
G["lead"] = G["stick_minus_cam_0.3mm"]
cand = G[G["size"].astype(float).between(16, 18)].dropna(subset=["lead", "stick_onset", "cam_0.3mm"])
cand = cand[cand.lead < 0]
cand = cand.iloc[(cand.lead - cand.lead.median()).abs().argsort()]

def finger(z, k, key, hold):
    m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0)
    t = z["t"][m]; M = z[f"{key}_flow"][m].astype(float)[..., :2]
    F = np.abs(z[key][m][..., 2].astype(float))
    ref = (t >= hold) & (t < hold + 1.0)
    fz = F[ref].mean(0)
    cm = C.resize_mask((fz > C.CONTACT_FRAC * fz.max()) & (fz > 1e-3), M.shape[1:3])
    D = C.causal_mean(M - M[ref].mean(0), C.SMOOTH_N)
    u = np.hypot(D[..., 0], D[..., 1])
    return dict(name=key, t=t, D=D, u=u, cm=cm, stick=(u[:, cm] < C.U_THR).mean(1))

pick = None
for _, row in cand.iterrows():
    s, trial = f"{float(row['size']):.1f}", row.trial
    ev = json.load(open(f"{H}/shear_{s}mm/{trial}_events.json"))
    hold, loosen = float(ev["hold"]), float(ev["loosen"])
    drop = float(pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv").set_index("folder").loc[trial, "drop_camera_s"])
    z = np.load(f"{H}/grasp_data/blueberry/size_{s}mm/{trial}/taxels.npz")
    FI = [finger(z, k, key, hold) for k, key in enumerate(("index", "thumb"))]
    t_g = float(row["cam_0.3mm"])
    ok = all(np.std(f["stick"][(f["t"] > t_g) & (f["t"] < drop)]) > 0.02 for f in FI)
    print(f"{trial} ({s} mm) lead {row.lead:+.2f} s: {'both fingers vary' if ok else 'a finger goes flat'}")
    if ok:
        pick = (s, trial, hold, loosen, drop, float(row.stick_onset), t_g, FI); break
if pick is None: raise SystemExit("no trial with both fingers varying")
s, trial, hold, loosen, drop, t_on, t_g, FI = pick

tg = np.unique(np.concatenate([f["t"] for f in FI]))
comb = np.min([np.interp(tg, f["t"], f["stick"]) for f in FI], axis=0)
w = lambda f: (f["t"] > loosen) & (f["t"] < drop)
f = min(FI, key=lambda f: f["stick"][w(f)].min())
times = [(loosen - 0.5, "(a) hold"), (t_on, "(b) stick-ratio onset"), (t_g, "(c) 0.3 mm gross slip"), (drop - 1.0, "(d) 1 s before drop")]
def pk(tt):
    w_ = np.where(np.abs(f["t"] - tt) < 0.3)[0]
    if len(w_) == 0: return int(np.argmin(np.abs(f["t"] - tt)))
    med = np.median(f["stick"][w_]); return int(w_[np.argmin(np.abs(f["stick"][w_] - med))])
idx = [pk(tt) for tt, _ in times]
smax = np.percentile(np.concatenate([f["u"][i][f["cm"]] for i in idx]), 95)
yy, xx = np.mgrid[0:f["cm"].shape[0], 0:f["cm"].shape[1]]
fig = plt.figure(figsize=(7.16, 4.8))
for j, (i, (tt, lab)) in enumerate(zip(idx, times)):
    ax = fig.add_subplot(2, 4, j + 1)
    D, u, cm = f["D"][i], f["u"][i], f["cm"]
    col = np.where(~cm, "0.75", np.where(u < C.U_THR, "tab:blue", "tab:red"))
    ax.scatter(xx.ravel(), yy.ravel(), c=col.ravel(), s=6, zorder=2)
    ax.quiver(xx[cm], yy[cm], D[..., 0][cm], -D[..., 1][cm], angles="xy", scale_units="xy",
              scale=smax / 1.2 if smax > 0 else 1, width=0.006, color="k", zorder=3)
    ax.set_xlim(-1, cm.shape[1]); ax.set_ylim(cm.shape[0], -1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{lab}\nt = {f['t'][i]:.1f} s, stuck {100 * (u[cm] < C.U_THR).mean():.0f}%", fontsize=8)
ax = fig.add_subplot(2, 1, 2)
for g in FI:
    ax.plot(g["t"], g["stick"], lw=0.8, alpha=0.4, label=f"{g['name']}")
ax.plot(tg, comb, lw=1.6, color="k", label="stick ratio (min of fingers)")
for x, c, l in [(loosen, "0.4", "loosening starts"), (t_on, "tab:blue", "stick-ratio onset"),
                (t_g, "tab:orange", "0.3 mm gross slip (camera)"), (drop, "tab:red", "drop (camera)")]:
    ax.axvline(x, color=c, ls="--", lw=1, label=l)
ax.set_xlim(loosen - 3, drop + 0.5); ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("time (s)", fontsize=8); ax.set_ylabel("fraction of contact stuck", fontsize=8)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, ncol=4, loc="lower left"); ax.grid(alpha=0.3)
ax.set_title(f"(e) {s.replace('.0', '')} mm berry; (a)-(d): {f['name']} finger, blue stuck, red moved > 0.5 px, grey outside contact", fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{H}/Desktop/fig_slip_field_v2.{ext}", dpi=300)
print(f"picked {trial} ({s} mm); saved ~/Desktop/fig_slip_field_v2.png/.pdf")
