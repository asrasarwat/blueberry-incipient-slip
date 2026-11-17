import os, json, numpy as np, pandas as pd
from scipy.ndimage import binary_erosion
import claim1_check_7 as C
HOME = os.path.expanduser("~")
K, NMARK, DUR = 5.0, 3, 0.25

def kabsch(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    return R, md - R @ ms

def residuals(P, D, inner, edge):
    src_i, src_e = P[inner], P[edge]
    out = np.empty((len(D), edge.sum()))
    for n in range(len(D)):
        R, tr = kabsch(src_i, src_i + D[n][inner])
        out[n] = np.linalg.norm(src_e @ R.T + tr - (src_e + D[n][edge]), axis=1)
    return out

def finger(z, k, key, hold, loosen):
    m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0)
    t = z["t"][m]
    M = z[f"{key}_flow"][m].astype(float)[..., :2]
    F = np.abs(z[key][m][..., 2].astype(float))
    ref = (t >= hold) & (t < hold + 1.0)
    if ref.sum() < 3: return None
    fz = F[ref].mean(0)
    cm = C.resize_mask((fz > C.CONTACT_FRAC * fz.max()) & (fz > 1e-3), M.shape[1:3])
    if cm.sum() < 8: return None
    inner = binary_erosion(cm, iterations=1)
    if inner.sum() < 3: inner = cm.copy()
    edge = cm & ~inner
    if edge.sum() <= NMARK: return None
    D = C.causal_mean(M - M[ref].mean(0), C.SMOOTH_N)
    yy, xx = np.mgrid[0:M.shape[1], 0:M.shape[2]]
    best = None
    for P in (np.stack([xx, yy], -1).astype(float), np.stack([yy, xx], -1).astype(float)):
        r = residuals(P, D, inner, edge)
        h = r[(t >= hold) & (t < loosen)].mean()
        if best is None or h < best[0]: best = (h, r)
    return t, best[1]

def onset(series, b0, b1, t0, t1):
    best = None
    for t, r in series:
        b = (t >= b0) & (t < b1)
        if b.sum() < 10: continue
        thr = r[b].mean() + K * max(r[b].std(), 1e-6)
        o = C.first_sustained(t, (r > thr).sum(1) > NMARK, t0, t1, DUR)
        if o is not None and (best is None or o < best): best = o
    return best

G = pd.read_csv(f"{HOME}/claim1_results/gross_slip_per_trial.csv").set_index("trial")
rows = []
for s in C.SIZES:
    tm = pd.read_csv(f"{HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        tr = f"{HOME}/grasp_data/blueberry/size_{s}mm/{r.folder}"
        ev = json.load(open(f"{HOME}/shear_{s}mm/{r.folder}_events.json"))
        hold, loosen, drop = float(ev["hold"]), float(ev["loosen"]), float(r.drop_camera_s)
        z = np.load(f"{tr}/taxels.npz")
        ser = [x for x in (finger(z, k, key, hold, loosen) for k, key in enumerate(("index", "thumb"))) if x is not None]
        if not ser: print("no contact", r.folder); continue
        on = onset(ser, hold, loosen, loosen, drop)
        mid = (hold + loosen) / 2
        fa = onset(ser, hold, mid, mid, loosen) is not None
        cam = G.loc[r.folder, "cam_0.3mm"] if r.folder in G.index else np.nan
        stick = G.loc[r.folder, "stick_onset"] if r.folder in G.index else np.nan
        rows.append(dict(trial=r.folder, size=s, onset=on, hold_FA=fa, drop=drop, cam03=cam, stick=stick,
                         minus_cam03=(on - cam) if on is not None and pd.notna(cam) else np.nan,
                         warning=(drop - on) if on is not None else np.nan))
        print(f"--- {r.folder} ({s} mm): rigid-body onset {on}, cam0.3 {cam}, stick {stick}", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{HOME}/claim1_results/dong_baseline_per_trial.csv", index=False)
x = D.minus_cam03.dropna()
print(f"\n=== Dong-style rigid-body detector, {len(D)} trials ===")
print(f"detected {D.onset.notna().sum()}/{len(D)}, hold false alarms {int(D.hold_FA.sum())}/{len(D)}, warning median {D.warning.median():.2f} s")
print(f"before 0.3 mm gross slip: {int((x <= 0).sum())}/{len(x)}, median {x.median():+.2f} s")
both = D.dropna(subset=["onset", "stick"])
print(f"head-to-head with stick ratio (n={len(both)}): rigid-body earlier in {int((both.onset < both.stick).sum())}, "
      f"median rigid - stick {(both.onset - both.stick).median():+.2f} s")
for s in C.SIZES:
    d = D[D["size"] == s]; xs = d.minus_cam03.dropna()
    print(f"   {s} mm: detected {d.onset.notna().sum()}/{len(d)}, FA {int(d.hold_FA.sum())}, before gross slip {int((xs <= 0).sum())}/{len(xs)}, median {xs.median():+.2f} s")
