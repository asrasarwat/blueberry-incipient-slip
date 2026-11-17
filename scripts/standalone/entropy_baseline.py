import os, json, numpy as np, pandas as pd
from scipy.stats import binomtest
import claim1_check_7 as C
H = os.path.expanduser("~"); norm = lambda v: f"{float(v):.1f}"
BINS = np.r_[np.arange(0, 5.0001, 0.25), np.inf]
def entropy_series(z, k, key, hold):
    m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0); t = z["t"][m]
    M = z[f"{key}_flow"][m].astype(float)[..., :2]; F = np.abs(z[key][m][..., 2].astype(float))
    ref = (t >= hold) & (t < hold + 1.0)
    if ref.sum() < 3: return None
    fz = F[ref].mean(0); cm = C.resize_mask((fz > C.CONTACT_FRAC * fz.max()) & (fz > 1e-3), M.shape[1:3])
    if cm.sum() < 8: return None
    D = C.causal_mean(M - M[ref].mean(0), C.SMOOTH_N); u = np.hypot(D[..., 0], D[..., 1])[:, cm]
    Hs = np.empty(len(u))
    for i, row in enumerate(u):
        p = np.histogram(row, BINS)[0].astype(float); p = p[p > 0] / p.sum(); Hs[i] = -(p * np.log2(p)).sum()
    return t, Hs
def onset(series, b0, b1, t0, t1):
    best = None
    for t, h in series:
        o = C.zrule(t, h, b0, b1, t0, t1, signed=True)
        if o is not None and (best is None or o < best): best = o
    return best
G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv").set_index("trial")
rows = []
for s in C.SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        ev = json.load(open(f"{H}/shear_{s}mm/{r.folder}_events.json")); hold, loosen, drop = float(ev["hold"]), float(ev["loosen"]), float(r.drop_camera_s)
        z = np.load(f"{H}/grasp_data/blueberry/size_{s}mm/{r.folder}/taxels.npz")
        ser = [x for x in (entropy_series(z, k, key, hold) for k, key in enumerate(("index", "thumb"))) if x is not None]
        on = onset(ser, hold, loosen, loosen, drop) if ser else None
        mid = (hold + loosen) / 2; fa = (onset(ser, hold, mid, mid, loosen) is not None) if ser else False
        cam = G.loc[r.folder, "cam_0.3mm"] if r.folder in G.index else np.nan
        stick = G.loc[r.folder, "stick_onset"] if r.folder in G.index else np.nan
        rows.append(dict(trial=r.folder, size=s, onset=on, hold_FA=fa, drop=drop, cam03=cam, stick=stick,
                         minus_cam03=(on - cam) if on is not None and pd.notna(cam) else np.nan,
                         warning=(drop - on) if on is not None else np.nan))
        print(f"--- {r.folder} ({s} mm): entropy onset {on}", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{H}/claim1_results/entropy_baseline_per_trial.csv", index=False)
x = D.minus_cam03.dropna(); k = int((x <= 0).sum())
print(f"\n=== Entropy baseline, {len(D)} trials ===")
print(f"detected {D.onset.notna().sum()}/{len(D)}, hold false alarms {int(D.hold_FA.sum())}/{len(D)}, warning median {D.warning.median():.2f} s")
print(f"before 0.3 mm gross slip: {k}/{len(x)}, sign-test p = {binomtest(k, len(x), 0.5).pvalue:.4f}, median lead {-x.median():+.2f} s")
both = D.dropna(subset=["onset", "stick"]); kb = int((both.stick < both.onset).sum())
print(f"head-to-head: stick ratio earlier in {kb}/{len(both)} (p = {binomtest(kb, len(both), 0.5).pvalue:.2g}), median entropy - stick {(both.onset - both.stick).median():+.2f} s")
for s in C.SIZES:
    d = D[D["size"] == s]; xs = d.minus_cam03.dropna()
    print(f"   {s} mm: detected {d.onset.notna().sum()}/{len(d)}, FA {int(d.hold_FA.sum())}, before gross slip {int((xs <= 0).sum())}/{len(xs)}, median lead {-xs.median():+.2f} s")
