import os, json, warnings, numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
H = os.path.expanduser("~"); G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv").set_index("trial")
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]

print("=== A. Stick-slip vibration: 10-40 Hz power of normal force, per phase (FORTE-style) ===")
def bandpower(t, f, t0, t1, fs=95.0):
    g = np.arange(t0, t1, 1 / fs)
    if len(g) < 48: return np.nan
    x = np.interp(g, t, f); x = x - np.polyval(np.polyfit(g, x, 2), g)
    fr, P = welch(x, fs=fs, nperseg=min(64, len(x)))
    return float(np.trapz(P[(fr >= 10) & (fr <= 40)], fr[(fr >= 10) & (fr <= 40)]))
rows = []
for s in SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        if r.folder not in G.index: continue
        g = G.loc[r.folder]; lo, on, gs, dr = g["loosen"], g["stick_onset"], g["cam_0.3mm"], float(r.drop_camera_s)
        if not (np.isfinite(on) and np.isfinite(gs)) or not (lo < on < gs - 0.125 < dr): continue
        z = np.load(f"{H}/grasp_data/blueberry/size_{s}mm/{r.folder}/taxels.npz"); row = dict(trial=r.folder, size=float(s))
        for k, key in enumerate(("index", "thumb")):
            m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0); t = z["t"][m]; f = np.abs(z[key][m][..., 2].astype(float)).sum((1, 2))
            row[f"{key}_hold"] = bandpower(t, f, lo - 3, lo)
            row[f"{key}_pre"] = bandpower(t, f, max(lo, on - 3), on)
            row[f"{key}_inc"] = bandpower(t, f, on, gs - 0.125)
            row[f"{key}_gross"] = bandpower(t, f, gs - 0.125, dr)
        rows.append(row)
V = pd.DataFrame(rows)
for key in ("index", "thumb"):
    for a, b in (("pre", "inc"), ("hold", "inc"), ("inc", "gross")):
        d = V[[f"{key}_{a}", f"{key}_{b}"]].dropna(); ratio = d[f"{key}_{b}"] / d[f"{key}_{a}"]
        print(f"{key:5s} {b} vs {a}: power ratio median {ratio.median():.2f}, higher in {int((ratio > 1).sum())}/{len(d)}, Wilcoxon p = {wilcoxon(d[f'{key}_{b}'], d[f'{key}_{a}']).pvalue:.4f}")

import train_deploy_model_v4 as T, slip_features as SF
parts = []
for s in T.SIZES:
    tm = pd.read_csv(f"{T.HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        df = T.trial_samples(r.folder, s, float(r.drop_camera_s))
        if df is not None: df["trial"], df["size"] = r.folder, s; parts.append(df)
D = pd.concat(parts, ignore_index=True); D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int); ALL = list(SF.FEATS)
def loso(feats, Xtr_map=None, Xte_map=None):
    ps, ys, caught, warns = [], [], 0, []
    for s in T.SIZES:
        tr, te = D[D["size"] != s], D[D["size"] == s]
        Xtr = tr[Xtr_map].values if Xtr_map else tr[feats].values; Xte = te[Xte_map].values if Xte_map else te[feats].values
        m = T.model().fit(Xtr, tr.y); ptr = m.predict_proba(Xtr)[:, 1]; thr = T.thr_from(tr, ptr); pte = m.predict_proba(Xte)[:, 1]
        ps.append(pte); ys.append(te.y.values)
        for tid, g in te.assign(p=pte).groupby("trial"):
            a = T.first_alarm(g.t.values, g.p.values, thr)
            if a is not None: caught += 1; warns.append(g["drop"].iat[0] - a - 0.25)
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps)), caught, np.median(warns) if warns else np.nan

print("\n=== B. Static vs dynamic features (LOSO) ===")
dyn = [f for f in ALL if f.endswith("_rate")]; sta = [f for f in ALL if f not in dyn]
print("dynamic features:", dyn)
for name, fs in (("all 29", ALL), (f"static only ({len(sta)})", sta), (f"dynamic only ({len(dyn)})", dyn)):
    auc, c, w = loso(fs); print(f"{name:22s} AUC {auc:.3f} | drops alarmed {c}/70 | causal warning median {w:.2f} s")

print("\n=== C. Cross-fingertip transfer (LOSO): train on one sensor, test on the other ===")
idx = sorted(f for f in ALL if f.startswith("index_")); thb = [f.replace("index_", "thumb_") for f in idx]
assert all(f in ALL for f in thb), "feature names do not pair up"
for name, tr_f, te_f in (("index only -> index", idx, idx), ("thumb only -> thumb", thb, thb), ("train index -> test thumb", idx, thb), ("train thumb -> test index", thb, idx)):
    auc, c, w = loso(None, tr_f, te_f); print(f"{name:26s} AUC {auc:.3f} | drops alarmed {c}/70 | causal warning median {w:.2f} s")
