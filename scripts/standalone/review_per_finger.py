import os, json, numpy as np, pandas as pd
from scipy.stats import binomtest
import claim1_check_7 as C
H = os.path.expanduser("~"); G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv").set_index("trial")
rows = []
for s in C.SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        T = f"{H}/grasp_data/blueberry/size_{s}mm/{r.folder}"
        ev = json.load(open(f"{H}/shear_{s}mm/{r.folder}_events.json")); hold, loosen, drop = float(ev["hold"]), float(ev["loosen"]), float(r.drop_camera_s)
        z = np.load(f"{T}/taxels.npz"); ser, w = {}, {}
        for k, key in enumerate(("index", "thumb")):
            x = C.finger_series(z, k, key, hold)
            if x is None: continue
            ser[key] = x; m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0); t = z["t"][m]
            w[key] = np.abs(z[key][m][..., 2].astype(float)).sum((1, 2))[(t >= hold) & (t < hold + 1)].mean()
        if not ser: continue
        grid = np.unique(np.concatenate([v[0] for v in ser.values()]))
        S = {k: np.interp(grid, v[0], v[1]) for k, v in ser.items()}
        agg = dict(S)
        if len(S) == 2:
            agg["min"] = np.minimum(S["index"], S["thumb"]); agg["mean"] = (S["index"] + S["thumb"]) / 2
            agg["force_weighted"] = (w["index"] * S["index"] + w["thumb"] * S["thumb"]) / (w["index"] + w["thumb"])
        cam = G.loc[r.folder, "cam_0.3mm"] if r.folder in G.index else np.nan
        row = dict(trial=r.folder, size=float(s))
        for name, st in agg.items():
            on = C.zrule(grid, 1 - st, hold, loosen, loosen, drop, signed=True)
            row[name] = (cam - 0.125 - on) if on is not None and pd.notna(cam) else np.nan
        rows.append(row)
D = pd.DataFrame(rows); D.to_csv(f"{H}/claim1_results/per_finger_aggregation.csv", index=False)
for name in ("index", "thumb", "min", "mean", "force_weighted"):
    for lab, sel in (("all", D["size"] > 0), ("14-18", D["size"] <= 18), ("20-21", D["size"] >= 20)):
        for tl, sh in (("onset", 0), ("causal", 0.25)):
            x = (D.loc[sel, name] - sh).dropna(); k = int((x >= 0).sum())
            if len(x): print(f"{name:15s} {lab:6s} {tl:7s} {k:2d}/{len(x):2d}  p = {binomtest(k, len(x)).pvalue:.4f}  median {x.median():+.2f} s")
