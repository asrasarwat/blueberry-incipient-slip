import numpy as np, pandas as pd, warnings
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
import train_deploy_model_v4 as T, slip_features as SF
warnings.filterwarnings("ignore")
parts = []
for s in T.SIZES:
    tm = pd.read_csv(f"{T.HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        df = T.trial_samples(r.folder, s, float(r.drop_camera_s))
        if df is not None: df["trial"], df["size"] = r.folder, s; parts.append(df)
D = pd.concat(parts, ignore_index=True); D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int)
ALL = list(SF.FEATS); FORCE = [f for f in ALL if "fz" in f]
print("force features:", FORCE)
def caught(feats, simple=False):
    out = {}
    for s in T.SIZES:
        tr, te = D[D["size"] != s], D[D["size"] == s]
        m = (LogisticRegression(max_iter=2000) if simple else T.model()).fit(tr[feats], tr.y)
        thr = T.thr_from(tr, m.predict_proba(tr[feats])[:, 1]); p = m.predict_proba(te[feats])[:, 1]
        for tid, g in te.assign(p=p).groupby("trial"):
            out[tid] = T.first_alarm(g.t.values, g.p.values, thr) is not None
    return pd.Series(out)
a, f, r = caught(ALL), caught(FORCE), caught(FORCE, simple=True)
print(f"LOSO drops alarmed: all {a.sum()}/70, force-only trees {f.sum()}/70, force+force-rate logistic {r.sum()}/70")
for name, b in (("force-only trees", f), ("force+rate logistic", r)):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    print(f"McNemar all vs {name}: {n10} only by all, {n01} only by {name}, exact p = {binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1:.4f}")
