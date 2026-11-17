import os
import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
import train_deploy_model_v4 as T, slip_features as SF
warnings.filterwarnings("ignore"); rng = np.random.default_rng(0)
parts = []
for s in T.SIZES:
    tm = pd.read_csv(f"{T.HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        df = T.trial_samples(r.folder, s, float(r.drop_camera_s))
        if df is not None:
            df["trial"], df["size"] = r.folder, s; parts.append(df)
D = pd.concat(parts, ignore_index=True); D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int); F = list(SF.FEATS)
def group(f):
    for key, lab in [("edge_lag", "edge lag"), ("edge_tr", "edge translation"), ("ctr_tr", "centre translation"), ("fz", "normal force"), ("trans", "contact translation")]:
        if key in f: return lab
    return "other"
G = {f: group(f) for f in F}; print("feature groups:"); print(pd.Series(G).value_counts().to_string())
imp, gimp = [], []
for s in T.SIZES:
    tr, te = D[D["size"] != s], D[D["size"] == s]
    m = T.model().fit(tr[F], tr.y); base = roc_auc_score(te.y, m.predict_proba(te[F])[:, 1])
    def drop(cols):
        out = []
        for _ in range(5):
            X = te[F].copy(); idx = rng.permutation(len(X))
            for c in cols: X[c] = X[c].values[idx]
            out.append(base - roc_auc_score(te.y, m.predict_proba(X)[:, 1]))
        return np.mean(out)
    for f in F: imp.append(dict(size=s, feature=f, group=G[f], auc_drop=drop([f])))
    for g in sorted(set(G.values())): gimp.append(dict(size=s, group=g, auc_drop=drop([f for f in F if G[f] == g])))
    print("importance done:", s, flush=True)
I = pd.DataFrame(imp); GI = pd.DataFrame(gimp)
I.to_csv(os.path.expanduser("~/model_results/perm_importance.csv"), index=False); GI.to_csv(os.path.expanduser("~/model_results/perm_importance_groups.csv"), index=False)
print("\ngroup importance (mean AUC drop over held-out sizes):")
print(GI.groupby("group").auc_drop.agg(["mean", "std"]).sort_values("mean", ascending=False).round(3).to_string())
print("\ntop 10 features:"); print(I.groupby(["feature", "group"]).auc_drop.mean().sort_values(ascending=False).head(10).round(3).to_string())
trials = D.drop_duplicates("trial")[["trial", "size"]]; rows = []
for n in (10, 20, 30, 40, 50, 60):
    for rep in range(5):
        for s in T.SIZES:
            pool = trials[trials["size"] != s].trial.values
            pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
            tr, te = D[D.trial.isin(pick)], D[D["size"] == s]
            if tr.y.nunique() < 2: continue
            m = T.model().fit(tr[F], tr.y); p = m.predict_proba(te[F])[:, 1]; thr = T.thr_from(tr, m.predict_proba(tr[F])[:, 1])
            te = te.assign(p=p); caught = fa = 0
            for _, x in te.groupby("trial"):
                al = T.first_alarm(x.t.values, x.p.values, thr)
                if al is not None and al < x.loosen.iat[0]: fa += 1
                elif al is not None: caught += 1
            rows.append(dict(n_train=n, rep=rep, size=s, auc=roc_auc_score(te.y, p), caught_frac=caught / te.trial.nunique(), fa=fa))
    print("learning curve done: n =", n, flush=True)
L = pd.DataFrame(rows); L.to_csv(os.path.expanduser("~/model_results/learning_curve.csv"), index=False)
print("\nlearning curve (mean over sizes and repeats):")
print(L.groupby("n_train").agg(auc=("auc", "mean"), auc_sd=("auc", "std"), detected=("caught_frac", "mean"), false_alarms=("fa", "sum")).round(3).to_string())
