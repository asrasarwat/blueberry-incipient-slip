import os
import numpy as np, pandas as pd, warnings
import train_deploy_model_v4 as T, slip_features as SF
warnings.filterwarnings("ignore")
parts = []
for s in T.SIZES:
    tm = pd.read_csv(f"{T.HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        df = T.trial_samples(r.folder, s, float(r.drop_camera_s))
        if df is not None:
            df["trial"], df["size"] = r.folder, s
            parts.append(df)
D = pd.concat(parts, ignore_index=True)
D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int)
F, rows = list(SF.FEATS), []
for s in T.SIZES:
    tr, te = D[D["size"] != s], D[D["size"] == s]
    m = T.model().fit(tr[F], tr.y)
    thr = T.thr_from(tr, m.predict_proba(tr[F])[:, 1])
    te = te.assign(p=m.predict_proba(te[F])[:, 1])
    for trial, x in te.groupby("trial"):
        al = T.first_alarm(x.t.values, x.p.values, thr)
        ok = al is not None and al >= x.loosen.iat[0]
        rows.append(dict(size=s, trial=trial, alarm_s=al, warn_s=(x["drop"].iat[0] - al) if ok else np.nan))
W = pd.DataFrame(rows)
W.to_csv(os.path.expanduser("~/model_results/warn_per_trial_loso.csv"), index=False)
print(W.groupby("size").warn_s.agg(["count", "median", "min", "max"]).round(2))
