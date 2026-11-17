import os, warnings, numpy as np, pandas as pd
import train_deploy_model_v4 as T, slip_features as SF
warnings.filterwarnings("ignore")
H = os.path.expanduser("~"); G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv").set_index("trial")
parts = []
for s in T.SIZES:
    tm = pd.read_csv(f"{T.HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        df = T.trial_samples(r.folder, s, float(r.drop_camera_s))
        if df is not None: df["trial"], df["size"] = r.folder, s; parts.append(df)
D = pd.concat(parts, ignore_index=True); D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int); F = list(SF.FEATS)
P, R = [], []
for s in T.SIZES:
    tr, te = D[D["size"] != s], D[D["size"] == s]
    m = T.model().fit(tr[F], tr.y); thr = T.thr_from(tr, m.predict_proba(tr[F])[:, 1])
    te = te.assign(p=m.predict_proba(te[F])[:, 1], thr=thr); P.append(te[["trial", "size", "t", "drop", "y", "p", "thr"]])
    for tid, g in te.groupby("trial"):
        a = T.first_alarm(g.t.values, g.p.values, thr); lo = G.loc[tid, "loosen"] if tid in G.index else np.nan
        R.append(dict(trial=tid, size=float(s), alarm=a, drop=g["drop"].iat[0], loosen=lo))
os.makedirs(f"{H}/model_results", exist_ok=True)
pd.concat(P).to_csv(f"{H}/model_results/extra_P.csv", index=False); pd.DataFrame(R).to_csv(f"{H}/model_results/extra_R.csv", index=False)
print("saved model_results/extra_P.csv and extra_R.csv")
