import os
import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
import train_deploy_model_v4 as T, slip_features as SF
warnings.filterwarnings("ignore")
parts = []
for s in T.SIZES:
    tm = pd.read_csv(f"{T.HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        df = T.trial_samples(r.folder, s, float(r.drop_camera_s))
        if df is not None:
            df["trial"], df["size"], df["berry"] = r.folder, s, r.berry_id
            parts.append(df)
D = pd.concat(parts, ignore_index=True)
D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int)
D["t_since_loosen"] = (D.t - D.loosen).clip(lower=0)
F = list(SF.FEATS)
force = [f for f in F if "fz" in f.lower() or "force" in f.lower()]
shear = [f for f in F if f not in force]
print(f"{D.trial.nunique()} trials, {D.berry.nunique()} berries; force features ({len(force)}): {force}")
sets = {"all (deployed)": F, "shear only": shear, "force only": force,
        "time only": ["t_since_loosen"], "force + time": force + ["t_since_loosen"]}
rows = []
for name, cols in sets.items():
    for split, key in [("LOBO", "berry"), ("LOSO", "size")]:
        preds, caught, fa, warn = [], 0, 0, []
        for g in D[key].unique():
            tr, te = D[D[key] != g], D[D[key] == g]
            m = T.model().fit(tr[cols], tr.y)
            thr = T.thr_from(tr, m.predict_proba(tr[cols])[:, 1])
            te = te.assign(p=m.predict_proba(te[cols])[:, 1]); preds.append(te)
            for _, x in te.groupby("trial"):
                al = T.first_alarm(x.t.values, x.p.values, thr)
                if al is not None and al < x.loosen.iat[0]: fa += 1
                elif al is not None: caught += 1; warn.append(x["drop"].iat[0] - al)
        P = pd.concat(preds)
        rows.append(dict(features=name, split=split, auc=round(roc_auc_score(P.y, P.p), 3),
                         caught=f"{caught}/{D.trial.nunique()}", hold_FA=fa,
                         warn_med_s=round(np.median(warn), 2) if warn else None))
        print("done:", name, split, flush=True)
R = pd.DataFrame(rows); print("\n" + R.to_string(index=False))
R.to_csv(os.path.expanduser("~/model_results_ablation_v4.csv"), index=False)
