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
            df["trial"], df["size"], df["berry"] = r.folder, s, r.berry_id; parts.append(df)
D = pd.concat(parts, ignore_index=True)
D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int)
D["t_since_loosen"] = (D.t - D.loosen).clip(lower=0)
F = list(SF.FEATS); force = [f for f in F if "fz" in f.lower()]; shear = [f for f in F if f not in force]
sets = {"all": F, "shear": shear, "force": force, "time": ["t_since_loosen"]}
out = []
for name, cols in sets.items():
    for split, key in [("LOBO", "berry"), ("LOSO", "size")]:
        for g in D[key].unique():
            tr, te = D[D[key] != g], D[D[key] == g]
            m = T.model().fit(tr[cols], tr.y)
            out.append(pd.DataFrame(dict(features=name, split=split, y=te.y.values, p=m.predict_proba(te[cols])[:, 1])))
        print("done:", name, split, flush=True)
pd.concat(out).to_csv(os.path.expanduser("~/model_results/roc_preds_v4.csv"), index=False)
print("saved ~/model_results/roc_preds_v4.csv")
