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
            df["trial"], df["size"], df["berry"] = r.folder, s, r.berry_id; parts.append(df)
D = pd.concat(parts, ignore_index=True); D["y"] = ((D["drop"] - D.t) <= 3.0).astype(int); F = list(SF.FEATS)
rng = np.random.default_rng(0)
for split, key in [("LOBO", "berry"), ("LOSO", "size")]:
    P = []
    for g in D[key].unique():
        tr, te = D[D[key] != g], D[D[key] == g]
        m = T.model().fit(tr[F], tr.y); P.append(te.assign(p=m.predict_proba(te[F])[:, 1]))
    P = pd.concat(P); berries = P.berry.unique(); by = {b: P[P.berry == b] for b in berries}
    boot = [roc_auc_score(*(lambda q: (q.y, q.p))(pd.concat([by[b] for b in rng.choice(berries, len(berries))]))) for _ in range(1000)]
    print(f"{split}: AUC {roc_auc_score(P.y, P.p):.3f}, 95% CI {np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f} (1000 berry-level bootstraps)")
