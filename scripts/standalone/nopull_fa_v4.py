import os
import glob, json, os, numpy as np, pandas as pd, joblib, warnings
import train_deploy_model_v4 as T, slip_features as SF
warnings.filterwarnings("ignore")
M = joblib.load(os.path.expanduser("~/grasp_dataset/slip_model_v4.pkl"))
m, thr = M["model"], M["threshold"]
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/shear_nopull16mm/*_shear.csv"))):
    name = os.path.basename(f).replace("_shear.csv", "")
    ev = json.load(open(f.replace("_shear.csv", "_events.json")))
    d = pd.read_csv(f).sort_values("t_s")
    hold, end = float(ev["hold"]), float(d.t_s.max())
    fs, recs, X, i = SF.FeatureStream(hold), d.to_dict("records"), [], 0
    grid = np.arange(hold + SF.BASE, end, T.DT)
    for t in grid:
        while i < len(recs) and recs[i]["t_s"] <= t:
            r = recs[i]; fs.push(r["t_s"], r["finger"], r); i += 1
        X.append(fs.vector(t))
    ok = [x is not None for x in X]
    if not any(ok):
        print("no features:", name); continue
    g, Xv = grid[ok], np.vstack([x for x in X if x is not None])
    p = m.predict_proba(pd.DataFrame(Xv, columns=SF.FEATS))[:, 1]
    al = T.first_alarm(g, p, thr)
    rows.append(dict(trial=name, monitored_s=round(g[-1] - g[0], 1), p_max=round(p.max(), 3),
                     false_alarm=al is not None, alarm_at_s=al))
R = pd.DataFrame(rows); print(R.to_string(index=False))
n, k = len(R), int(R.false_alarm.sum())
print(f"\nfalse alarms: {k}/{n} trials, total monitored {R.monitored_s.sum():.0f} s, threshold {thr:.3f}")
R.to_csv(os.path.expanduser("~/model_results/nopull_fa_v4.csv"), index=False)
