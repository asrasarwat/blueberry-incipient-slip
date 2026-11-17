#!/usr/bin/env python3
"""train_deploy_model.py - robot slip model with the same features as the live detector."""
import os, glob, json, argparse
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import slip_features as SF

HOME = os.path.expanduser("~")
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
DT, SUSTAIN, MARGIN = 0.05, 0.25, 0.02

def trial_samples(folder, size, drop):
    f = glob.glob(f"{HOME}/shear_{size}mm/{folder}_shear.csv")
    e = glob.glob(f"{HOME}/shear_{size}mm/{folder}_events.json")
    if not f or not e:
        return None
    ev = json.load(open(e[0]))
    hold, loosen = float(ev["hold"]), float(ev["loosen"])
    d = pd.read_csv(f[0]).sort_values("t_s")
    fs = SF.FeatureStream(hold)
    rows = d.to_dict("records")
    grid = np.arange(hold + SF.BASE, drop, DT)
    X, i = [], 0
    for t in grid:
        while i < len(rows) and rows[i]["t_s"] <= t:
            r = rows[i]; fs.push(r["t_s"], r["finger"], r); i += 1
        X.append(fs.vector(t))
    if any(x is None for x in X):
        return None
    df = pd.DataFrame(np.vstack(X), columns=SF.FEATS)
    df["t"], df["loosen"], df["drop"] = grid, loosen, drop
    return df

def model():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)

def thr_from(tr, p):
    hold = (tr.t < tr.loosen).values
    return float(np.max(p[hold])) + MARGIN if hold.any() else 0.5

def first_alarm(t, p, thr):
    on = None
    for ti, pi in zip(t, p):
        if pi > thr:
            on = ti if on is None else on
            if ti - on >= SUSTAIN:
                return on
        else:
            on = None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--out", default=f"{HOME}/grasp_dataset/slip_model_v4.pkl")
    a = ap.parse_args()
    parts = []
    for s in SIZES:
        tm = pd.read_csv(f"{HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
        for _, r in tm[tm.status == "valid"].iterrows():
            df = trial_samples(r.folder, s, float(r.drop_camera_s))
            if df is None:
                print("skip", r.folder); continue
            df["trial"], df["size"], df["berry"] = r.folder, s, r.berry_id
            parts.append(df)
    D = pd.concat(parts, ignore_index=True)
    D["y"] = ((D["drop"] - D.t) <= a.horizon).astype(int)
    print(f"{D.trial.nunique()} trials, {len(D)} samples (whole hold + loosening), {len(SF.FEATS)} features, horizon {a.horizon} s")
    rows = []
    for split, grp in (("leave-one-berry-out", "berry"), ("leave-one-size-out", "size")):
        aucs, per = [], []
        for g in D[grp].unique():
            tr, te = D[D[grp] != g], D[D[grp] == g]
            m = model().fit(tr[SF.FEATS], tr.y)
            thr = thr_from(tr, m.predict_proba(tr[SF.FEATS])[:, 1])
            p = m.predict_proba(te[SF.FEATS])[:, 1]
            if te.y.nunique() == 2:
                aucs.append(roc_auc_score(te.y, p))
            te = te.assign(p=p)
            for tn, x in te.groupby("trial"):
                al = first_alarm(x.t.values, x.p.values, thr)
                lo, dr = x.loosen.iat[0], x["drop"].iat[0]
                per.append(dict(trial=tn, alarm=al, false_hold=al is not None and al < lo,
                                warn=None if al is None or al < lo else dr - al))
        P = pd.DataFrame(per)
        ok = P[P.warn.notna()]
        rows.append(dict(split=split, auc=round(np.mean(aucs), 3), caught=f"{len(ok)}/{len(P)}",
                         hold_false_alarms=int(P.false_hold.sum()),
                         warn_median_s=round(ok.warn.median(), 2), warn_min_s=round(ok.warn.min(), 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    m = model().fit(D[SF.FEATS], D.y)
    thr = thr_from(D, m.predict_proba(D[SF.FEATS])[:, 1])
    joblib.dump({"model": m, "feats": SF.FEATS, "threshold": thr, "sustain_s": SUSTAIN, "dt": DT,
                 "horizon_s": a.horizon, "ref_win": SF.REF_WIN, "base_s": SF.BASE, "rate_win": SF.RATE_WIN,
                 "n_trials": int(D.trial.nunique())}, a.out)
    print(f"\nfinal model (all trials): threshold {thr:.3f}  ->  saved {a.out}")

if __name__ == "__main__":
    main()
