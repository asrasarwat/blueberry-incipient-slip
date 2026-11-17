#!/usr/bin/env python3
"""claim1_check.py - does a pre-registered tactile feature change at or before the camera onset?"""
import os, json
import numpy as np, pandas as pd

HOME = os.path.expanduser("~")
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
Z, DUR = 5.0, 0.25
U_THR = 0.5
SMOOTH_N = 5
CONTACT_FRAC = 0.10
MIN_F = 0.05


def resize_mask(mask, shape):
    H, W = shape
    h, w = mask.shape
    ri = np.clip((np.arange(H) * h / H).astype(int), 0, h - 1)
    ci = np.clip((np.arange(W) * w / W).astype(int), 0, w - 1)
    return mask[np.ix_(ri, ci)]


def causal_mean(x, n):
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], 0), 0)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - n + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def first_sustained(t, above, t_from, t_to, dur):
    start = None
    for ti, a in zip(t, above):
        if ti < t_from:
            continue
        if ti > t_to:
            break
        if a:
            start = ti if start is None else start
            if ti - start >= dur:
                return start
        else:
            start = None
    return None


def finger_series(z, k, key, hold):
    m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0)
    t = z["t"][m]
    M = z[f"{key}_flow"][m].astype(np.float64)[..., :2]
    F = np.abs(z[key][m][..., 2].astype(np.float64))
    ref = (t >= hold) & (t < hold + 1.0)
    if ref.sum() < 3:
        return None
    fz_ref = F[ref].mean(0)
    cmask_f = (fz_ref > CONTACT_FRAC * fz_ref.max()) & (fz_ref > 1e-3)
    cmask = resize_mask(cmask_f, M.shape[1:3])
    if cmask.sum() < 8:
        return None
    D = causal_mean(M - M[ref].mean(0), SMOOTH_N)
    u = np.hypot(D[..., 0], D[..., 1])[:, cmask]
    stick = (u < U_THR).mean(1)
    force = F.sum((1, 2))
    area = (F > CONTACT_FRAC * fz_ref.max()).sum((1, 2)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where((force > MIN_F) & (area > 0), np.log(area) - (2 / 3) * np.log(force), np.nan)
    return t, stick, resid


def zrule(t, x, b0, b1, t_from, t_to, signed):
    m = (t >= b0) & (t < b1) & np.isfinite(x)
    if m.sum() < 10:
        return None
    mu, sd = x[m].mean(), max(x[m].std(), 1e-6)
    zz = (x - mu) / sd
    zz = zz if signed else np.abs(zz)
    zz[~np.isfinite(zz)] = 0
    return first_sustained(t, zz > Z, t_from, t_to, DUR)


def onsets(series, b0, b1, t_from, t_to):
    out = {}
    grid = np.unique(np.concatenate([s[0] for s in series]))
    st = np.min([np.interp(grid, s[0], s[1]) for s in series], axis=0)
    out["min_stick_frac"] = zrule(grid, 1 - st, b0, b1, t_from, t_to, signed=True)
    best = None
    for s in series:
        o = zrule(s[0], s[2], b0, b1, t_from, t_to, signed=False)
        if o is not None and (best is None or o < best):
            best = o
    out["area_force_resid"] = best
    return out


def main():
    rows = []
    for s in SIZES:
        tm = pd.read_csv(f"{HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
        res = pd.read_csv(f"{HOME}/results_{s}mm.csv").set_index("trial")
        for _, r in tm[tm.status == "valid"].iterrows():
            trial = f"{HOME}/grasp_data/blueberry/size_{s}mm/{r.folder}"
            evf = f"{HOME}/shear_{s}mm/{r.folder}_events.json"
            if not (os.path.exists(f"{trial}/taxels.npz") and os.path.exists(evf)):
                print("skip", r.folder); continue
            ev = json.load(open(evf))
            hold, loosen, drop = float(ev["hold"]), float(ev["loosen"]), float(r.drop_camera_s)
            cam = float(res.loc[r.folder, "camera_onset_s"]) if r.folder in res.index else np.nan
            trans = float(res.loc[r.folder, "translation_onset_s"]) if r.folder in res.index else np.nan
            z = np.load(f"{trial}/taxels.npz")
            ser = [x for x in (finger_series(z, k, key, hold) for k, key in enumerate(("index", "thumb"))) if x is not None]
            if not ser:
                print("no contact", r.folder); continue
            on = onsets(ser, hold, loosen, loosen, drop)
            mid = (hold + loosen) / 2
            fa = onsets(ser, hold, mid, mid, loosen)
            row = dict(trial=r.folder, size=s, berry=r.berry_id, loosen=loosen, camera_onset=cam, drop=drop,
                       translation_onset=trans)
            for f in ("min_stick_frac", "area_force_resid"):
                o = on[f]
                row[f"{f}_onset"] = o
                row[f"{f}_minus_camera"] = None if o is None or not np.isfinite(cam) else o - cam
                row[f"{f}_minus_loosen"] = None if o is None else o - loosen
                row[f"{f}_warning"] = None if o is None else drop - o
                row[f"{f}_hold_FA"] = fa[f] is not None
            row["translation_minus_camera"] = trans - cam if np.isfinite(trans) and np.isfinite(cam) else None
            row["camera_minus_loosen"] = cam - loosen if np.isfinite(cam) else None
            rows.append(row)
            print(f"--- {r.folder} ({s} mm): stick {on['min_stick_frac']}, resid {on['area_force_resid']}, camera {cam}")
    D = pd.DataFrame(rows)
    out = f"{HOME}/claim1_results"
    os.makedirs(out, exist_ok=True)
    D.to_csv(f"{out}/claim1_per_trial.csv", index=False)
    print(f"\n=== {len(D)} valid trials, rule z > {Z} for {DUR} s, baseline = hold ===")
    print("camera onset - loosen start: median %.2f s (IQR %.2f..%.2f)" % tuple(
        D.camera_minus_loosen.astype(float).quantile([.5, .25, .75])))
    for f, lab in (("min_stick_frac", "min_stick_frac"), ("area_force_resid", "area/force residual"),
                   ("translation", "contact translation (ref.)")):
        if f == "translation":
            d = D.translation_minus_camera.dropna().astype(float)
            print(f"\n{lab}: n={len(d)}  onset - camera median {d.median():+.2f} s  "
                  f"tactile at/before camera {int((d <= 0).sum())}/{len(d)}")
            continue
        d = D[f"{f}_minus_camera"].dropna().astype(float)
        w = D[f"{f}_warning"].dropna().astype(float)
        lo = D[f"{f}_minus_loosen"].dropna().astype(float)
        print(f"\n{lab}: detected {D[f'{f}_onset'].notna().sum()}/{len(D)}, hold false alarms "
              f"{int(D[f'{f}_hold_FA'].sum())}/{len(D)}")
        if len(d):
            print(f"   onset - camera: median {d.median():+.2f} s (IQR {d.quantile(.25):+.2f}..{d.quantile(.75):+.2f}); "
                  f"tactile AT/BEFORE camera in {int((d <= 0).sum())}/{len(d)}")
            print(f"   onset - loosen start: median {lo.median():+.2f} s;  warning before drop: median {w.median():.2f} s")
        for s in SIZES:
            ds = D[(D["size"] == s)][f"{f}_minus_camera"].dropna().astype(float)
            if len(ds):
                print(f"   {s} mm: n={len(ds)} median {ds.median():+.2f} s, at/before camera {int((ds <= 0).sum())}/{len(ds)}")
    print(f"\nsaved {out}/claim1_per_trial.csv")


if __name__ == "__main__":
    main()
