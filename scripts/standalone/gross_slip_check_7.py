#!/usr/bin/env python3
"""gross_slip_check.py - pre-registered 4 Nov 2026: min_stick_frac onset vs camera gross slip (>= 0.3 mm primary;
0.1/0.2/0.3/0.5 mm all reported). Negative = tactile BEFORE camera gross slip."""
import os, argparse
import numpy as np, pandas as pd

HOME = os.path.expanduser("~")
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
MM = [0.1, 0.2, 0.3, 0.5]
PRIMARY = 0.3
DUR = 0.25
SCORE_MIN = 0.5


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


def rel_displacement(tr, loosen):
    t = tr.t_s.values
    fx, fy = [], []
    for p in ("topL", "botL", "topR", "botR"):
        ok = tr[f"{p}_score"].fillna(0).values >= SCORE_MIN
        if ok.mean() < 0.5:
            continue
        fx.append(np.where(ok, tr[f"{p}_dx"].values, np.nan))
        fy.append(np.where(ok, tr[f"{p}_dy"].values, np.nan))
    if not fx:
        return None, None, "no fingertip patch"
    finx, finy = np.nanmean(fx, 0), np.nanmean(fy, 0)
    bok = tr.berry_score.fillna(0).values >= SCORE_MIN
    rx = np.where(bok, tr.berry_dx.values - finx, np.nan)
    ry = np.where(bok, tr.berry_dy.values - finy, np.nan)
    base = (t >= loosen - 0.5) & (t < loosen)
    if np.isfinite(rx[base]).sum() < 5:
        base = t < loosen
    rx = rx - np.nanmedian(rx[base]); ry = ry - np.nanmedian(ry[base])
    d = pd.Series(np.hypot(rx, ry), index=t)
    dt = np.median(np.diff(t)) if len(t) > 2 else 0.01
    n = max(3, int(round(DUR / dt)) | 1)
    d = d.rolling(n, center=False, min_periods=max(2, n // 2)).median()
    return t, d.values, f"{len(fx)} fingertip patches"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px-per-mm", type=float, default=29.45)
    a = ap.parse_args()
    c1 = pd.read_csv(f"{HOME}/claim1_results/claim1_per_trial.csv").set_index("trial")
    rows = []
    for s in SIZES:
        for _, r in c1[c1["size"].astype(str).str.startswith(s.split('.')[0])].iterrows():
            f = f"{HOME}/track_{s}mm/{r.name}_track.csv"
            if not os.path.exists(f):
                print("no track", r.name); continue
            tr = pd.read_csv(f)
            t, d, note = rel_displacement(tr, r['loosen'])
            row = dict(trial=r.name, size=s, loosen=r['loosen'], drop=r['drop'], note=note,
                       stick_onset=r.min_stick_frac_onset, translation_onset=r.translation_onset)
            if t is not None:
                dmm = d / a.px_per_mm
                pre = dmm[(t > r['loosen']) & (t < r['drop'] - 0.3)]
                row["max_mm_before_drop"] = float(np.nanmax(pre)) if np.isfinite(pre).any() else np.nan
                for mm in MM:
                    o = first_sustained(t, np.nan_to_num(dmm) >= mm, r['loosen'], r['drop'], DUR)
                    row[f"cam_{mm}mm"] = o
                    for k in ("stick", "translation"):
                        on = row[f"{k}_onset"]
                        row[f"{k}_minus_cam_{mm}mm"] = (on - o) if (o is not None and pd.notna(on)) else np.nan
            rows.append(row)
    D = pd.DataFrame(rows)
    out = f"{HOME}/claim1_results"
    D.to_csv(f"{out}/gross_slip_per_trial.csv", index=False)
    print(f"\n=== {len(D)} trials, scale {a.px_per_mm} px/mm; negative = tactile BEFORE camera gross slip ===")
    print("berry-vs-fingertip motion before the drop: median max %.2f mm (IQR %.2f..%.2f)" %
          tuple(D.max_mm_before_drop.quantile([.5, .25, .75])))
    print(f"{'camera rule':14s} {'cam found':>9s} | {'min_stick_frac: median':>24s} {'first':>8s} | "
          f"{'translation: median':>20s} {'first':>8s}")
    for mm in MM:
        found = D[f"cam_{mm}mm"].notna().sum()
        line = f"{('>= %.1f mm' % mm) + (' *' if mm == PRIMARY else ''):14s} {found:5d}/{len(D):<3d} |"
        for k in ("stick", "translation"):
            x = D[f"{k}_minus_cam_{mm}mm"].dropna()
            line += f" {x.median():+21.2f} s {int((x <= 0).sum()):3d}/{len(x):<3d} |" if len(x) else f" {'-':>23s} {'-':>8s} |"
        print(line)
    print("  * = pre-registered primary rule")
    x = D[f"stick_minus_cam_{PRIMARY}mm"].dropna()
    if len(x):
        print(f"\nPRIMARY ({PRIMARY} mm): min_stick_frac before gross slip in {int((x <= 0).sum())}/{len(x)} trials, "
              f"median {x.median():+.2f} s (IQR {x.quantile(.25):+.2f}..{x.quantile(.75):+.2f})")
        for s in SIZES:
            xs = D[D["size"] == s][f"stick_minus_cam_{PRIMARY}mm"].dropna()
            if len(xs):
                print(f"   {s} mm: median {xs.median():+.2f} s, tactile first {int((xs <= 0).sum())}/{len(xs)}")
    print(f"\nsaved {out}/gross_slip_per_trial.csv")


if __name__ == "__main__":
    main()
