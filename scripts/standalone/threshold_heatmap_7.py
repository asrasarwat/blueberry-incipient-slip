#!/usr/bin/env python3
"""threshold_heatmap.py - onset-rule sensitivity (z threshold x duration) for edge lag, translation, force drop."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
ZS = [2, 3, 4, 5, 6, 8, 10, 15]
DURS = [0.05, 0.1, 0.25, 0.5, 1.0]
PRE = (5, 0.25)
SIGNALS = {"edge lag": "lag", "contact translation": "trans", "normal-force drop": "fz"}


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


def zseries(d, sig, b0, b1):
    if sig == "lag":
        x = d.edge_lag_px.values
    elif sig == "trans":
        x = np.hypot(d.all_dx.values, d.all_dy.values)
    else:
        x = -d.fz_N.values
    t = d.t_s.values
    m = (t >= b0) & (t < b1) & np.isfinite(x)
    if m.sum() < 10:
        return None
    mu, sd = x[m].mean(), max(x[m].std(), 1e-6)
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0
    return t, z


def onset(D, sig, b0, b1, t_from, t_to, zthr, dur):
    best = None
    for fin, d in D.groupby("finger"):
        r = zseries(d, sig, b0, b1)
        if r is None:
            continue
        o = first_sustained(r[0], r[1] > zthr, t_from, t_to, dur)
        if o is not None and (best is None or o < best):
            best = o
    return best


def main():
    trials = []
    for s in SIZES:
        tm = pd.read_csv(f"{HOME}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
        res = pd.read_csv(f"{HOME}/results_{s}mm.csv").set_index("trial")
        for _, r in tm[tm.status == "valid"].iterrows():
            f = f"{HOME}/shear_{s}mm/{r.folder}_shear.csv"
            e = f"{HOME}/shear_{s}mm/{r.folder}_events.json"
            if not (os.path.exists(f) and os.path.exists(e)):
                print("skip", r.folder); continue
            ev = json.load(open(e))
            cam = res.loc[r.folder, "camera_onset_s"] if r.folder in res.index else np.nan
            trials.append(dict(name=r.folder, size=s, D=pd.read_csv(f).sort_values("t_s"), hold=float(ev["hold"]),
                               loosen=float(ev["loosen"]), drop=float(r.drop_camera_s), cam=float(cam)))
    print(f"{len(trials)} valid trials")
    rows = []
    for label, sig in SIGNALS.items():
        for zt in ZS:
            for du in DURS:
                det, fa, warn, lead = 0, 0, [], []
                for T in trials:
                    mid = (T["hold"] + T["loosen"]) / 2
                    if onset(T["D"], sig, T["hold"], mid, mid, T["loosen"], zt, du) is not None:
                        fa += 1
                    o = onset(T["D"], sig, T["hold"], T["loosen"], T["loosen"], T["drop"], zt, du)
                    if o is not None:
                        det += 1; warn.append(T["drop"] - o)
                        if np.isfinite(T["cam"]):
                            lead.append(T["cam"] - o)
                n = len(trials)
                rows.append(dict(signal=label, z=zt, dur_s=du, detected_pct=100 * det / n, hold_fa_pct=100 * fa / n,
                                 warn_median_s=np.median(warn) if warn else np.nan,
                                 vs_camera_median_s=np.median(lead) if lead else np.nan, n=n))
        print("done:", label)
    G = pd.DataFrame(rows)
    out = f"{HOME}/heatmap_results_7"
    os.makedirs(out, exist_ok=True)
    G.round(3).to_csv(f"{out}/threshold_grid.csv", index=False)
    metrics = [("detected_pct", "detected before drop (%)", "Greens", (0, 100)),
               ("hold_fa_pct", "false alarm in hold (%)", "Reds", (0, 100)),
               ("warn_median_s", "median warning before drop (s)", "Blues", None),
               ("vs_camera_median_s", "camera - tactile onset (s)\n(+ = tactile first)", "RdBu", "sym")]
    fig, axes = plt.subplots(len(SIGNALS), len(metrics), figsize=(18, 4.2 * len(SIGNALS)), squeeze=False)
    for i, label in enumerate(SIGNALS):
        g = G[G.signal == label]
        for j, (col, title, cmap, rng) in enumerate(metrics):
            ax = axes[i, j]
            M = g.pivot(index="z", columns="dur_s", values=col).loc[ZS, DURS].values
            if rng == "sym":
                v = np.nanmax(np.abs(G[col].values)); vmin, vmax = -v, v
            elif rng is None:
                vmin, vmax = np.nanmin(G[col].values), np.nanmax(G[col].values)
            else:
                vmin, vmax = rng
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            for a in range(len(ZS)):
                for b in range(len(DURS)):
                    v = M[a, b]
                    txt = "-" if not np.isfinite(v) else (f"{v:.0f}" if "pct" in col else f"{v:.1f}")
                    rgb = im.cmap(im.norm(v))[:3] if np.isfinite(v) else (1, 1, 1)
                    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    ax.text(b, a, txt, ha="center", va="center", fontsize=8, color="white" if lum < 0.5 else "black")
            pi, pj = ZS.index(PRE[0]), DURS.index(PRE[1])
            ax.add_patch(plt.Rectangle((pj - 0.5, pi - 0.5), 1, 1, fill=False, ec="black", lw=2.5))
            ax.set_xticks(range(len(DURS))); ax.set_xticklabels([f"{d:g}" for d in DURS])
            ax.set_yticks(range(len(ZS))); ax.set_yticklabels(ZS)
            ax.set_xlabel("min. duration above threshold (s)")
            if j == 0:
                ax.set_ylabel(f"{label}\nz threshold")
            if i == 0:
                ax.set_title(title, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Onset-rule sensitivity, {len(trials)} valid trials, 7 sizes (14-21 mm). "
                 f"Box = pre-registered rule (z > {PRE[0]}, {PRE[1]} s)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out}/threshold_heatmap.png", dpi=150)
    pre = G[(G.z == PRE[0]) & (G.dur_s == PRE[1])]
    print("\npre-registered rule (z>5, 0.25 s):")
    print(pre[["signal", "detected_pct", "hold_fa_pct", "warn_median_s", "vs_camera_median_s"]].round(2).to_string(index=False))
    best = G[G.hold_fa_pct == 0].sort_values(["signal", "detected_pct", "warn_median_s"], ascending=[True, False, False])
    print("\nbest rule per signal with 0 hold false alarms:")
    print(best.groupby("signal").head(1)[["signal", "z", "dur_s", "detected_pct", "warn_median_s", "vs_camera_median_s"]]
          .round(2).to_string(index=False))
    print(f"\nsaved {out}/threshold_heatmap.png and threshold_grid.csv")


if __name__ == "__main__":
    main()
