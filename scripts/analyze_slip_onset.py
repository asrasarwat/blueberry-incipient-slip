import os
#!/usr/bin/env python3
"""
analyze_slip_onset.py  -  does the Xense signal change at / before the camera-defined slip onset?

The two onsets are found INDEPENDENTLY:
  * camera onset : sub-pixel berry displacement (phase correlation on the berry ROI) relative to a
                   baseline frame; onset = first sustained excursion above the baseline noise.
                   No tactile information is used.
  * Xense onset  : first sustained excursion of each tactile feature above its own baseline noise,
                   on the chosen sensor only, with frozen (stale) samples removed.
Lag = Xense onset - camera onset   (negative = Xense changes BEFORE the camera sees movement).

Usage
  python3 analyze_slip_onset.py TRIAL_DIR [--cam cam2] [--sensor thumb] [--baseline T0 T1]
                                [--roi X Y W H] [--stride 4] [--k 4]
TRIAL_DIR must contain tactile_data.csv, camera_<cam>_timestamps.csv and either
camera_<cam>/frame_XXXXXX.png or <cam>_frame_XXXXXX.png. trial_info.json is optional.

Outputs (in TRIAL_DIR/onset_analysis/): onset_summary.json, camera_displacement.csv, onset_figure.png
No verdict text is hard-coded: everything printed comes from the data.
"""
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STALE_RUN = 4          # identical (force, area, shear) samples in a row -> frozen sensor
MAX_STALE_FRAC = 0.2   # warn if more of the sensor's samples than this are frozen


# ------------------------------------------------------------------ generic onset detector
def detect_onset(t, y, base_mask, search_from, k=4.0, min_abs=0.0, sustain=2):
    """First time >= search_from where |y - baseline mean| > max(k*std, min_abs) for `sustain`
    consecutive valid samples. NaNs (stale samples) are skipped. Returns (t_onset or None, info)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    b = base_mask & ok
    if b.sum() < 3:
        return None, {"error": f"baseline has only {int(b.sum())} valid samples"}
    mu, sd = float(y[b].mean()), float(y[b].std())
    thr = max(k * sd, min_abs)
    idx = np.where(ok & (t >= search_from))[0]
    exc = np.abs(y[idx] - mu) > thr
    for i in range(len(idx) - sustain + 1):
        if exc[i:i + sustain].all():
            j = idx[i]
            return float(t[j]), {"baseline_mean": mu, "baseline_std": sd, "threshold": thr,
                                 "direction": "up" if y[j] > mu else "down", "n_baseline": int(b.sum())}
    return None, {"baseline_mean": mu, "baseline_std": sd, "threshold": thr, "n_baseline": int(b.sum()),
                  "note": "no sustained excursion found"}


# ------------------------------------------------------------------ tactile
def mark_stale(df, sensor):
    """NaN-mask samples that belong to a frozen run (identical force, area and shear)."""
    key = df[[f"{sensor}_force_N", f"{sensor}_contact_area", f"{sensor}_shear_N"]].round(6).apply(tuple, axis=1)
    run = (key != key.shift()).cumsum()
    run_len = key.groupby(run).transform("size")
    return (run_len >= STALE_RUN).to_numpy()


def tactile_features(df, sensor):
    stale = mark_stale(df, sensor)
    feats = {"Fz": df[f"{sensor}_force_N"].abs().to_numpy(float),
             "contact_area": df[f"{sensor}_contact_area"].to_numpy(float),
             "shear_sum": df[f"{sensor}_shear_N"].to_numpy(float)}
    if {f"{sensor}_Fx", f"{sensor}_Fy", f"{sensor}_Fz"} <= set(df.columns):
        fx, fy, fz = (df[f"{sensor}_{c}"].to_numpy(float) for c in ("Fx", "Fy", "Fz"))
        feats["tangential_over_normal"] = np.hypot(fx, fy) / np.maximum(np.abs(fz), 0.05)   # slip-specific
        feats["Fxy"] = np.hypot(fx, fy)
    if {f"{sensor}_cop_row", f"{sensor}_cop_col"} <= set(df.columns):
        r, c = df[f"{sensor}_cop_row"].to_numpy(float), df[f"{sensor}_cop_col"].to_numpy(float)
        feats["cop_row"], feats["cop_col"] = r, c                                           # slip-specific
    for name in feats:
        feats[name] = np.where(stale, np.nan, feats[name])
    return feats, stale


# ------------------------------------------------------------------ camera
def find_frames(trial, cam):
    pats = [os.path.join(trial, f"camera_{cam}", "frame_*.png"), os.path.join(trial, f"{cam}_frame_*.png")]
    for p in pats:
        files = sorted(glob.glob(p))
        if files:
            return {int(re.search(r"(\d+)\.png$", f).group(1)): f for f in files}
    sys.exit(f"no frames found for {cam} in {trial}")


def auto_roi(gray, pad=60):
    ys, xs = np.where(gray > max(150, np.percentile(gray, 99.5)))
    if len(xs) < 50:
        h, w = gray.shape
        return w // 4, h // 4, w // 2, h // 2
    x0, x1, y0, y1 = np.percentile(xs, 2), np.percentile(xs, 98), np.percentile(ys, 2), np.percentile(ys, 98)
    x, y = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    return x, y, int(min(gray.shape[1], x1 + pad) - x), int(min(gray.shape[0], y1 + pad) - y)


def camera_displacement(files, ts, roi_arg, base_win, stride):
    nums = sorted(files)[::stride]
    t_of = dict(zip(ts["frame"], ts["elapsed_s"]))
    nums = [n for n in nums if n in t_of]
    times = np.array([t_of[n] for n in nums])
    base_idx = np.where((times >= base_win[0]) & (times <= base_win[1]))[0]
    if len(base_idx) < 3:
        sys.exit(f"fewer than 3 camera frames in baseline window {base_win}; check --baseline/--stride")
    ref_n = nums[base_idx[len(base_idx) // 2]]
    ref = cv2.cvtColor(cv2.imread(files[ref_n]), cv2.COLOR_BGR2GRAY)
    roi = tuple(roi_arg) if roi_arg else auto_roi(ref)
    x, y, w, h = roi
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    prep = lambda g: cv2.GaussianBlur(g[y:y + h, x:x + w].astype(np.float32), (0, 0), 1.5)
    ref_p = prep(ref)
    out = []
    for n, tt in zip(nums, times):
        g = cv2.imread(files[n], cv2.IMREAD_GRAYSCALE)
        (dx, dy), resp = cv2.phaseCorrelate(ref_p, prep(g), win)
        out.append((n, tt, dx, dy, float(np.hypot(dx, dy)), resp))
    return pd.DataFrame(out, columns=["frame", "elapsed_s", "dx_px", "dy_px", "disp_px", "response"]), roi, ref_n


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--cam", default="cam2")
    ap.add_argument("--sensor", default="thumb", choices=["thumb", "index"])
    ap.add_argument("--baseline", type=float, nargs=2, default=None, help="baseline window in trial seconds")
    ap.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--k", type=float, default=4.0, help="threshold = k * baseline std")
    ap.add_argument("--min-response", type=float, default=0.1, help="reject camera frames whose phase-correlation peak is below this")
    ap.add_argument("--min-px", type=float, default=0.3, help="minimum camera displacement threshold (px)")
    a = ap.parse_args()

    warnings = []
    info = {}
    ip = os.path.join(a.trial, "trial_info.json")
    if os.path.exists(ip):
        info = json.load(open(ip))
    loosen_start = info.get("loosen_start_elapsed_s")
    if a.baseline:
        base = tuple(a.baseline)
    elif loosen_start:
        base = (max(0.0, loosen_start - 1.5), loosen_start)
    else:
        base = (0.0, 0.5)
        warnings.append("no loosening start known: baseline = first 0.5 s, which may already include loosening/unloading")
    search_from = base[1]

    tac = pd.read_csv(os.path.join(a.trial, "tactile_data.csv"))
    ts = pd.read_csv(os.path.join(a.trial, f"camera_{a.cam}_timestamps.csv"))
    t = tac["elapsed_s"].to_numpy(float)
    base_mask_t = (t >= base[0]) & (t <= base[1])

    # ---- camera onset (independent of tactile)
    files = find_frames(a.trial, a.cam)
    disp, roi, ref_n = camera_displacement(files, ts, a.roi, base, a.stride)
    bad = (disp.response < a.min_response).to_numpy()
    disp.loc[bad, "disp_px"] = np.nan                      # alignment failed -> do not trust the shift
    if bad.mean() > 0.2:
        warnings.append(f"{bad.mean():.0%} of camera frames rejected (phase-correlation response < {a.min_response}); "
                        f"check --roi / lighting before trusting the camera onset")
    bm = ((disp.elapsed_s >= base[0]) & (disp.elapsed_s <= base[1])).to_numpy()
    cam_t, cam_info = detect_onset(disp.elapsed_s, disp.disp_px, bm, search_from, k=a.k, min_abs=a.min_px, sustain=5)

    # ---- Xense onsets
    feats, stale = tactile_features(tac, a.sensor)
    stale_frac = float(stale.mean())
    if stale_frac > MAX_STALE_FRAC:
        warnings.append(f"{a.sensor} sensor: {stale_frac:.0%} of samples are frozen (identical runs >= {STALE_RUN}); results unreliable")
    if len(t) > 1 and np.median(np.diff(t)) > 0.05:
        warnings.append(f"tactile rate is only ~{1/np.median(np.diff(t)):.1f} Hz (median interval {np.median(np.diff(t)):.3f} s); "
                        f"onset resolution is limited to that interval")
    slip_specific = {"tangential_over_normal", "cop_row", "cop_col", "Fxy", "shear_sum"}
    results = {}
    for name, y in feats.items():
        ot, oi = detect_onset(t, y, base_mask_t, search_from, k=a.k)
        oi["onset_s"] = ot
        oi["lag_vs_camera_s"] = None if (ot is None or cam_t is None) else round(ot - cam_t, 3)
        oi["type"] = "slip-sensitive" if name in slip_specific else "unloading (also changes when fingers just open)"
        results[name] = oi

    summary = {"trial": os.path.abspath(a.trial), "camera": a.cam, "sensor": a.sensor, "baseline_s": base,
               "camera_onset_s": cam_t, "camera_info": cam_info, "roi": roi, "ref_frame": ref_n,
               "stale_fraction": round(stale_frac, 3), "xense": results, "warnings": warnings}
    out = os.path.join(a.trial, "onset_analysis"); os.makedirs(out, exist_ok=True)
    json.dump(summary, open(os.path.join(out, "onset_summary.json"), "w"), indent=2, default=float)
    disp.to_csv(os.path.join(out, "camera_displacement.csv"), index=False)

    # ---- figure
    names = list(feats)
    fig, ax = plt.subplots(2 + len(names), 1, figsize=(11, 2.2 * (2 + len(names))), sharex=True)
    ax[0].plot(disp.elapsed_s, disp.disp_px, "k-"); ax[0].set_ylabel("berry disp (px)")
    if cam_t is not None:
        ax[0].axhline(cam_info["threshold"] + cam_info["baseline_mean"], color="gray", ls=":")
    ax[1].plot(disp.elapsed_s, disp.response, "gray"); ax[1].set_ylabel("PC response")
    for i, name in enumerate(names, start=2):
        y = feats[name]
        ax[i].plot(t, y, "o-", ms=3, color="tab:red"); ax[i].set_ylabel(name, fontsize=8)
        o = results[name]["onset_s"]
        if o is not None:
            ax[i].axvline(o, color="tab:red", ls="--")
    for axx in ax:
        axx.axvspan(base[0], base[1], color="tab:blue", alpha=0.08)
        if cam_t is not None:
            axx.axvline(cam_t, color="k", ls="--")
    ax[-1].set_xlabel("trial time (s)   blue = baseline, black dashed = camera onset, red dashed = Xense feature onset")
    fig.suptitle(f"{os.path.basename(os.path.normpath(a.trial))}  {a.sensor}  (stale samples omitted)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "onset_figure.png"), dpi=150); plt.close(fig)

    # ---- printed result (all from data)
    print(f"baseline {base[0]:.2f}-{base[1]:.2f} s | ROI {roi} | camera onset: "
          + (f"{cam_t:.3f} s" if cam_t is not None else "not found"))
    print(f"{a.sensor} stale fraction {stale_frac:.0%}")
    print(f"{'feature':26s}{'onset (s)':>10s}{'lag vs cam (s)':>16s}  type")
    for n, r in results.items():
        print(f"{n:26s}{('%.3f' % r['onset_s']) if r['onset_s'] is not None else '-':>10s}"
              f"{('%+.3f' % r['lag_vs_camera_s']) if r['lag_vs_camera_s'] is not None else '-':>16s}  {r['type']}")
    for w in warnings:
        print("WARNING:", w)
    print("outputs:", out)


if __name__ == "__main__":
    main()
