#!/usr/bin/env python3
"""
lead_time_protocol.py
----------------------
Purpose: honestly test whether the Xense tactile signal changes BEFORE the
camera-visible slip/motion event, without the two biases that inflated
earlier results:

  Bias 1 (time-prior / frame-number leakage):
    Fixed by never using row-index/frame-number as a feature or as part of
    the detection logic. Everything here works in elapsed_s (physical time),
    and camera + Xense are read from the SAME trial's own event log, so a
    trial where the pull happens at a different time cannot be "predicted"
    just by knowing where it sits in the sequence.

  Bias 2 (multiple-comparison cherry-picking):
    Fixed by requiring ONE pre-registered feature (e.g. thumb_shear_N),
    chosen BEFORE looking at results, instead of taking the earliest of
    6 features (which biases the apparent lead time upward just from noise).

Usage
-----
Step 1 - find the camera motion-onset time for a trial (no pad tracking,
just frame-to-frame pixel change in a fixed ROI, robust to tracker jumps):

    python3 lead_time_protocol.py camera <trial_dir>/cam/cam1 \
        --roi-mode auto

    -> prints a candidate onset time; LOOK at the printed frame numbers /
       open a couple of frames around it to confirm by eye before trusting it.
       This script does NOT auto-decide the camera label for you - a human
       must confirm it, exactly like the professor wants (camera = ground truth).

Step 2 - once you have a confirmed camera onset time for each trial, put them
into a small CSV (trial, camera_t) and the trials' tactile_data.csv paths into
a config, then run:

    python3 lead_time_protocol.py report trials.csv \
        --feature thumb_shear_N --baseline 8 2

    -> outputs one row per trial: camera_t, xense_onset_t, lead_s, plus
       control trials (camera_t = NaN) report whether a false onset fired.

trials.csv format (one row per trial):
    name,data_dir,camera_t,is_control
    b3r1,/path/to/trial1/data,33.60,0
    ctrl1,/path/to/control1/data,,1
"""
import argparse, glob, csv, sys
import numpy as np
import pandas as pd
import cv2


def _load_frames(cam1_dir, timestamps_csv):
    """Returns list of (elapsed_s, filepath) sorted by elapsed_s.

    Two supported layouts:
    - exported zip style: cam1_dir/*.jpg with time embedded in the
      filename as '..._t<seconds>.jpg' (from export_trial*.py)
    - raw capture style: cam1_dir/frame_NNNNNN.png, with frame->time
      given separately in a camera_camX_timestamps.csv (columns
      frame, wall_time, elapsed_s, ...)
    """
    jpgs = sorted(glob.glob(f"{cam1_dir}/*.jpg"))
    if jpgs and timestamps_csv is None:
        out = []
        for f in jpgs:
            try:
                t = float(f.split("_t")[1][:-4])
            except Exception:
                continue
            out.append((t, f))
        return out

    pngs = glob.glob(f"{cam1_dir}/frame_*.png")
    if not pngs:
        sys.exit(f"[ERROR] no .jpg (with _t<secs> in filename) or "
                  f"frame_*.png files found in {cam1_dir}")
    if timestamps_csv is None:
        sys.exit(f"[ERROR] found frame_*.png files in {cam1_dir} but no "
                  "--timestamps CSV given. Pass the matching "
                  "camera_camX_timestamps.csv (it has frame,elapsed_s "
                  "columns) with --timestamps.")
    ts = pd.read_csv(timestamps_csv)
    ts = ts.set_index("frame")
    out = []
    for f in pngs:
        try:
            n = int(f.split("frame_")[1][:-4])
        except Exception:
            continue
        if n not in ts.index:
            continue
        out.append((float(ts.loc[n, "elapsed_s"]), f))
    out.sort(key=lambda r: r[0])
    if len(out) < 10:
        sys.exit(f"[ERROR] only matched {len(out)} saved frames to the "
                  "timestamps CSV - check --timestamps points at the right "
                  "camera_camX_timestamps.csv for this trial.")
    return out


def camera_onset(cam1_dir, diff_thresh_frac=0.08, consec=3, roi=None,
                  pixel_delta=25, timestamps_csv=None, skip=1):
    """Frame-differencing motion onset: robust alternative to template/pad
    tracking (which was shown to jump +/-45-50px on single frames in earlier
    trials). Flags the first point of SUSTAINED frame-to-frame change,
    after a quiet baseline, inside an ROI you give (default: central box,
    to exclude background/specular-glare flicker near the frame edges) -
    good for a human to then go confirm visually.

    IMPORTANT (known limitation): the Hikrobot frames have a bright
    specular highlight on the berry that shifts a little even when nothing
    actually moves, which creates real (not fake) frame-to-frame pixel
    change. This detector can still pick up that flicker as "motion" in a
    busy scene. Treat its output as a CANDIDATE only - always open a few
    frames around it and confirm by eye before using it as camera_t."""
    frames = _load_frames(cam1_dir, timestamps_csv)
    if skip > 1:
        frames = frames[::skip]
        print(f"[INFO] using every {skip}th saved frame "
              f"({len(frames)} frames after skipping) for speed")
    ts, diffs = [], []
    prev = None
    for t, f in frames:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if roi:
            x0, y0, x1, y1 = roi
            g = g[y0:y1, x0:x1]
        else:
            h, w = g.shape
            g = g[int(0.15 * h):int(0.85 * h), int(0.2 * w):int(0.8 * w)]
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if prev is not None:
            d = np.abs(g.astype(int) - prev.astype(int))
            diffs.append(float((d > pixel_delta).sum()))
            ts.append(t)
        prev = g
    diffs = np.array(diffs)
    ts = np.array(ts)
    if len(diffs) < 10:
        sys.exit("[ERROR] too few frames to estimate motion")

    # Robust baseline over the WHOLE trial (median/MAD), not just the first
    # frames - the first frames often still contain placement/settle motion,
    # which would make a naive "first N frames" baseline too noisy.
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med)) * 1.4826  # -> comparable to a std
    mad = max(mad, 1.0)
    thresh = med + max(8 * mad, diff_thresh_frac * diffs.max())

    onset_i = None
    for i in range(len(diffs) - consec):
        if all(diffs[i + k] > thresh for k in range(consec)):
            onset_i = i
            break

    print(f"[INFO] {cam1_dir}: {len(frames)} frames, ROI={'custom' if roi else 'auto-center'}, "
          f"baseline motion median={med:.0f}px (MAD~{mad:.0f}), threshold={thresh:.0f}")

    # KNOWN LIMITATION: the specular highlight on the berry flickers even when
    # nothing physically moves, so the "auto" threshold above is not reliable
    # on its own - treat it only as a hint. Always print the actual motion
    # curve so a human can see the real pattern and pick the onset by eye,
    # the way the professor wants (camera = ground truth, human-verified).
    print("\n[MOTION CURVE] elapsed_s : changed-pixel count (look for a "
          "SUSTAINED step up, not a single-frame spike - glare causes spikes)")
    step = max(1, len(ts) // 60)
    for i in range(0, len(ts), step):
        bar = "#" * int(40 * diffs[i] / max(diffs.max(), 1))
        marker = " <-- auto candidate" if onset_i is not None and i == onset_i else ""
        print(f"  {ts[i]:7.2f} : {diffs[i]:9.0f} {bar}{marker}")

    if onset_i is None:
        print("\n[RESULT] auto-threshold found nothing sustained. Read the "
              "curve above yourself and pick the onset time where the level "
              "steps up and STAYS up for several rows, not a single spike.")
        return None
    t_onset = ts[onset_i]
    print(f"\n[RESULT] auto candidate onset at elapsed_s = {t_onset:.3f} "
          "- this is a HINT, not a verdict.")
    print("[ACTION] compare it against the curve above and against a few "
          "actual frames around that timestamp before using it as camera_t "
          "in trials.csv.")
    return t_onset


def xense_onset(tactile_csv, feature, baseline_pre, baseline_post, after_t=None):
    """Single pre-registered feature, 3-sigma / 3-consecutive-sample rule,
    baseline taken from a quiet window the caller specifies (e.g. 8-2s
    before the expected event, or an explicit absolute window for a long
    constant hold). Returns onset elapsed_s or None."""
    d = pd.read_csv(tactile_csv)
    if feature not in d.columns:
        sys.exit(f"[ERROR] feature '{feature}' not in {tactile_csv}. "
                  f"Available: {list(d.columns)}")

    if after_t is None:
        after_t = d.elapsed_s.min()

    base = d[(d.elapsed_s >= baseline_pre) & (d.elapsed_s <= baseline_post)]
    if len(base) < 5:
        sys.exit(f"[ERROR] baseline window [{baseline_pre},{baseline_post}] "
                  f"has only {len(base)} samples - widen it.")
    mu, sd = base[feature].mean(), base[feature].std()
    floor = {"force_N": 0.1, "shear_N": 0.1, "contact_area": 20}
    for k, v in floor.items():
        if feature.endswith(k):
            sd = max(sd, v)
    x = d[feature].values
    t = d.elapsed_s.values
    dev = np.abs(x - mu) > 3 * sd

    onset = None
    for i in range(len(x) - 2):
        if t[i] < baseline_post:
            continue
        if dev[i] and dev[i + 1] and dev[i + 2]:
            onset = t[i]
            break
    return onset, mu, sd


def cmd_camera(args):
    roi = tuple(args.roi) if args.roi else None
    camera_onset(args.cam1_dir, diff_thresh_frac=args.diff_thresh_frac,
                 roi=roi, pixel_delta=args.pixel_delta,
                 timestamps_csv=args.timestamps, skip=args.skip)


def cmd_report(args):
    rows = list(csv.DictReader(open(args.trials_csv)))
    out = []
    for r in rows:
        name = r["name"]
        data_dir = r["data_dir"].rstrip("/")
        tactile_csv = f"{data_dir}/tactile_data.csv"
        is_control = r.get("is_control", "0") == "1"
        camera_t = float(r["camera_t"]) if r.get("camera_t") else None

        if camera_t is not None:
            b_pre, b_post = camera_t - args.baseline[0], camera_t - args.baseline[1]
        else:
            # control trial: no camera event, use the explicit baseline window
            # given via --control-baseline (defaults to --baseline interpreted
            # as absolute seconds into the hold)
            b_pre, b_post = args.control_baseline

        onset, mu, sd = xense_onset(tactile_csv, args.feature, b_pre, b_post)
        lead = None if (onset is None or camera_t is None) else camera_t - onset

        out.append(dict(
            name=name, is_control=is_control, camera_t=camera_t,
            xense_onset_t=onset, lead_s=lead, baseline_mean=round(mu, 3),
            baseline_sd=round(sd, 3),
        ))

    print(f"\nFeature (pre-registered, fixed for ALL trials): {args.feature}")
    def fmt(v):
        return "" if v is None else f"{v:.2f}"

    print(f"{'trial':12s} {'control':8s} {'camera_t':>9s} {'xense_t':>9s} {'lead_s':>8s}")
    for o in out:
        print(f"{o['name']:12s} {str(o['is_control']):8s} "
              f"{fmt(o['camera_t']):>9s} "
              f"{fmt(o['xense_onset_t']):>9s} "
              f"{fmt(o['lead_s']):>8s}")

    real = [o for o in out if not o["is_control"]]
    ctrl = [o for o in out if o["is_control"]]
    leads = [o["lead_s"] for o in real if o["lead_s"] is not None]
    fa = [o for o in ctrl if o["xense_onset_t"] is not None]

    print("\n--- Verdict ---")
    if leads:
        print(f"Pull trials: {len(leads)}/{len(real)} gave a lead; "
              f"mean={np.mean(leads):.2f}s sd={np.std(leads):.2f}s "
              f"range=[{min(leads):.2f},{max(leads):.2f}]s")
    else:
        print("No pull trial produced a measurable lead with this feature.")
    if ctrl:
        print(f"Control trials: {len(fa)}/{len(ctrl)} fired a false onset "
              "(should be 0 for the claim to hold).")
    if leads and np.std(leads) > 0.6 * np.mean(leads):
        print("WARNING: lead time is NOT consistent across trials "
              "(sd is large relative to mean) - do not claim a reliable "
              "lead time yet, collect more trials.")
    if fa:
        print("WARNING: at least one control trial fired a false onset on "
              "this feature - the signal is not slip-specific, it reacts to "
              "something else (grip drift, sensor noise). Report this "
              "honestly; do not drop the control trial from the table.")

    if args.out_csv:
        pd.DataFrame(out).to_csv(args.out_csv, index=False)
        print(f"\n[SAVED] {args.out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("camera", help="find a candidate camera motion-onset time")
    pc.add_argument("cam1_dir")
    pc.add_argument("--diff-thresh-frac", type=float, default=0.08)
    pc.add_argument("--pixel-delta", type=int, default=25,
                     help="min per-pixel gray-level change to count as 'changed' (default 25)")
    pc.add_argument("--roi", type=int, nargs=4, default=None,
                     metavar=("X0", "Y0", "X1", "Y1"),
                     help="crop box in pixels (default: central 60%% x 70%% "
                          "of frame, to avoid edge/background glare flicker)")
    pc.add_argument("--skip", type=int, default=1,
                     help="use every Nth saved frame instead of all of "
                          "them (default 1 = all). Full-res PNG trials can "
                          "have thousands of saved frames and take minutes "
                          "to decode - try --skip 5 for a fast first look, "
                          "then re-run with --skip 1 once you know roughly "
                          "where the onset is.")
    pc.add_argument("--timestamps", default=None,
                     help="camera_camX_timestamps.csv for this trial - "
                          "REQUIRED when cam1_dir holds raw frame_NNNNNN.png "
                          "files (the default capture layout). Not needed "
                          "for an already-exported cam1 folder with "
                          "'..._t<secs>.jpg' filenames.")
    pc.set_defaults(func=cmd_camera)

    pr = sub.add_parser("report", help="combine trials into a lead-time table")
    pr.add_argument("trials_csv")
    pr.add_argument("--feature", required=True,
                     help="pre-register ONE feature, e.g. thumb_shear_N. "
                          "Decide this before looking at any results.")
    pr.add_argument("--baseline", type=float, nargs=2, default=[8, 2],
                     metavar=("SEC_BEFORE_EVENT", "SEC_BEFORE_EVENT"),
                     help="for pull trials: baseline window = "
                          "[camera_t - arg0, camera_t - arg1]")
    pr.add_argument("--control-baseline", type=float, nargs=2,
                     default=[5, 15],
                     metavar=("T0", "T1"),
                     help="for control trials (no camera_t): absolute "
                          "elapsed_s window [T0,T1] used as baseline, and "
                          "onset search starts after T1")
    pr.add_argument("--out-csv", default=None)
    pr.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)
