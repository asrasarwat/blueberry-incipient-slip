#!/usr/bin/env python3
"""
berry_rotation.py - in-plane rotation of the berry in camera 2 over time (degrees, + = counter-clockwise on
screen), using the berry positions already tracked by track_camera_v3.py.

Why: during loosening the two fingertips move sideways in opposite directions. A berry ROLLING between them
turns in place (no sliding at the contacts, little gel shear); a berry SLIDING does not need to turn. Sliding at
each contact = (berry - fingertip) sideways motion minus the part explained by rolling (radius x angle).

For each frame it rotates the hold-reference marker patch in small steps around the previous angle and keeps the
best normalised cross-correlation (parabola-refined); when the match weakens the reference is re-cut from the
current frame and the angle carried over.

Per trial it writes <out>/<name>_rot.csv (t_s, rot_deg, score) and zips <out>.zip.

Usage (trials need roi.json; --track is the folder track_camera_v3.py wrote):
  python3 berry_rotation.py <trial> [<trial> ...] --track ~/track_16mm --out ~/rot_16mm
"""
import argparse
import csv
import os
import time
import zipfile

import cv2
import numpy as np

import analyze_slip as A
import track_camera_v3 as T

STEP = 0.5          # deg
SPAN = 3.0          # deg searched either side of the previous angle
SHIFT = 10          # px of translation slack around the tracked position
INNER = 0.70        # central part of the rotated patch used (avoids rotated borders)
REANCHOR = 0.70
MIN_SCORE = 0.35


def rot_patch(img, cx, cy, w, h, ang):
    """Central INNER part of the w x h patch centred at (cx, cy), rotated by ang (deg, + = counter-clockwise)."""
    r = int(0.75 * max(w, h)) + 4
    x0, y0 = int(round(cx)) - r, int(round(cy)) - r
    crop = img[max(0, y0):y0 + 2 * r, max(0, x0):x0 + 2 * r]
    ccx, ccy = cx - max(0, x0), cy - max(0, y0)
    M = cv2.getRotationMatrix2D((float(ccx), float(ccy)), float(ang), 1.0)
    rc = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR)
    iw, ih = int(w * INNER), int(h * INNER)
    px, py = int(round(ccx - iw / 2)), int(round(ccy - ih / 2))
    return rc[py:py + ih, px:px + iw]


def score_at(img, tpl, cx, cy):
    ih, iw = tpl.shape
    x0, y0 = int(round(cx - iw / 2 - SHIFT)), int(round(cy - ih / 2 - SHIFT))
    win = img[max(0, y0):y0 + ih + 2 * SHIFT, max(0, x0):x0 + iw + 2 * SHIFT]
    if win.shape[0] < ih or win.shape[1] < iw:
        return -1.0
    return float(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED).max())


def run(trial, track_dir, out):
    name = os.path.basename(trial.rstrip("/"))
    path = os.path.join(track_dir, f"{name}_track.csv")
    if not os.path.exists(path):
        print(f"--- {name}: no {path}, skipped"); return
    tr = list(csv.DictReader(open(path)))
    ev = A.load_events(trial)
    import json
    roi = json.load(open(os.path.join(trial, "roi.json")))
    bx, by, bw, bh = roi["berry"]
    ref_f = T.frames(trial, ev["hold"], ev["hold"] + A.REF_WIN)
    ref = np.mean([cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) for _, p in ref_f[:30]], axis=0)
    src, scx, scy = ref, bx + bw / 2, by + bh / 2          # image the reference is cut from, and its centre
    base = 0.0                                             # angle of that reference
    ang = 0.0
    fl = dict((round(t, 4), p) for t, p in T.frames(trial, float(tr[0]["t_s"]) - 0.01, float(tr[-1]["t_s"]) + 0.01))
    rows = []
    t0 = time.time()
    for r in tr:
        t = float(r["t_s"])
        p = fl.get(round(t, 4))
        if p is None or r["berry_dx"] == "":
            rows.append([t, "", 0.0]); continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        cx, cy = bx + bw / 2 + float(r["berry_dx"]), by + bh / 2 + float(r["berry_dy"])
        angs = np.arange(ang - SPAN, ang + SPAN + 1e-9, STEP)
        sc = np.array([score_at(img, rot_patch(src, scx, scy, bw, bh, a - base), cx, cy) for a in angs])
        i = int(np.argmax(sc))
        best = float(angs[i])
        if 0 < i < len(sc) - 1:
            d = sc[i - 1] - 2 * sc[i] + sc[i + 1]
            if abs(d) > 1e-9:
                best += STEP * float(np.clip(0.5 * (sc[i - 1] - sc[i + 1]) / d, -0.5, 0.5))
        s = float(sc[i])
        if s < MIN_SCORE:
            rows.append([t, "", round(s, 3)]); continue
        ang = best
        rows.append([t, round(ang, 2), round(s, 3)])
        if s < REANCHOR:
            src, scx, scy, base = img, cx, cy, ang
    with open(os.path.join(out, f"{name}_rot.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t_s", "rot_deg", "score"]); w.writerows(rows)
    a = [x[1] for x in rows if x[1] != ""]
    print(f"--- {name}: {len(rows)} frames in {time.time() - t0:.0f} s; final rotation {a[-1] if a else 'n/a'} deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials", nargs="+")
    ap.add_argument("--track", default=os.path.expanduser("~/track_16mm"))
    ap.add_argument("--out", default=os.path.expanduser("~/rot_16mm"))
    p = ap.parse_args()
    out = os.path.expanduser(p.out).rstrip("/")
    os.makedirs(out, exist_ok=True)
    for trial in p.trials:
        try:
            run(trial, os.path.expanduser(p.track), out)
        except Exception as e:
            print(f"--- {os.path.basename(trial.rstrip('/'))}: FAILED ({type(e).__name__}: {e})")
    with zipfile.ZipFile(out + ".zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out)):
            zf.write(os.path.join(out, f), os.path.join(os.path.basename(out), f))
    print(f"\nwritten: {out}.zip")


if __name__ == "__main__":
    main()
