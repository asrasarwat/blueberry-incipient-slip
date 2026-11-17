#!/usr/bin/env python3
"""
track_camera_v3.py - camera-2 tracking of the berry AND both fingertips with template matching, so slip can be
measured as berry motion relative to the fingertips (tangential = along the finger surface, image x; normal =
image y).

Why v3: in the 16 mm batch the phase-correlation fingertip regions reported ~0 motion while the DIP joints opened
0.04-0.15 rad and the fingers visibly moved, so the berry moving WITH the fingers was counted as slip. Here each
fingertip is tracked by two patches on the finger itself (left and right of the berry, so the berry never enters
the template), the berry by its markers, all with normalised cross-correlation in a search window around the
last position. Templates are re-anchored when the match score drops (berry rolling), and every match score is
saved so bad tracking is visible.

Per trial it writes to <out>/:
  <name>_track.csv     t_s, berry/topL/topR/botL/botR x,y displacement (px, full resolution) and match score
  <name>_track.jpg     8 frames from loosening to drop with the tracked boxes drawn (check these!)
  <name>_events.json   hold / loosen / drop times and the regions used
and zips the folder to <out>.zip

Usage (same folder as analyze_slip.py; trials need roi.json):
  python3 track_camera_v3.py <trial> [<trial> ...] --out ~/track_16mm
"""
import argparse
import csv
import json
import os
import time
import zipfile

import cv2
import numpy as np

import analyze_slip as A

SEARCH = 60          # px search margin around the last position (full resolution)
SEARCH_LOST = 160    # after a failed match
MIN_SCORE = 0.35     # below: no valid match this frame
REANCHOR = 0.75      # below: take a fresh template at the matched position
PATCH_W = 150        # fingertip patch width (px)
PATCH_H = 90         # fingertip patch height (px)
SIDE_GAP = 110       # gap between the berry marker box and a fingertip patch (berry outline is wider than the box)
PRE_S = 1.0          # start this long before loosening
POST_S = 0.3         # stop this long after the drop


def frames(trial, t0, t1):
    out = []
    with open(os.path.join(trial, f"camera_{A.CAM}_timestamps.csv")) as fh:
        for r in csv.DictReader(fh):
            if r.get("saved", "1") not in ("1", "True"):
                continue
            t = float(r["elapsed_s"])
            if t0 <= t <= t1:
                p = os.path.join(trial, f"camera_{A.CAM}", f"frame_{int(r['frame']):06d}.png")
                if os.path.exists(p):
                    out.append((t, p))
    return out


def clip_box(b, W, H):
    x, y, w, h = [int(round(v)) for v in b]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return [x0, y0, x1 - x0, y1 - y0]


def edge_row(ref, x, w, y_guess, finger_above):
    """Row of the fingertip edge in columns x..x+w: the strongest finger-to-background step within +-150 px of
    the guess (finger above: bright->dark going down; finger below: dark->bright)."""
    H = ref.shape[0]
    y0, y1 = max(2, y_guess - 150), min(H - 2, y_guess + 150)
    prof = cv2.GaussianBlur(ref[:, max(0, x):x + w].mean(1).reshape(-1, 1), (1, 9), 0).ravel()
    g = np.gradient(prof)[y0:y1]
    return int(y0 + (np.argmin(g) if finger_above else np.argmax(g)))


def patches(roi, W, H, ref=None):
    bx, by, bw, bh = roi["berry"]
    ye = roi["top"][1] + 60          # upper fingertip edge (regions were drawn 60 px above / 40 px below it)
    yb = roi["bottom"][1] + 40       # lower fingertip edge
    lx, rx = bx - SIDE_GAP - PATCH_W, bx + bw + SIDE_GAP
    p = {"berry": [bx, by, bw, bh]}
    for side, x in (("L", lx), ("R", rx)):
        et = edge_row(ref, x, PATCH_W, ye, True) if ref is not None else ye
        eb = edge_row(ref, x, PATCH_W, yb, False) if ref is not None else yb
        p["top" + side] = [x, et - PATCH_H // 2, PATCH_W, PATCH_H]      # centred on the finger edge
        p["bot" + side] = [x, eb - PATCH_H // 2, PATCH_W, PATCH_H]
    return {k: clip_box(v, W, H) for k, v in p.items()}


def subpix(r, loc):
    x, y = loc
    dx = dy = 0.0
    if 0 < x < r.shape[1] - 1:
        a, b, c = r[y, x - 1], r[y, x], r[y, x + 1]
        d = a - 2 * b + c
        dx = 0.5 * (a - c) / d if abs(d) > 1e-9 else 0.0
    if 0 < y < r.shape[0] - 1:
        a, b, c = r[y - 1, x], r[y, x], r[y + 1, x]
        d = a - 2 * b + c
        dy = 0.5 * (a - c) / d if abs(d) > 1e-9 else 0.0
    return x + float(np.clip(dx, -1, 1)), y + float(np.clip(dy, -1, 1))


class Tracker:
    def __init__(self, ref, box):
        x, y, w, h = box
        self.w, self.h = w, h
        self.home = np.array([x, y], float)       # template origin in the reference image
        self.tpl = ref[y:y + h, x:x + w].copy()
        self.tpl_pos = self.home.copy()           # where the current template was cut from
        self.pos = self.home.copy()               # current top-left estimate
        self.lost = False
        self.ok = w >= 20 and h >= 20 and float(self.tpl.std()) > 2.0

    def update(self, img):
        if not self.ok:
            return np.nan, np.nan, 0.0
        H, W = img.shape
        m = SEARCH_LOST if self.lost else SEARCH
        x0, y0 = int(max(0, self.pos[0] - m)), int(max(0, self.pos[1] - m))
        x1, y1 = int(min(W, self.pos[0] + self.w + m)), int(min(H, self.pos[1] + self.h + m))
        win = img[y0:y1, x0:x1]
        if win.shape[0] < self.h + 2 or win.shape[1] < self.w + 2:
            self.lost = True
            return np.nan, np.nan, 0.0
        r = cv2.matchTemplate(win, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        if score < MIN_SCORE:
            self.lost = True
            return np.nan, np.nan, float(score)
        sx, sy = subpix(r, loc)
        self.pos = np.array([x0 + sx, y0 + sy])
        self.lost = False
        if score < REANCHOR:
            ix, iy = int(round(self.pos[0])), int(round(self.pos[1]))
            if 0 <= ix and 0 <= iy and ix + self.w <= W and iy + self.h <= H:
                self.tpl = img[iy:iy + self.h, ix:ix + self.w].copy()
                self.tpl_pos = np.array([ix, iy], float)
                self.pos = self.tpl_pos.copy()
        d = self.pos - self.home
        return float(d[0]), float(d[1]), float(score)


def run(trial, out):
    name = os.path.basename(trial.rstrip("/"))
    ev = A.load_events(trial)
    if "hold" not in ev or "loosen" not in ev:
        print(f"--- {name}: no hold/loosen event, skipped"); return
    roi = json.load(open(os.path.join(trial, "roi.json")))
    t_end = ev.get("drop", ev.get("post", ev["loosen"] + 60)) + POST_S
    ref_f = frames(trial, ev["hold"], ev["hold"] + A.REF_WIN)
    ref = np.mean([cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) for _, p in ref_f[:30]], axis=0)
    H, W = ref.shape
    P = patches(roi, W, H, ref)
    tr = {k: Tracker(ref, b) for k, b in P.items()}
    print(f"--- {name}: tracking {', '.join(k for k, t in tr.items() if t.ok)}"
          + (f" (no texture: {', '.join(k for k, t in tr.items() if not t.ok)})" if not all(t.ok for t in tr.values()) else ""))
    fl = frames(trial, ev["loosen"] - PRE_S, t_end)
    rows, snaps = [], []
    want = list(np.linspace(ev["loosen"], fl[-1][0] if fl else t_end, 8)) if fl else []
    t_start = time.time()
    for t, p in fl:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = g.astype(np.float32)
        row = [round(t, 4)]
        for k in P:
            dx, dy, s = tr[k].update(g)
            row += [round(dx, 2) if np.isfinite(dx) else "", round(dy, 2) if np.isfinite(dy) else "", round(s, 3)]
        rows.append(row)
        if want and t >= want[0]:
            want.pop(0)
            vis = cv2.cvtColor(np.clip(g * 2.2, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for k, col in (("berry", (0, 255, 255)), ("topL", (0, 255, 0)), ("topR", (0, 255, 0)),
                           ("botL", (255, 0, 255)), ("botR", (255, 0, 255))):
                x, y = tr[k].pos
                cv2.rectangle(vis, (int(x), int(y)), (int(x + tr[k].w), int(y + tr[k].h)), col, 2)
            cv2.putText(vis, f"{t - ev['loosen']:+.1f} s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
            snaps.append(cv2.resize(vis, (W // 3, H // 3)))
    with open(os.path.join(out, f"{name}_track.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s"] + [f"{k}_{c}" for k in P for c in ("dx", "dy", "score")])
        w.writerows(rows)
    if snaps:
        while len(snaps) % 4:
            snaps.append(np.zeros_like(snaps[0]))
        sheet = np.vstack([np.hstack(snaps[i:i + 4]) for i in range(0, len(snaps), 4)])
        cv2.imwrite(os.path.join(out, f"{name}_track.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump({"events": ev, "patches": P, "roi": roi}, open(os.path.join(out, f"{name}_events.json"), "w"), indent=1)
    sc = np.array([[r[3 + 3 * i] for i in range(len(P))] for r in rows], float) if rows else np.zeros((0, len(P)))
    good = ", ".join(f"{k} {100 * np.mean(sc[:, i] >= MIN_SCORE):.0f}%" for i, k in enumerate(P)) if len(sc) else "-"
    print(f"    {len(rows)} frames in {time.time() - t_start:.0f} s; valid matches: {good}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials", nargs="+")
    ap.add_argument("--out", default=os.path.expanduser("~/track_camera"))
    p = ap.parse_args()
    out = os.path.expanduser(p.out).rstrip("/")
    os.makedirs(out, exist_ok=True)
    for trial in p.trials:
        try:
            run(trial, out)
        except Exception as e:  # keep going with the other trials
            print(f"--- {os.path.basename(trial.rstrip('/'))}: FAILED ({type(e).__name__}: {e})")
    zpath = out + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out)):
            zf.write(os.path.join(out, f), os.path.join(os.path.basename(out), f))
    print(f"\nwritten: {zpath}")


if __name__ == "__main__":
    main()
