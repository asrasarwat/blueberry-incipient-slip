#!/usr/bin/env python3
"""
ruler_calibrate.py - pixel scale (px/mm) of camera 2 from the mm ruler in view.

Put the ruler beside the fingertips, at the same distance from camera 2 as the berry (same depth),
and keep it there for the whole session. Then, for any trial recorded in that session:

  python3 ruler_calibrate.py <trial_folder>              (opens a window; default span 20 mm)
  python3 ruler_calibrate.py <trial_folder> --mm 30      (click two marks 30 mm apart)

In the window: click the first ruler mark, then the second mark --mm further along, then press ENTER.
Press r to redo the clicks, q to quit without saving. The window can be zoomed with + / -.
Writes <trial_folder>/ruler.json  {"px_per_mm": ..., "points": [[x1, y1], [x2, y2]], "span_mm": ..., "frame": ...}

Without a display (or if OpenCV has no GUI): it saves ruler_frame.png next to the trial so you can read the
two pixel positions in any image viewer, then
  python3 ruler_calibrate.py <trial_folder> --points 412,880,832,881 --mm 20
Use a long span (20-30 mm): a 1-pixel click error is then about 0.2 % of the scale.
"""
import argparse
import csv
import json
import math
import os
import sys

import cv2


def pick_frame(trial):
    """First saved camera-2 frame of the hold phase (the berry is still and the ruler unobstructed)."""
    ts = os.path.join(trial, "camera_cam2_timestamps.csv")
    best = None
    if os.path.exists(ts):
        with open(ts) as fh:
            for r in csv.DictReader(fh):
                if r.get("saved", "1") not in ("1", "True"):
                    continue
                p = os.path.join(trial, "camera_cam2", f"frame_{int(r['frame']):06d}.png")
                if not os.path.exists(p):
                    continue
                if r.get("phase") == "hold":
                    return p
                best = best or p
    if best:
        return best
    d = os.path.join(trial, "camera_cam2")
    files = sorted(f for f in os.listdir(d) if f.endswith(".png")) if os.path.isdir(d) else []
    if not files:
        sys.exit(f"no camera-2 PNG frames found in {trial}")
    return os.path.join(d, files[0])


def save(trial, frame, pts, span):
    px = math.dist(pts[0], pts[1])
    out = {"px_per_mm": round(px / span, 4), "span_mm": span, "span_px": round(px, 2),
           "points": [list(map(float, p)) for p in pts], "frame": os.path.relpath(frame, trial)}
    with open(os.path.join(trial, "ruler.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{px:.1f} px over {span} mm  ->  {out['px_per_mm']} px/mm   (saved {os.path.join(trial, 'ruler.json')})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial")
    ap.add_argument("--mm", type=float, default=20.0, help="distance between the two clicked marks (mm)")
    ap.add_argument("--points", default=None, help="x1,y1,x2,y2 in full-image pixels (no window)")
    ap.add_argument("--frame", default=None, help="use this image instead of picking one")
    p = ap.parse_args()
    frame = p.frame or pick_frame(p.trial)
    img = cv2.imread(frame)
    if img is None:
        sys.exit(f"cannot read {frame}")
    if p.points:
        v = [float(x) for x in p.points.split(",")]
        save(p.trial, frame, [(v[0], v[1]), (v[2], v[3])], p.mm); return

    pts, zoom = [], [1.0 if img.shape[1] <= 1400 else 1400 / img.shape[1]]

    def draw():
        z = zoom[0]
        view = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST if z > 1 else cv2.INTER_AREA)
        for q in pts:
            c = (int(q[0] * z), int(q[1] * z))
            cv2.drawMarker(view, c, (0, 0, 255), cv2.MARKER_CROSS, 24, 1)
        if len(pts) == 2:
            a = (int(pts[0][0] * z), int(pts[0][1] * z)); b = (int(pts[1][0] * z), int(pts[1][1] * z))
            cv2.line(view, a, b, (0, 255, 255), 1)
            txt = f"{math.dist(*pts):.1f} px / {p.mm} mm = {math.dist(*pts) / p.mm:.3f} px/mm  - ENTER to save, r to redo"
        else:
            txt = f"click mark {len(pts) + 1} of 2 ({p.mm} mm apart)   +/- zoom   q quit"
        cv2.putText(view, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ruler", view)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x / zoom[0], y / zoom[0])); draw()

    try:
        cv2.namedWindow("ruler", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("ruler", on_mouse)
    except cv2.error:
        out = os.path.join(p.trial, "ruler_frame.png")
        cv2.imwrite(out, img)
        sys.exit(f"OpenCV has no window support here. Saved {out}; read the two mark positions in an image "
                 f"viewer and run again with --points x1,y1,x2,y2 --mm {p.mm}")
    draw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (13, 10) and len(pts) == 2:
            save(p.trial, frame, pts, p.mm); break
        if k == ord("r"):
            pts.clear(); draw()
        if k in (ord("+"), ord("=")):
            zoom[0] *= 1.25; draw()
        if k == ord("-"):
            zoom[0] /= 1.25; draw()
        if k == ord("q"):
            print("quit, nothing saved"); break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
