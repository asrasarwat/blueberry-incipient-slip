import os
#!/usr/bin/env python3
"""check_drop.py <trial> - when did the berry really fall, and where are the sensor gaps?"""
import csv, json, os, sys
import numpy as np
import cv2

d = os.path.expanduser(sys.argv[1].rstrip("/"))
info = json.load(open(os.path.join(d, "trial_info.json")))
ev = {r["event"]: float(r["elapsed_s"]) for r in csv.DictReader(open(os.path.join(d, "events.csv"))) if r["elapsed_s"]}
t_lo = info.get("loosen_start_elapsed_s", ev.get("loosen"))
t_end = ev.get("post", ev.get("drop"))
print(f"trial {os.path.basename(d)}  loosen {t_lo:.2f} s  script drop {ev.get('drop', 'none')}  end {t_end}")
print("unloaded |Fz| (N) index, thumb:", info.get("unloaded_fz_N"))

sc = list(csv.DictReader(open(os.path.join(d, "slip_curve.csv"))))
st = np.array([float(r["elapsed_s"]) for r in sc]); ang = np.array([float(r["loosen_rad"]) for r in sc])
fi = np.array([float(r["index_Fz_N"]) for r in sc]); ft = np.array([float(r["thumb_Fz_N"]) for r in sc])
rad_at = lambda t: float(np.interp(t, st, ang))

z = np.load(os.path.join(d, "taxels.npz"))
t, new, stale = z["t"], z["new"], z["stale"]
win = t >= t_lo
for k, name in enumerate(("index", "thumb")):
    ok = win & (new[:, k] == 1) & (stale[:, k] == 0)
    tt = t[ok]; g = np.diff(tt)
    big = np.where(g > 0.2)[0]
    print(f"{name} gaps > 0.2 s after loosening: " + (", ".join(
        f"{g[i]:.2f} s at t={tt[i]:.1f} ({rad_at(tt[i]):.3f} rad)" for i in big) or "none"))

print("\n  t_s   rad    index_N thumb_N   (1-s means)")
for t0 in np.arange(t_lo, st[-1] + 1e-9, 2.0):
    m = (st >= t0) & (st < t0 + 1.0)
    if m.any():
        print(f"{t0:6.1f} {rad_at(t0):6.3f}  {fi[m].mean():6.2f}  {ft[m].mean():6.2f}")

rows = [r for r in csv.DictReader(open(os.path.join(d, "camera_cam2_timestamps.csv")))
        if r.get("saved", "1") in ("1", "True") and t_lo - 0.5 <= float(r["elapsed_s"]) <= (t_end or 1e9)]
pick, last = [], -1e9
for r in rows:
    if float(r["elapsed_s"]) - last >= 1.0:
        pick.append(r); last = float(r["elapsed_s"])
ref, out = None, []
for r in pick:
    p = os.path.join(d, "camera_cam2", f"frame_{int(r['frame']):06d}.png")
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None:
        continue
    g = g[200:850, 550:1100].astype(np.float32)
    ref = g if ref is None else ref
    out.append((float(r["elapsed_s"]), float(np.abs(g - ref).mean())))
if out:
    mx = max(v for _, v in out)
    first = next((tt for tt, v in out if v >= 0.5 * mx), None)
    print("\ncamera 2 change in berry region (every 1 s):")
    print("  ".join(f"{tt:.1f}s:{v:.1f}" for tt, v in out))
    if first is not None:
        print(f"\nberry leaves the fingers (camera, half of max change) at t = {first:.1f} s = {rad_at(first):.3f} rad, "
              f"{first - t_lo:.1f} s after loosening began")
