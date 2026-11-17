import os
#!/usr/bin/env python3
"""contact_sheet.py <trial> <t_from> <t_to> [step_s] - camera-2 frames every step_s (default 0.5 s) in one labelled
JPEG, ~/sheet_<trial>.jpg, to see when the berry leaves the fingers."""
import csv, os, sys
import cv2, numpy as np
d = os.path.expanduser(sys.argv[1].rstrip("/")); t0, t1 = float(sys.argv[2]), float(sys.argv[3])
step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
rows = [r for r in csv.DictReader(open(os.path.join(d, "camera_cam2_timestamps.csv")))
        if r.get("saved", "1") in ("1", "True") and t0 <= float(r["elapsed_s"]) <= t1]
tiles, last = [], -1e9
for r in rows:
    t = float(r["elapsed_s"])
    if t - last < step - 1e-6:
        continue
    g = cv2.imread(os.path.join(d, "camera_cam2", f"frame_{int(r['frame']):06d}.png"))
    if g is None:
        continue
    last = t
    g = cv2.resize(np.clip(g.astype(np.float32) * 2.2, 0, 255).astype(np.uint8), (320, 256))
    cv2.putText(g, f"{t:.1f}s", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    tiles.append(g)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
out = os.path.expanduser(f"~/sheet_{os.path.basename(d)}.jpg")
cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(f"written {out} ({len([x for x in tiles if x.any()])} frames)")
