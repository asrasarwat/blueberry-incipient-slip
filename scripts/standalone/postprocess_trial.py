import os
#!/usr/bin/env python3
"""
postprocess_trial.py  -  run AFTER a capture, not during it.

  python3 postprocess_trial.py /media/dataset_storage/toy/blueberry/trial_YYYYMMDD_HHMMSS [--every 1] [--video]

  1. Converts raw Bayer frames (camera_cam*/frame_*.raw) to PNG (and optionally MP4)
  2. Draws taxel heatmaps with shear arrows from taxels.npz (heatmaps/)
  3. Prints a quality check: tactile rate, camera fps, frozen-sensor samples, drops
"""
import argparse, csv, glob, json, os
import numpy as np
import cv2

BAYER = {  # Hikrobot PixelType values -> OpenCV conversion
    "0x1080008": cv2.COLOR_BayerGR2BGR, "0x1080009": cv2.COLOR_BayerRG2BGR,
    "0x108000a": cv2.COLOR_BayerGB2BGR, "0x108000b": cv2.COLOR_BayerBG2BGR,
}


def convert_camera(trial, cam, meta, every, video):
    d = os.path.join(trial, f"camera_{cam}")
    files = sorted(glob.glob(os.path.join(d, "frame_*.raw")))
    if not files:
        return
    w, h, pt = meta["width"], meta["height"], meta["pixel_type"].lower()
    code = BAYER.get(pt)
    out = os.path.join(d, "png"); os.makedirs(out, exist_ok=True)
    vw = None
    if video:
        fps = meta.get("fps", 150) or 150
        vw = cv2.VideoWriter(os.path.join(trial, f"{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i, f in enumerate(files):
        if i % every:
            continue
        raw = np.fromfile(f, dtype=np.uint8)
        if raw.size != w * h:
            print(f"[{cam}] {os.path.basename(f)}: unexpected size {raw.size}, skipped"); continue
        img = raw.reshape(h, w)
        img = cv2.cvtColor(img, code) if code is not None else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(out, os.path.basename(f).replace(".raw", ".png")), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if vw is not None:
            vw.write(img)
    if vw is not None:
        vw.release()
    print(f"[{cam}] converted {len(files[::every])} frames (pixel type {pt})")


def heatmaps(trial, every):
    p = os.path.join(trial, "taxels.npz")
    if not os.path.exists(p):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(p); idx, thm, t = z["index"], z["thumb"], z["t"]
    out = os.path.join(trial, "heatmaps"); os.makedirs(out, exist_ok=True)
    vmax = max(np.abs(idx[..., 2]).max(), np.abs(thm[..., 2]).max(), 1e-6)   # ONE colour scale for the whole trial
    for k in range(0, len(t), every):
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        for ax, arr, name in ((axs[0], idx[k], "Index"), (axs[1], thm[k], "Thumb")):
            im = ax.imshow(np.abs(arr[..., 2]), cmap="magma", vmin=0, vmax=vmax)
            r, c = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
            ax.quiver(c, r, arr[..., 0], -arr[..., 1], color="cyan", scale=None, width=0.004)
            ax.set_title(f"{name}  |Fz| sum {np.abs(arr[..., 2]).sum():.2f}"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axs, shrink=0.8, label="|Fz| per taxel")
        fig.suptitle(f"t = {t[k]:.3f} s")
        fig.savefig(os.path.join(out, f"heatmap_{k:06d}.png"), dpi=110); plt.close(fig)
    print(f"[heatmaps] {len(range(0, len(t), every))} images (shared colour scale, arrows = shear Fx, Fy)")


def quality(trial):
    info = json.load(open(os.path.join(trial, "trial_info.json")))
    print("\n=== Quality check ===")
    print(f"tactile rate: {info.get('tactile_rate_hz')} Hz  (want >= 30)")
    for cam, m in info.get("cameras", {}).items():
        print(f"{cam}: {m.get('frames')} frames @ {m.get('fps')} fps, queue waits {m.get('queue_full_waits')}")
    rows = list(csv.DictReader(open(os.path.join(trial, "tactile_data.csv"))))
    for s in ("index", "thumb"):
        n = sum(1 for r in rows if int(r[f"{s}_stale"]) >= 25)
        print(f"{s}: {n} samples flagged frozen (>= 25 identical arrays)")
    print(f"hold {info.get('hold_s')} s, rate {info.get('loosen_rate_rad_s')} rad/s, "
          f"drop at {info.get('drop_detected_elapsed_s', 'not detected')} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--every", type=int, default=1, help="convert / draw every Nth frame")
    ap.add_argument("--video", action="store_true", help="also write cam1.mp4 / cam2.mp4")
    ap.add_argument("--heatmap-every", type=int, default=5)
    a = ap.parse_args()
    info = json.load(open(os.path.join(a.trial, "trial_info.json")))
    for cam, meta in info.get("cameras", {}).items():
        if "width" in meta:
            convert_camera(a.trial, cam, meta, a.every, a.video)
    heatmaps(a.trial, a.heatmap_every)
    quality(a.trial)
