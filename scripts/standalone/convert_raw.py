#!/usr/bin/env python3
"""convert_raw.py - convert the camera .raw frames of one or more trial folders to PNG.
Usage: python3 convert_raw.py <trial> [<trial> ...]   (add --keep-raw to keep the .raw files)"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

DEFAULT = {"width": 1280, "height": 1024, "pixel_type": "0x108000a"}


def bayer_code(pixel_type):
    import cv2
    return {"0x1080008": cv2.COLOR_BayerGB2BGR, "0x1080009": cv2.COLOR_BayerBG2BGR,
            "0x108000a": cv2.COLOR_BayerGR2BGR, "0x108000b": cv2.COLOR_BayerRG2BGR}.get(str(pixel_type).lower())


def convert_one(job):
    import cv2
    path, w, h, code, keep_raw = job
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != w * h:
        return 0
    img = raw.reshape(h, w)
    img = cv2.cvtColor(img, code) if code is not None else img
    ok = cv2.imwrite(path[:-4] + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if ok and not keep_raw:
        os.remove(path)
    return int(ok)


def camera_meta(trial):
    try:
        with open(os.path.join(trial, "trial_info.json")) as fh:
            return json.load(fh).get("cameras", {})
    except (OSError, ValueError):
        return {}


def main():
    keep_raw = "--keep-raw" in sys.argv
    trials = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not trials:
        sys.exit(__doc__)
    with Pool(max(1, cpu_count() - 1)) as pool:
        for trial in trials:
            cams = camera_meta(trial)
            for cam_dir in sorted(glob.glob(os.path.join(trial, "camera_cam*"))):
                if not os.path.isdir(cam_dir):
                    continue
                name = os.path.basename(cam_dir).replace("camera_", "")
                m = {**DEFAULT, **{k: v for k, v in cams.get(name, {}).items() if k in DEFAULT}}
                files = sorted(glob.glob(os.path.join(cam_dir, "frame_*.raw")))
                if not files:
                    print(f"{trial} {name}: no .raw files (already converted?)")
                    continue
                code = bayer_code(m["pixel_type"])
                t = time.time()
                jobs = [(f, int(m["width"]), int(m["height"]), code, keep_raw) for f in files]
                done = sum(pool.imap_unordered(convert_one, jobs, chunksize=16))
                print(f"{trial} {name}: converted {done}/{len(files)} frames in {time.time() - t:.0f} s")
                if done < len(files):
                    print(f"   {len(files) - done} frames skipped (wrong size) - their .raw files were kept")


if __name__ == "__main__":
    main()
