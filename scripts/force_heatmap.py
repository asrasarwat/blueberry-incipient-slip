from __future__ import annotations
#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
from xensesdk import Sensor

INDEX_ID = "OG000961"
THUMB_ID = "OG000324"

SAVE_DIR = os.path.expanduser("~/force_heatmaps")
os.makedirs(SAVE_DIR, exist_ok=True)

print("Initializing sensors...")
s1 = Sensor.create(INDEX_ID)
s2 = Sensor.create(THUMB_ID)
print("Ready! Saving heatmaps every 0.5 seconds to", SAVE_DIR)
print("Press Ctrl+C to stop")

frame = 0
t0 = time.time()

while True:
    try:
        f1 = s1.selectSensorInfo(Sensor.OutputType.Force)
        f2 = s2.selectSensorInfo(Sensor.OutputType.Force)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Xsense Tactile Force Heatmap — Frame %d (t=%.1fs)" % (frame, time.time()-t0), fontsize=13)

        # Index finger heatmap
        im1 = axes[0].imshow(f1, cmap="hot", interpolation="bilinear")
        axes[0].set_title("Index Fingertip (OG000961)\nTotal Force: %.3f N" % float(np.sum(np.abs(f1))))
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        plt.colorbar(im1, ax=axes[0], label="Force (N)")

        # Thumb heatmap
        im2 = axes[1].imshow(f2, cmap="hot", interpolation="bilinear")
        axes[1].set_title("Thumb Tip (OG000324)\nTotal Force: %.3f N" % float(np.sum(np.abs(f2))))
        axes[1].set_xlabel("X (pixels)")
        axes[1].set_ylabel("Y (pixels)")
        plt.colorbar(im2, ax=axes[1], label="Force (N)")

        plt.tight_layout()
        fname = "%s/heatmap_%04d.png" % (SAVE_DIR, frame)
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()

        print("Saved frame %d — Index=%.3fN  Thumb=%.3fN" % (frame, float(np.sum(np.abs(f1))), float(np.sum(np.abs(f2)))))
        frame += 1
        time.sleep(0.5)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(0.5)

s1.release()
s2.release()
print("Done! Heatmaps saved to", SAVE_DIR)
