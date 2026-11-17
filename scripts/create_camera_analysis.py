import os
#!/usr/bin/env python3
"""
Generate camera-based slip analysis figures (matching original report style)
Creates: Frame sequences, annotated slip progression, and camera-tactile correlation
"""
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

TRIAL = "/media/dataset_storage/toy/blueberry/trial_20261014_122604"
OUTPUT_DIR = os.path.expanduser("~/Desktop/blueberry_analysis")

# Load metadata
with open(f"{TRIAL}/trial_info.json") as f:
    info = json.load(f)

# Load timestamps
cam1_ts = pd.read_csv(f"{TRIAL}/camera_cam1_timestamps.csv")
cam2_ts = pd.read_csv(f"{TRIAL}/camera_cam2_timestamps.csv")
tactile = pd.read_csv(f"{TRIAL}/tactile_data.csv")

print(f"Loaded trial {info['run_ts']}")
print(f"  Cam1: {len(cam1_ts)} frames, {info['cameras']['cam1']['fps']:.1f} fps")
print(f"  Cam2: {len(cam2_ts)} frames, {info['cameras']['cam2']['fps']:.1f} fps")

# Identify slip onset from tactile
tactile["force_drop"] = tactile["index_Fz"].diff().abs()
slip_idx = tactile[tactile["force_drop"] > 1.0].index
if len(slip_idx) > 0:
    slip_t = tactile.loc[slip_idx[0], "elapsed_s"]
    print(f"  Slip onset: {slip_t:.3f}s")
else:
    slip_t = 3.0  # fallback
    print(f"  Using default slip time: {slip_t}s")

# Find camera frames around slip event
# Key moments: stable grip, incipient slip, full slip
cam1_ts["elapsed_s"] = cam1_ts["elapsed_s"].astype(float)
cam2_ts["elapsed_s"] = cam2_ts["elapsed_s"].astype(float)

# Select 7 frames spanning stable → slip
times = [
    slip_t - 2.0,  # stable
    slip_t - 1.0,  # stable
    slip_t - 0.5,  # pre-slip
    slip_t,        # incipient
    slip_t + 0.3,  # progressing
    slip_t + 0.8,  # full slip
    slip_t + 2.0   # post-slip
]

key_frames_cam1 = []
key_frames_cam2 = []
for t in times:
    idx1 = (cam1_ts["elapsed_s"] - t).abs().idxmin()
    idx2 = (cam2_ts["elapsed_s"] - t).abs().idxmin()
    key_frames_cam1.append({
        "frame": int(cam1_ts.loc[idx1, "frame"]),
        "elapsed_s": cam1_ts.loc[idx1, "elapsed_s"]
    })
    key_frames_cam2.append({
        "frame": int(cam2_ts.loc[idx2, "frame"]),
        "elapsed_s": cam2_ts.loc[idx2, "elapsed_s"]
    })

print(f"\nSelected key frames (cam1): {[f['frame'] for f in key_frames_cam1]}")
print(f"Selected key frames (cam2): {[f['frame'] for f in key_frames_cam2]}")

# ============================================================
# FIGURE 6: Camera Frame Sequence (both cameras)
# ============================================================
fig, axes = plt.subplots(2, 7, figsize=(20, 6))
fig.suptitle(f"Blueberry Trial {info['run_ts']} - Slip Progression (Camera Evidence)",
             fontsize=16, fontweight='bold')

labels = ["Stable", "Stable", "Pre-slip", "Incipient", "Progressing", "Full slip", "Post-slip"]

# Load and display cam1 frames
for i, (kf, label) in enumerate(zip(key_frames_cam1, labels)):
    img_path = f"{TRIAL}/camera_cam1/frame_{kf['frame']:06d}.png"
    if Path(img_path).exists():
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
    else:
        axes[0, i].text(0.5, 0.5, "Frame\nMissing", ha='center', va='center',
                       fontsize=10, transform=axes[0, i].transAxes)
    axes[0, i].set_title(f"Frame {kf['frame']}\n{label}\nt={kf['elapsed_s']:.2f}s",
                        fontsize=9)
    axes[0, i].axis('off')

# Camera label
axes[0, 0].text(-0.15, 0.5, "Camera 1", rotation=90, va='center',
               fontsize=12, fontweight='bold', transform=axes[0, 0].transAxes)

# Load and display cam2 frames
for i, (kf, label) in enumerate(zip(key_frames_cam2, labels)):
    img_path = f"{TRIAL}/camera_cam2/frame_{kf['frame']:06d}.png"
    if Path(img_path).exists():
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
    else:
        axes[1, i].text(0.5, 0.5, "Frame\nMissing", ha='center', va='center',
                       fontsize=10, transform=axes[1, i].transAxes)
    axes[1, i].set_title(f"Frame {kf['frame']}\nt={kf['elapsed_s']:.2f}s", fontsize=9)
    axes[1, i].axis('off')

axes[1, 0].text(-0.15, 0.5, "Camera 2", rotation=90, va='center',
               fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure6_camera_sequence.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved Figure6_camera_sequence.png")

# ============================================================
# FIGURE 7: Camera-Tactile Correlation
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(f"Blueberry Trial {info['run_ts']} - Camera-Tactile Synchronization",
             fontsize=16, fontweight='bold')

# Panel A: Camera frame rate (frames over time)
ax = axes[0]
ax.scatter(cam1_ts["elapsed_s"], cam1_ts["frame"], s=2, alpha=0.5, label="Cam1")
ax.scatter(cam2_ts["elapsed_s"], cam2_ts["frame"], s=2, alpha=0.5, label="Cam2")
for kf in key_frames_cam1[:4]:  # Mark first 4 key frames
    ax.axvline(kf["elapsed_s"], color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, label=f"Slip onset ({slip_t:.2f}s)")
ax.set_ylabel("Frame Number", fontsize=12, fontweight='bold')
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.95, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Panel B: Tactile forces
ax = axes[1]
ax.plot(tactile["elapsed_s"], tactile["index_Fz"], 'b-', linewidth=2, label="Index Fz")
ax.plot(tactile["elapsed_s"], tactile["thumb_Fz"], 'r-', linewidth=2, label="Thumb Fz")
for kf in key_frames_cam1[:4]:
    ax.axvline(kf["elapsed_s"], color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(slip_t, color='k', linestyle='--', linewidth=2)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
ax.set_ylabel("Normal Force (N)", fontsize=12, fontweight='bold')
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.95, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Panel C: Contact areas
ax = axes[2]
ax.plot(tactile["elapsed_s"], tactile["index_contact_area"], 'b-', linewidth=2, label="Index area")
ax.plot(tactile["elapsed_s"], tactile["thumb_contact_area"], 'r-', linewidth=2, label="Thumb area")
for kf in key_frames_cam1[:4]:
    ax.axvline(kf["elapsed_s"], color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(slip_t, color='k', linestyle='--', linewidth=2)
ax.set_xlabel("Elapsed Time from Loosening Onset (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Contact Area (pixels)", fontsize=12, fontweight='bold')
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.95, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure7_camera_tactile_sync.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure7_camera_tactile_sync.png")

print(f"\n{'='*60}")
print(f"CAMERA ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Generated 2 additional figures:")
print(f"  - Figure 6: Frame sequence showing visual slip progression")
print(f"  - Figure 7: Camera-tactile synchronization verification")
print(f"\nKey frames correspond to tactile slip onset at {slip_t:.3f}s")
print(f"All figures in {OUTPUT_DIR}/")
print(f"{'='*60}")
