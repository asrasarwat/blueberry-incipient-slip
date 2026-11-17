#!/usr/bin/env python3
"""
Complete blueberry trial analysis - generate all report-style figures
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

TRIAL = "/media/dataset_storage/toy/blueberry/trial2"
OUTPUT_DIR = os.path.expanduser("~/Desktop/blueberry_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
tactile = pd.read_csv(f"{TRIAL}/tactile_data.csv")
cam1_ts = pd.read_csv(f"{TRIAL}/camera_cam1_timestamps.csv")
cam2_ts = pd.read_csv(f"{TRIAL}/camera_cam2_timestamps.csv")
with open(f"{TRIAL}/trial_info.json") as f:
    info = json.load(f)

print(f"Loaded trial {info['run_ts']}")
print(f"  Tactile: {len(tactile)} samples @ {info['tactile_rate_hz']:.1f} Hz")
print(f"  Cam1: {len(cam1_ts)} frames @ {info['cameras']['cam1']['fps']:.1f} fps")
print(f"  Cam2: {len(cam2_ts)} frames @ {info['cameras']['cam2']['fps']:.1f} fps")

# Detect frozen sensor issue
tactile["force_drop"] = tactile["index_Fz"].diff().abs()
large_jumps = tactile[tactile["force_drop"] > 1.0].index

# Check if index sensor was frozen (constant value for >10 samples)
index_frozen = False
if len(tactile) > 15:
    first_val = tactile["index_Fz"].iloc[0]
    frozen_count = (tactile["index_Fz"].iloc[:20] == first_val).sum()
    if frozen_count > 10:
        index_frozen = True
        print(f"  WARNING: Index sensor frozen at {first_val:.4f}N for {frozen_count} samples")

# Only mark slip if not a sensor artifact
if len(large_jumps) > 0 and index_frozen:
    print(f"  Large force jump at t={tactile.loc[large_jumps[0], 'elapsed_s']:.3f}s is sensor UNFREEZE, not slip")
    slip_frame, slip_t = None, None
elif len(large_jumps) > 0:
    slip_frame = large_jumps[0]
    slip_t = tactile.loc[slip_frame, "elapsed_s"]
    print(f"  Potential slip: frame {slip_frame}, t={slip_t:.3f}s")
else:
    slip_frame, slip_t = None, None

# ============================================================
# FIGURE 1: Force + Contact Timeline (Figure 5 style)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f"Blueberry Trial {info['run_ts']} - Tactile Timeline",
             fontsize=16, fontweight='bold', y=0.995)

# Panel A: Normal Forces
ax = axes[0]
ax.plot(tactile["elapsed_s"], tactile["index_Fz"], 'b-', linewidth=2.5, label="Index Fz")
ax.plot(tactile["elapsed_s"], tactile["thumb_Fz"], 'r-', linewidth=2.5, label="Thumb Fz")
if slip_t is not None:
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7,
               label=f"Slip onset ({slip_t:.2f}s)")
ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
ax.set_ylabel("Normal Force (N)", fontsize=13, fontweight='bold')
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.text(0.02, 0.95, 'A', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top')

# Panel B: Contact Areas
ax = axes[1]
ax.plot(tactile["elapsed_s"], tactile["index_contact_area"], 'b-',
        linewidth=2.5, label="Index area")
ax.plot(tactile["elapsed_s"], tactile["thumb_contact_area"], 'r-',
        linewidth=2.5, label="Thumb area")
if slip_t is not None:
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.set_ylabel("Contact Area (pixels)", fontsize=13, fontweight='bold')
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.text(0.02, 0.95, 'B', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top')

# Panel C: Shear Forces
ax = axes[2]
ax.plot(tactile["elapsed_s"], tactile["index_shear_N"], 'b-',
        linewidth=2.5, label="Index shear")
ax.plot(tactile["elapsed_s"], tactile["thumb_shear_N"], 'r-',
        linewidth=2.5, label="Thumb shear")
if slip_t is not None:
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.set_xlabel("Elapsed Time from Loosening Onset (s)", fontsize=13, fontweight='bold')
ax.set_ylabel("Shear Force (N)", fontsize=13, fontweight='bold')
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.text(0.02, 0.95, 'C', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure1_timeline.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved Figure1_timeline.png")

# ============================================================
# FIGURE 2: Motion Analysis (optical flow magnitude)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(tactile["elapsed_s"], tactile["motion_magnitude"], 'purple',
        linewidth=2.5, label="Motion magnitude")
ax.plot(tactile["elapsed_s"], tactile["motion_max"], 'orange',
        linewidth=2, alpha=0.7, label="Max motion")
if slip_t is not None:
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7,
               label=f"Slip onset ({slip_t:.2f}s)")
ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.set_xlabel("Elapsed Time from Loosening Onset (s)", fontsize=13, fontweight='bold')
ax.set_ylabel("Motion Magnitude (pixels)", fontsize=13, fontweight='bold')
ax.set_title(f"Blueberry Trial {info['run_ts']} - Surface Motion",
             fontsize=15, fontweight='bold')
ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure2_motion.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure2_motion.png")

# ============================================================
# FIGURE 3: Grip Strength Evolution (index+thumb combined)
# ============================================================
tactile["total_Fz"] = tactile["index_Fz"].abs() + tactile["thumb_Fz"].abs()
tactile["total_area"] = tactile["index_contact_area"] + tactile["thumb_contact_area"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Blueberry Trial {info['run_ts']} - Grip Strength Evolution",
             fontsize=16, fontweight='bold', y=0.995)

# Panel A: Total normal force
ax = axes[0]
ax.plot(tactile["elapsed_s"], tactile["total_Fz"], 'darkgreen', linewidth=3)
if slip_t is not None:
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7,
               label=f"Slip onset ({slip_t:.2f}s)")
ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
ax.set_ylabel("Total Normal Force |Fz| (N)", fontsize=13, fontweight='bold')
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.text(0.02, 0.95, 'A', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top')

# Panel B: Total contact area
ax = axes[1]
ax.plot(tactile["elapsed_s"], tactile["total_area"], 'darkblue', linewidth=3)
if slip_t is not None:
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.set_xlabel("Elapsed Time from Loosening Onset (s)", fontsize=13, fontweight='bold')
ax.set_ylabel("Total Contact Area (pixels)", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.text(0.02, 0.95, 'B', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure3_grip_evolution.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure3_grip_evolution.png")

# ============================================================
# FIGURE 4: Pre-slip vs Post-slip Comparison
# ============================================================
if slip_t is not None:
    pre_slip = tactile[tactile["elapsed_s"] < slip_t]
    post_slip = tactile[tactile["elapsed_s"] >= slip_t]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Blueberry Trial {info['run_ts']} - Pre-slip vs Post-slip",
                 fontsize=16, fontweight='bold', y=0.995)

    # Force comparison
    ax = axes[0, 0]
    ax.hist(pre_slip["index_Fz"].abs(), bins=20, alpha=0.6, label="Pre-slip", color='blue')
    ax.hist(post_slip["index_Fz"].abs(), bins=20, alpha=0.6, label="Post-slip", color='red')
    ax.set_xlabel("Index |Fz| (N)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title("Index Normal Force Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(pre_slip["thumb_Fz"].abs(), bins=20, alpha=0.6, label="Pre-slip", color='blue')
    ax.hist(post_slip["thumb_Fz"].abs(), bins=20, alpha=0.6, label="Post-slip", color='red')
    ax.set_xlabel("Thumb |Fz| (N)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title("Thumb Normal Force Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Contact area comparison
    ax = axes[1, 0]
    ax.hist(pre_slip["index_contact_area"], bins=20, alpha=0.6, label="Pre-slip", color='blue')
    ax.hist(post_slip["index_contact_area"], bins=20, alpha=0.6, label="Post-slip", color='red')
    ax.set_xlabel("Index Contact Area (pixels)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title("Index Contact Area Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(pre_slip["thumb_contact_area"], bins=20, alpha=0.6, label="Pre-slip", color='blue')
    ax.hist(post_slip["thumb_contact_area"], bins=20, alpha=0.6, label="Post-slip", color='red')
    ax.set_xlabel("Thumb Contact Area (pixels)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title("Thumb Contact Area Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure4_pre_post_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure4_pre_post_comparison.png")

# ============================================================
# FIGURE 5: Detailed Slip Event Zoom (±1 second window)
# ============================================================
if slip_t is not None:
    window = 1.0  # seconds
    zoom_data = tactile[(tactile["elapsed_s"] >= slip_t - window) &
                        (tactile["elapsed_s"] <= slip_t + window)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Blueberry Trial {info['run_ts']} - Slip Event Detail (±{window}s)",
                 fontsize=16, fontweight='bold', y=0.995)

    # Forces
    ax = axes[0]
    ax.plot(zoom_data["elapsed_s"], zoom_data["index_Fz"], 'b-', linewidth=2.5, label="Index Fz")
    ax.plot(zoom_data["elapsed_s"], zoom_data["thumb_Fz"], 'r-', linewidth=2.5, label="Thumb Fz")
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel("Normal Force (N)", fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Contact areas
    ax = axes[1]
    ax.plot(zoom_data["elapsed_s"], zoom_data["index_contact_area"], 'b-', linewidth=2.5, label="Index")
    ax.plot(zoom_data["elapsed_s"], zoom_data["thumb_contact_area"], 'r-', linewidth=2.5, label="Thumb")
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel("Contact Area (px)", fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Shear forces
    ax = axes[2]
    ax.plot(zoom_data["elapsed_s"], zoom_data["index_shear_N"], 'b-', linewidth=2.5, label="Index")
    ax.plot(zoom_data["elapsed_s"], zoom_data["thumb_shear_N"], 'r-', linewidth=2.5, label="Thumb")
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel("Shear Force (N)", fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Motion
    ax = axes[3]
    ax.plot(zoom_data["elapsed_s"], zoom_data["motion_magnitude"], 'purple', linewidth=2.5)
    ax.axvline(slip_t, color='k', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel("Elapsed Time (s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Motion (px)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure5_slip_event_zoom.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure5_slip_event_zoom.png")

# ============================================================
# Summary Statistics
# ============================================================
print(f"\n{'='*60}")
print(f"TIMING VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"Trial: {info['run_ts']}")
print(f"Object: {info['object']}")
print(f"Hold duration: {info['hold_s']}s")
print(f"Loosening rate: {info['loosen_rate_rad_s']} rad/s")
print(f"Total loosening: {info['loosen_total_rad']:.3f} rad over {info['loosen_end_elapsed_s']:.2f}s")
print(f"\nData rates:")
print(f"  Tactile: {info['tactile_samples']} samples @ {info['tactile_rate_hz']:.2f} Hz")
print(f"  Cam1: {info['cameras']['cam1']['frames']} frames @ {info['cameras']['cam1']['fps']:.1f} fps")
print(f"  Cam2: {info['cameras']['cam2']['frames']} frames @ {info['cameras']['cam2']['fps']:.1f} fps")
print(f"\nt0 reference: {info['t0']:.3f} (loosening onset)")
if slip_t is not None:
    print(f"Event detected at: {slip_t:.3f}s after loosening began")
    print(f"  → Force drop: {pre_slip['index_Fz'].iloc[-1]:.2f}N → {post_slip['index_Fz'].iloc[0]:.2f}N (index)")
elif index_frozen:
    print(f"WARNING: Index sensor frozen - no valid slip detection possible")
    print(f"  Only thumb sensor provides usable data")
print(f"\nAll figures saved to {OUTPUT_DIR}/")
print(f"{'='*60}")
