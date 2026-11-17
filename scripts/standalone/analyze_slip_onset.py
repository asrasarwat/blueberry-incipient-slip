#!/usr/bin/env python3
"""
Analyzes camera footage to detect slip onset by tracking optical flow.
Uses Lucas-Kanade optical flow to detect when markers start moving significantly.
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def detect_slip_onset(data_dir, camera='cam1', start_frame=800, end_frame=1900,
                     motion_threshold=2.0, window_size=10):
    """
    Detect slip onset by analyzing optical flow magnitude over time.

    Args:
        data_dir: Path to data directory (e.g., ~/grape_unified_20260916_143409/)
        camera: 'cam1' or 'cam2'
        start_frame: Frame to start analysis (after squeeze, during hold)
        end_frame: Frame to end analysis
        motion_threshold: Optical flow magnitude threshold for slip detection
        window_size: Smoothing window for motion signal

    Returns:
        slip_onset_frame: First frame where sustained motion detected
    """

    data_path = Path(os.path.expanduser(data_dir))
    cam_dir = data_path / f"camera_{camera}"
    timestamps_file = data_path / f"camera_{camera}_timestamps.csv"
    slip_curve_file = data_path / "ground_truth_slip_curve.csv"

    # Load timestamps and slip curve (if exists)
    timestamps = pd.read_csv(timestamps_file)
    slip_curve_file = data_path / "ground_truth_slip_curve.csv"

    if slip_curve_file.exists():
        slip_curve = pd.read_csv(slip_curve_file)
    else:
        slip_curve = None

    print(f"[INFO] Analyzing {camera} from frame {start_frame} to {end_frame}")
    print(f"[INFO] Total frames available: {len(timestamps)}")

    # Optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Feature detection parameters
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Storage for motion analysis
    motion_magnitudes = []
    frame_numbers = []

    # Load first frame and detect features
    prev_frame_path = cam_dir / f"frame_{start_frame:06d}.png"
    prev_gray = cv2.imread(str(prev_frame_path), cv2.IMREAD_GRAYSCALE)

    if prev_gray is None:
        print(f"[ERROR] Could not read frame {start_frame}")
        return None

    # Detect features on grape region (center of image)
    h, w = prev_gray.shape
    roi_mask = np.zeros_like(prev_gray)
    cv2.rectangle(roi_mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=roi_mask, **feature_params)

    if prev_pts is None:
        print("[ERROR] No features detected in first frame")
        return None

    print(f"[INFO] Tracking {len(prev_pts)} features")

    # Analyze each frame
    for frame_idx in range(start_frame + 1, min(end_frame, len(timestamps))):
        frame_path = cam_dir / f"frame_{frame_idx:06d}.png"
        curr_gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

        if curr_gray is None:
            continue

        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params
        )

        # Select good points
        if curr_pts is not None and prev_pts is not None:
            good_curr = curr_pts[status == 1]
            good_prev = prev_pts[status == 1]

            if len(good_curr) > 0:
                # Calculate motion magnitude
                motion = good_curr - good_prev
                magnitudes = np.sqrt(motion[:, 0]**2 + motion[:, 1]**2)
                avg_magnitude = np.mean(magnitudes)

                motion_magnitudes.append(avg_magnitude)
                frame_numbers.append(frame_idx)

        # Re-detect features periodically to handle lost tracks
        if frame_idx % 50 == 0:
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=roi_mask, **feature_params)
            if prev_pts is None:
                print(f"[WARN] Lost all features at frame {frame_idx}, re-detecting")
                break
        else:
            prev_pts = curr_pts

        prev_gray = curr_gray

    # Smooth motion signal
    motion_magnitudes = np.array(motion_magnitudes)
    frame_numbers = np.array(frame_numbers)

    # Apply moving average
    smoothed_motion = np.convolve(motion_magnitudes,
                                   np.ones(window_size)/window_size,
                                   mode='same')

    # Debug: print motion statistics
    print(f"[DEBUG] Motion range: {smoothed_motion.min():.3f} to {smoothed_motion.max():.3f} pixels")
    print(f"[DEBUG] Motion mean: {smoothed_motion.mean():.3f}, std: {smoothed_motion.std():.3f}")
    print(f"[DEBUG] Threshold: {motion_threshold}")

    # Find slip onset: first significant motion spike above threshold
    slip_onset_frame = None
    above_threshold = smoothed_motion > motion_threshold
    if np.any(above_threshold):
        # Find first frame where smoothed motion crosses threshold
        first_idx = np.where(above_threshold)[0][0]
        slip_onset_frame = frame_numbers[first_idx]

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot motion magnitude
    ax1.plot(frame_numbers, motion_magnitudes, 'b-', alpha=0.3, label='Raw motion')
    ax1.plot(frame_numbers, smoothed_motion, 'r-', linewidth=2, label='Smoothed')
    ax1.axhline(y=motion_threshold, color='g', linestyle='--', label='Threshold')

    if slip_onset_frame:
        ax1.axvline(x=slip_onset_frame, color='orange', linestyle='--',
                   linewidth=2, label=f'Slip onset (frame {slip_onset_frame})')
        onset_time = timestamps[timestamps['frame'] == slip_onset_frame]['elapsed_s'].values[0]
        ax1.set_title(f'Optical Flow Magnitude - Slip Onset at Frame {slip_onset_frame} (t={onset_time:.2f}s)')
    else:
        ax1.set_title('Optical Flow Magnitude - No Clear Slip Onset Detected')

    ax1.set_ylabel('Motion Magnitude (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot force data aligned with frames
    ax2_twin = ax2.twinx()

    if slip_curve is not None:
        # Map slip curve times to frame numbers
        for _, row in slip_curve.iterrows():
            elapsed = row['elapsed_s']
            # Find closest frame
            closest_frame_idx = np.argmin(np.abs(timestamps['elapsed_s'] - elapsed))
            closest_frame = timestamps.loc[closest_frame_idx, 'frame']

            ax2.plot(closest_frame, row['index_force_N'], 'bo', markersize=6)
            ax2_twin.plot(closest_frame, row['thumb_force_N'], 'ro', markersize=6)

        ax2.set_ylabel('Index Force (N)', color='b')
        ax2_twin.set_ylabel('Thumb Force (N)', color='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
    else:
        ax2.text(0.5, 0.5, 'No force data available\n(stable grasp trial)',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_ylabel('Force (N)')

    ax2.set_xlabel('Frame Number')
    ax2.grid(True, alpha=0.3)

    if slip_onset_frame:
        ax2.axvline(x=slip_onset_frame, color='orange', linestyle='--', linewidth=2)

    plt.tight_layout()

    # Save figure
    output_path = data_path / f"slip_onset_analysis_{camera}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Analysis plot saved to: {output_path}")
    plt.close()

    return slip_onset_frame, onset_time if slip_onset_frame else None


def extract_slip_onset_images(data_dir, slip_onset_frame, camera='cam1', context_frames=5):
    """Extract images before/after slip onset for visual inspection."""

    data_path = Path(os.path.expanduser(data_dir))
    cam_dir = data_path / f"camera_{camera}"
    output_dir = data_path / "slip_onset_frames"
    output_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Extracting frames around slip onset (frame {slip_onset_frame})")

    for offset in range(-context_frames, context_frames + 1):
        frame_num = slip_onset_frame + offset
        frame_path = cam_dir / f"frame_{frame_num:06d}.png"

        if frame_path.exists():
            img = cv2.imread(str(frame_path))

            # Add frame number text
            cv2.putText(img, f"Frame {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if offset == 0:
                cv2.putText(img, "SLIP ONSET", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            output_path = output_dir / f"frame_{frame_num:06d}_offset{offset:+d}.png"
            cv2.imwrite(str(output_path), img)

    print(f"[INFO] Extracted {2*context_frames + 1} frames to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_slip_onset.py <data_dir> [camera] [motion_threshold]")
        print("Example: python3 analyze_slip_onset.py ~/grape_unified_20260916_143409 cam1 2.0")
        sys.exit(1)

    data_dir = sys.argv[1]
    camera = sys.argv[2] if len(sys.argv) > 2 else 'cam1'
    motion_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    result = detect_slip_onset(
        data_dir=data_dir,
        camera=camera,
        start_frame=800,
        end_frame=1900,
        motion_threshold=motion_threshold,
        window_size=10
    )

    if result and result[0]:
        slip_frame, slip_time = result
        print(f"\n{'='*60}")
        print(f"[RESULT] Slip onset detected at:")
        print(f"  Frame: {slip_frame}")
        print(f"  Time: {slip_time:.2f} seconds")
        print(f"{'='*60}")

        # Extract frames for visual inspection
        extract_slip_onset_images(data_dir, slip_frame, camera, context_frames=5)
    else:
        print("\n[RESULT] No clear slip onset detected with current threshold")
        print("Try adjusting motion_threshold parameter (lower = more sensitive)")
