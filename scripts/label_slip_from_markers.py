#!/usr/bin/env python3
"""
label_slip_from_markers.py
Automatically detect slip events by tracking colored markers on object surface.

Usage:
  python label_slip_from_markers.py --data_dir ~/grape_slip_color_20260913_123456
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class MarkerTracker:
    """Track colored markers using HSV color segmentation"""

    def __init__(self, color='red'):
        """
        Initialize marker tracker
        Args:
            color: 'red', 'green', 'blue', 'yellow', or 'custom'
        """
        self.color = color
        self.hsv_ranges = self._get_hsv_range(color)

    def _get_hsv_range(self, color):
        """Get HSV range for different marker colors"""
        ranges = {
            'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                    (np.array([170, 100, 100]), np.array([180, 255, 255]))],  # Red wraps around
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
            'orange': [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
        }
        return ranges.get(color, ranges['red'])

    def detect_marker(self, frame):
        """
        Detect marker in frame and return centroid
        Returns:
            (x, y, area) or None if not found
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask by combining all HSV ranges for this color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour (assume it's the marker)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Filter tiny detections (noise)
        if area < 20:  # Minimum 20 pixels
            return None

        # Calculate centroid
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy, area)


def load_camera_data(data_dir, cam_name):
    """Load camera frames and timestamps"""
    cam_dir = Path(data_dir) / f"camera_{cam_name}"
    timestamp_file = Path(data_dir) / f"camera_{cam_name}_timestamps.csv"

    if not cam_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    # Load timestamps
    timestamps = pd.read_csv(timestamp_file)

    # Get frame files
    frame_files = sorted(cam_dir.glob("frame_*.png"))

    return frame_files, timestamps


def track_markers_in_video(frame_files, marker_colors=['red', 'green']):
    """
    Track multiple markers across video frames

    Args:
        frame_files: List of frame file paths
        marker_colors: List of marker colors to track

    Returns:
        DataFrame with columns: frame, marker_id, x, y, area
    """
    trackers = [MarkerTracker(color) for color in marker_colors]

    results = []

    print(f"Tracking {len(marker_colors)} markers across {len(frame_files)} frames...")

    for frame_idx, frame_path in enumerate(tqdm(frame_files)):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        for marker_id, tracker in enumerate(trackers):
            detection = tracker.detect_marker(frame)
            if detection is not None:
                x, y, area = detection
                results.append({
                    'frame': frame_idx,
                    'marker_id': marker_id,
                    'marker_color': marker_colors[marker_id],
                    'x': x,
                    'y': y,
                    'area': area
                })

    return pd.DataFrame(results)


def compute_slip_events(marker_tracks, velocity_threshold=5.0, area_threshold=0.3):
    """
    Detect slip events from marker trajectories

    Args:
        marker_tracks: DataFrame with marker positions over time
        velocity_threshold: Pixels/frame threshold for sudden movement
        area_threshold: Fractional area change threshold (0.3 = 30% change)

    Returns:
        DataFrame with slip events: frame, slip_detected, slip_type, marker_id
    """
    slip_events = []

    for marker_id in marker_tracks['marker_id'].unique():
        marker_data = marker_tracks[marker_tracks['marker_id'] == marker_id].sort_values('frame')

        if len(marker_data) < 3:
            continue

        # Compute frame-to-frame displacement
        marker_data = marker_data.copy()
        marker_data['dx'] = marker_data['x'].diff()
        marker_data['dy'] = marker_data['y'].diff()
        marker_data['velocity'] = np.sqrt(marker_data['dx']**2 + marker_data['dy']**2)

        # Compute area change
        marker_data['area_change'] = marker_data['area'].pct_change().abs()

        # Detect slip events
        for idx, row in marker_data.iterrows():
            slip_detected = False
            slip_type = 'none'

            # Type 1: Sudden movement (marker moved quickly)
            if pd.notna(row['velocity']) and row['velocity'] > velocity_threshold:
                slip_detected = True
                slip_type = 'sudden_movement'

            # Type 2: Marker disappears/reappears (contact loss)
            if pd.notna(row['area_change']) and row['area_change'] > area_threshold:
                slip_detected = True
                slip_type = 'contact_change'

            slip_events.append({
                'frame': int(row['frame']),
                'marker_id': int(marker_id),
                'slip_detected': slip_detected,
                'slip_type': slip_type,
                'velocity': row['velocity'] if pd.notna(row['velocity']) else 0.0,
                'area_change': row['area_change'] if pd.notna(row['area_change']) else 0.0
            })

    slip_df = pd.DataFrame(slip_events)

    # Aggregate across all markers (if ANY marker slips, frame is labeled as slip)
    frame_labels = []
    for frame in range(int(slip_df['frame'].max()) + 1):
        frame_slips = slip_df[slip_df['frame'] == frame]
        if len(frame_slips) == 0:
            frame_labels.append({'frame': frame, 'slip_detected': False, 'slip_type': 'none'})
        else:
            any_slip = frame_slips['slip_detected'].any()
            slip_types = frame_slips[frame_slips['slip_detected']]['slip_type'].values
            slip_type = slip_types[0] if len(slip_types) > 0 else 'none'
            frame_labels.append({'frame': frame, 'slip_detected': any_slip, 'slip_type': slip_type})

    return pd.DataFrame(frame_labels), slip_df


def merge_with_tactile(slip_labels, tactile_csv, timestamps_csv):
    """
    Merge slip labels with tactile sensor data based on timestamps

    Returns:
        DataFrame with: frame, elapsed_s, slip_detected, index_force_N, thumb_force_N, ...
    """
    # Load data
    timestamps = pd.read_csv(timestamps_csv)
    tactile = pd.read_csv(tactile_csv)

    # Merge slip labels with timestamps
    merged = slip_labels.merge(timestamps, on='frame', how='left')

    # Merge with tactile data based on elapsed_s (closest match)
    if 'elapsed_s' in merged.columns and 'elapsed_s' in tactile.columns:
        # Find closest tactile sample for each frame
        merged['tactile_idx'] = merged['elapsed_s'].apply(
            lambda t: np.argmin(np.abs(tactile['elapsed_s'].values - t)) if pd.notna(t) else None
        )

        # Add tactile features
        for col in ['index_force_N', 'thumb_force_N', 'index_contact_area', 'thumb_contact_area']:
            if col in tactile.columns:
                merged[col] = merged['tactile_idx'].apply(
                    lambda idx: tactile.iloc[int(idx)][col] if pd.notna(idx) else np.nan
                )

    return merged


def visualize_slip_events(frame_files, marker_tracks, slip_labels, output_video, max_frames=500):
    """
    Create visualization video with marker tracking and slip annotations

    Args:
        frame_files: List of frame paths
        marker_tracks: DataFrame with marker positions
        slip_labels: DataFrame with slip events
        output_video: Output video path
        max_frames: Maximum frames to process (for speed)
    """
    print(f"Creating visualization video: {output_video}")

    # Get first frame to determine video size
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))

    # Color map for markers
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
    ]

    num_frames = min(len(frame_files), max_frames)

    for frame_idx in tqdm(range(num_frames), desc="Rendering video"):
        frame = cv2.imread(str(frame_files[frame_idx]))
        if frame is None:
            continue

        # Get markers for this frame
        frame_markers = marker_tracks[marker_tracks['frame'] == frame_idx]

        # Draw markers
        for _, marker in frame_markers.iterrows():
            marker_id = int(marker['marker_id'])
            x, y = int(marker['x']), int(marker['y'])
            color = colors[marker_id % len(colors)]

            # Draw circle
            cv2.circle(frame, (x, y), 10, color, 2)
            cv2.putText(frame, f"M{marker_id}", (x+15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check if slip detected in this frame
        frame_slip = slip_labels[slip_labels['frame'] == frame_idx]
        if len(frame_slip) > 0 and frame_slip.iloc[0]['slip_detected']:
            slip_type = frame_slip.iloc[0]['slip_type']
            # Draw red warning
            cv2.rectangle(frame, (10, 10), (w-10, 80), (0, 0, 255), 3)
            cv2.putText(frame, f"SLIP DETECTED: {slip_type}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Frame info
        cv2.putText(frame, f"Frame: {frame_idx}", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Video saved: {output_video}")


def plot_slip_analysis(slip_data_merged, output_dir):
    """Create analysis plots"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Force over time with slip events
    ax = axes[0]
    ax.plot(slip_data_merged['elapsed_s'], slip_data_merged['index_force_N'],
            label='Index Force', alpha=0.7)
    ax.plot(slip_data_merged['elapsed_s'], slip_data_merged['thumb_force_N'],
            label='Thumb Force', alpha=0.7)

    # Mark slip events
    slip_frames = slip_data_merged[slip_data_merged['slip_detected']]
    ax.scatter(slip_frames['elapsed_s'], slip_frames['index_force_N'],
              color='red', s=100, marker='x', label='Slip Event', zorder=5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Grip Force with Slip Events')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Contact area over time
    ax = axes[1]
    ax.plot(slip_data_merged['elapsed_s'], slip_data_merged['index_contact_area'],
            label='Index Area', alpha=0.7)
    ax.plot(slip_data_merged['elapsed_s'], slip_data_merged['thumb_contact_area'],
            label='Thumb Area', alpha=0.7)

    ax.scatter(slip_frames['elapsed_s'], slip_frames['index_contact_area'],
              color='red', s=100, marker='x', zorder=5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Contact Area (taxels)')
    ax.set_title('Contact Area with Slip Events')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Slip event distribution
    ax = axes[2]
    slip_counts = slip_data_merged.groupby('slip_type').size()
    slip_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
    ax.set_xlabel('Slip Type')
    ax.set_ylabel('Count')
    ax.set_title('Slip Event Distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plot_path = Path(output_dir) / 'slip_analysis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Analysis plot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Label slip events from colored markers')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to slip recording directory')
    parser.add_argument('--camera', type=str, default='cam1',
                       help='Which camera to use (cam1 or cam2)')
    parser.add_argument('--markers', nargs='+', default=['red', 'green'],
                       help='Marker colors to track')
    parser.add_argument('--velocity_threshold', type=float, default=5.0,
                       help='Velocity threshold for slip detection (pixels/frame)')
    parser.add_argument('--create_video', action='store_true',
                       help='Create visualization video')
    parser.add_argument('--max_video_frames', type=int, default=500,
                       help='Max frames for visualization video')

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print("\n" + "="*70)
    print("  SLIP DETECTION - MARKER TRACKING")
    print("="*70)
    print(f"  Data directory: {data_dir}")
    print(f"  Camera: {args.camera}")
    print(f"  Marker colors: {args.markers}")
    print(f"  Velocity threshold: {args.velocity_threshold} px/frame")
    print("="*70 + "\n")

    # Load camera data
    print("Loading camera data...")
    frame_files, timestamps = load_camera_data(data_dir, args.camera)
    print(f"  Loaded {len(frame_files)} frames")

    # Track markers
    marker_tracks = track_markers_in_video(frame_files, args.markers)
    print(f"  Tracked {len(marker_tracks)} marker detections")

    # Save marker tracks
    marker_track_file = data_dir / f'marker_tracks_{args.camera}.csv'
    marker_tracks.to_csv(marker_track_file, index=False)
    print(f"  Saved: {marker_track_file}")

    # Detect slip events
    print("\nDetecting slip events...")
    slip_labels, slip_details = compute_slip_events(
        marker_tracks,
        velocity_threshold=args.velocity_threshold
    )

    num_slips = slip_labels['slip_detected'].sum()
    print(f"  Detected {num_slips} slip frames out of {len(slip_labels)} total frames")
    print(f"  Slip rate: {100*num_slips/len(slip_labels):.1f}%")

    # Save slip labels
    slip_label_file = data_dir / f'slip_labels_{args.camera}.csv'
    slip_labels.to_csv(slip_label_file, index=False)
    print(f"  Saved: {slip_label_file}")

    # Save detailed slip events
    slip_detail_file = data_dir / f'slip_details_{args.camera}.csv'
    slip_details.to_csv(slip_detail_file, index=False)
    print(f"  Saved: {slip_detail_file}")

    # Merge with tactile data
    print("\nMerging with tactile data...")
    tactile_csv = data_dir / 'tactile_data.csv'
    timestamps_csv = data_dir / f'camera_{args.camera}_timestamps.csv'

    if tactile_csv.exists():
        slip_data_merged = merge_with_tactile(slip_labels, tactile_csv, timestamps_csv)
        merged_file = data_dir / f'slip_data_merged_{args.camera}.csv'
        slip_data_merged.to_csv(merged_file, index=False)
        print(f"  Saved: {merged_file}")

        # Create analysis plots
        print("\nCreating analysis plots...")
        plot_slip_analysis(slip_data_merged, data_dir)

    # Create visualization video
    if args.create_video:
        output_video = str(data_dir / f'slip_visualization_{args.camera}.mp4')
        visualize_slip_events(frame_files, marker_tracks, slip_labels,
                            output_video, max_frames=args.max_video_frames)

    print("\n" + "="*70)
    print("  SLIP LABELING COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Marker tracks: marker_tracks_{args.camera}.csv")
    print(f"  - Slip labels: slip_labels_{args.camera}.csv")
    print(f"  - Slip details: slip_details_{args.camera}.csv")
    if tactile_csv.exists():
        print(f"  - Merged data: slip_data_merged_{args.camera}.csv")
        print(f"  - Analysis plot: slip_analysis.png")
    if args.create_video:
        print(f"  - Visualization: slip_visualization_{args.camera}.mp4")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
