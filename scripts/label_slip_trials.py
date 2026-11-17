#!/usr/bin/env python3
"""
Interactive tool to label slip onset frames by reviewing camera footage.
Displays video + tactile force plots, user marks exact slip onset.
"""
import cv2
import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN_SLIP_TRIALS = [
    'grape_unified_20260922_183343',
    'grape_unified_20260922_184136',
    'grape_unified_20260922_184632',
    'grape_unified_20260922_184953',
    'grape_unified_20260922_185310',
    'grape_unified_20260922_190315',
    'grape_unified_20260922_190639',
    'grape_unified_20260922_191020',
    'grape_unified_20260923_114950',
    'grape_unified_20260923_115730',
    'grape_unified_20260923_115932',
    'grape_unified_20260923_120329',
]

def find_trial_path(trial_name):
    """Find trial directory."""
    home_path = os.path.join(os.path.expanduser('~'), trial_name)
    storage_path = f'/media/dataset_storage/{trial_name}'

    if os.path.exists(home_path):
        return home_path
    elif os.path.exists(storage_path):
        return storage_path
    return None

def load_tactile_data(trial_dir):
    """Load tactile CSV."""
    tactile_csv = os.path.join(trial_dir, 'tactile_data.csv')
    if not os.path.exists(tactile_csv):
        return None

    with open(tactile_csv, 'r') as f:
        reader = list(csv.DictReader(f))

    return {
        'timestamp': np.array([float(row['timestamp']) for row in reader]),
        'index_force': np.array([float(row['index_force_N']) for row in reader]),
        'thumb_force': np.array([float(row['thumb_force_N']) for row in reader]),
    }

def label_trial(trial_name, labels_file):
    """Label slip onset for one trial."""
    trial_path = find_trial_path(trial_name)
    if not trial_path:
        print(f"Trial not found: {trial_name}")
        return None

    # Load existing labels
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}

    # Skip if already labeled
    if trial_name in labels:
        print(f"\n{trial_name}: Already labeled (slip_onset_frame={labels[trial_name]['slip_onset_frame']})")
        return labels[trial_name]['slip_onset_frame']

    print(f"\n{'='*80}")
    print(f"Labeling: {trial_name}")
    print(f"{'='*80}")

    # Load tactile data
    data = load_tactile_data(trial_path)
    if data is None:
        print("No tactile data found")
        return None

    # Check cameras
    cam1_dir = os.path.join(trial_path, 'camera_cam1')
    cam2_dir = os.path.join(trial_path, 'camera_cam2')

    cam1_exists = os.path.exists(cam1_dir)
    cam2_exists = os.path.exists(cam2_dir)

    print(f"Tactile samples: {len(data['timestamp'])}")
    print(f"Camera cam1: {'✓' if cam1_exists else '✗'}")
    print(f"Camera cam2: {'✓' if cam2_exists else '✗'}")

    # Plot tactile forces
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data['index_force'], label='Index', linewidth=2)
    plt.ylabel('Index Force (N)')
    plt.legend()
    plt.grid(True)
    plt.title(f'{trial_name} - Mark the slip onset frame')

    plt.subplot(2, 1, 2)
    plt.plot(data['thumb_force'], label='Thumb', linewidth=2, color='orange')
    plt.xlabel('Frame')
    plt.ylabel('Thumb Force (N)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    # User input
    print("\nInstructions:")
    print("  1. Review the camera footage and tactile plot")
    print("  2. Identify the frame where slip BEGINS (first visible motion)")
    print("  3. Enter the frame number, or 'skip' to skip this trial")

    slip_frame = input("\nSlip onset frame: ").strip()
    plt.close()

    if slip_frame.lower() == 'skip':
        return None

    try:
        slip_frame = int(slip_frame)
        labels[trial_name] = {
            'slip_onset_frame': slip_frame,
            'total_frames': len(data['timestamp'])
        }

        # Save immediately
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)

        print(f"✓ Saved: slip onset at frame {slip_frame}")
        return slip_frame

    except ValueError:
        print("Invalid input, skipping")
        return None

def main():
    labels_file = os.path.expanduser('~/slip_onset_labels.json')

    print("="*80)
    print("SLIP ONSET LABELING TOOL")
    print("="*80)
    print(f"Labels will be saved to: {labels_file}")
    print(f"Total trials to label: {len(CLEAN_SLIP_TRIALS)}")

    for trial_name in CLEAN_SLIP_TRIALS:
        label_trial(trial_name, labels_file)

    print("\n" + "="*80)
    print("Labeling complete!")
    print(f"Labels saved to: {labels_file}")
    print("="*80)

if __name__ == '__main__':
    main()
