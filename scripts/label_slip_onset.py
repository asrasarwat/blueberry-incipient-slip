#!/usr/bin/env python3
"""
Auto-label slip onset frames using force threshold method.
Finds the first frame where BOTH tactile forces drop below thresholds.
"""

import csv
import os
import glob
import json

# Force thresholds for slip detection (Option A)
INDEX_THRESHOLD = 1.5  # N
THUMB_THRESHOLD = 0.5  # N

def find_slip_onset_frame(tactile_csv_path):
    """
    Find the frame number where slip begins based on force thresholds.
    Returns: (slip_frame, index_force, thumb_force, elapsed_s) or (None, None, None, None) if no slip detected
    """
    with open(tactile_csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            index_force = float(row['index_force_N'])
            thumb_force = float(row['thumb_force_N'])

            # Check if BOTH forces are below threshold
            if index_force < INDEX_THRESHOLD and thumb_force < THUMB_THRESHOLD:
                return (int(row['frame']), index_force, thumb_force, float(row['elapsed_s']))

    # No slip detected
    return None, None, None, None

def label_all_slip_trials(dataset_root='/media/dataset_storage'):
    """
    Label all slip trials in the dataset.
    """
    # Find all slip trial directories
    slip_dirs = sorted(glob.glob(f"{dataset_root}/grape_unified_*"))

    results = {}
    print(f"Found {len(slip_dirs)} trial directories")
    print(f"\nUsing thresholds: Index < {INDEX_THRESHOLD}N, Thumb < {THUMB_THRESHOLD}N")
    print("="*80)

    for trial_dir in slip_dirs:
        trial_name = os.path.basename(trial_dir)
        tactile_csv = os.path.join(trial_dir, 'tactile_data.csv')

        if not os.path.exists(tactile_csv):
            print(f"⚠️  {trial_name}: No tactile_data.csv found")
            continue

        slip_frame, index_f, thumb_f, elapsed = find_slip_onset_frame(tactile_csv)

        if slip_frame is not None:
            results[trial_name] = {
                'slip_onset_frame': slip_frame,
                'index_force_N': round(index_f, 3),
                'thumb_force_N': round(thumb_f, 3),
                'elapsed_time_s': round(elapsed, 2)
            }
            print(f"✓ {trial_name}: Frame {slip_frame:4d} @ {elapsed:.1f}s (I={index_f:.2f}N, T={thumb_f:.2f}N)")
        else:
            results[trial_name] = {
                'slip_onset_frame': None,
                'note': 'No slip detected (forces stayed above threshold)'
            }
            print(f"✗ {trial_name}: No slip detected")

    # Save results to JSON
    output_path = os.path.join(dataset_root, 'slip_onset_labels.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*80)
    print(f"\nLabels saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total trials: {len(results)}")
    print(f"  Slip detected: {sum(1 for v in results.values() if v.get('slip_onset_frame') is not None)}")
    print(f"  No slip: {sum(1 for v in results.values() if v.get('slip_onset_frame') is None)}")

    return results

if __name__ == '__main__':
    results = label_all_slip_trials()
