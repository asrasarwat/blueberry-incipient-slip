#!/usr/bin/env python3
"""
Verify slip onset labels and identify problematic trials.
"""
import csv
import os
import glob
import json

def check_trial(trial_dir):
    """Check a single trial for issues."""
    trial_name = os.path.basename(trial_dir)
    tactile_csv = os.path.join(trial_dir, 'tactile_data.csv')

    # Determine trial type
    has_slip_curve = os.path.exists(os.path.join(trial_dir, 'ground_truth_slip_curve.csv'))
    has_squeeze_curve = os.path.exists(os.path.join(trial_dir, 'squeeze_curve.csv'))

    trial_type = "SLIP" if has_slip_curve else "STABLE"

    if not os.path.exists(tactile_csv):
        return None

    # Read first 5 and last 5 rows to check forces
    with open(tactile_csv, 'r') as f:
        reader = list(csv.DictReader(f))

    first_5 = reader[:5]
    last_5 = reader[-5:]

    # Get force statistics
    index_forces = [float(row['index_force_N']) for row in reader]
    thumb_forces = [float(row['thumb_force_N']) for row in reader]

    first_index = index_forces[0]
    first_thumb = thumb_forces[0]
    max_index = max(index_forces)
    max_thumb = max(thumb_forces)
    min_index = min(index_forces)
    min_thumb = min(thumb_forces)

    # Check for anomalies
    issues = []
    if first_index < 0.2 and first_thumb < 0.2:
        issues.append("STARTS_TOO_LOW")
    if trial_type == "SLIP" and max_index < 1.5:
        issues.append("NEVER_GRIPPED")
    if max_thumb > 10.0 or max_index > 10.0:
        issues.append("SENSOR_SPIKE")

    return {
        'name': trial_name,
        'type': trial_type,
        'first_index': round(first_index, 3),
        'first_thumb': round(first_thumb, 3),
        'max_index': round(max_index, 2),
        'max_thumb': round(max_thumb, 2),
        'min_index': round(min_index, 2),
        'min_thumb': round(min_thumb, 2),
        'total_frames': len(reader),
        'issues': issues
    }

def main():
    trials = sorted(glob.glob('/media/dataset_storage/grape_unified_*'))

    print("="*100)
    print(f"{'Trial Name':<35} {'Type':<8} {'Start Forces':<18} {'Max Forces':<18} {'Issues'}")
    print("="*100)

    stable_count = 0
    slip_count = 0
    problematic = []

    for trial_dir in trials:
        info = check_trial(trial_dir)
        if not info:
            continue

        if info['type'] == 'STABLE':
            stable_count += 1
        else:
            slip_count += 1

        issues_str = ', '.join(info['issues']) if info['issues'] else 'OK'

        print(f"{info['name']:<35} {info['type']:<8} "
              f"I={info['first_index']:>4.2f} T={info['first_thumb']:>4.2f}     "
              f"I={info['max_index']:>4.1f} T={info['max_thumb']:>4.1f}     "
              f"{issues_str}")

        if info['issues']:
            problematic.append(info)

    print("="*100)
    print(f"\nSummary:")
    print(f"  Stable trials: {stable_count}")
    print(f"  Slip trials: {slip_count}")
    print(f"  Problematic trials: {len(problematic)}")

    if problematic:
        print(f"\n⚠️  Trials with issues:")
        for p in problematic:
            print(f"  - {p['name']}: {', '.join(p['issues'])}")

if __name__ == '__main__':
    main()
