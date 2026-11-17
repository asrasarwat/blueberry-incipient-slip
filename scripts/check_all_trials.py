#!/usr/bin/env python3
"""
Check all trials across both locations.
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

    # Incomplete trial if missing both
    if not has_slip_curve and not has_squeeze_curve:
        return None  # Skip incomplete trials

    trial_type = "SLIP" if has_slip_curve else "STABLE"

    if not os.path.exists(tactile_csv):
        return None

    # Read first 5 and last 5 rows to check forces
    with open(tactile_csv, 'r') as f:
        reader = list(csv.DictReader(f))

    if len(reader) == 0:
        return None

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
        'path': trial_dir,
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
    # Check both locations
    home_trials = glob.glob(os.path.expanduser("~/grape_unified_*"))
    storage_trials = glob.glob('/media/dataset_storage/grape_unified_*')

    all_trials = sorted(home_trials + storage_trials)

    print("="*110)
    print(f"{'Trial Name':<35} {'Type':<8} {'Start Forces':<18} {'Max Forces':<18} {'Issues':<15} Location")
    print("="*110)

    stable_count = 0
    slip_count = 0
    problematic = []
    stable_ok = []
    slip_ok = []

    for trial_dir in all_trials:
        info = check_trial(trial_dir)
        if not info:
            continue

        if info['type'] == 'STABLE':
            stable_count += 1
            if not info['issues']:
                stable_ok.append(info['name'])
        else:
            slip_count += 1
            if not info['issues']:
                slip_ok.append(info['name'])

        issues_str = ', '.join(info['issues']) if info['issues'] else 'OK'
        location = 'home' if info['path'].startswith(os.path.expanduser("~/grape")) else 'storage'

        print(f"{info['name']:<35} {info['type']:<8} "
              f"I={info['first_index']:>4.2f} T={info['first_thumb']:>4.2f}     "
              f"I={info['max_index']:>4.1f} T={info['max_thumb']:>4.1f}     "
              f"{issues_str:<15} {location}")

        if info['issues']:
            problematic.append(info)

    print("="*110)
    print(f"\nSummary:")
    print(f"  Stable trials: {stable_count} ({len(stable_ok)} clean)")
    print(f"  Slip trials: {slip_count} ({len(slip_ok)} clean)")
    print(f"  Problematic trials: {len(problematic)}")

    print(f"\n✓ Clean STABLE trials: {len(stable_ok)}")
    for name in stable_ok:
        print(f"    {name}")

    print(f"\n✓ Clean SLIP trials: {len(slip_ok)}")
    for name in slip_ok:
        print(f"    {name}")

    if problematic:
        print(f"\n⚠️  Trials to EXCLUDE (with issues):")
        for p in problematic:
            print(f"  - {p['name']}: {', '.join(p['issues'])}")

if __name__ == '__main__':
    main()
