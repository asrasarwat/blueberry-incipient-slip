#!/usr/bin/env python3
"""
Batch process all grape grasp trials to detect slip onset and build ground truth labels.
Outputs a CSV with slip detection results for each trial.

Usage:
  /usr/share/bin/xensegui/bin/python3 batch_analyze_slip.py
"""

import os
import glob
import pandas as pd
import subprocess
import sys
from pathlib import Path

# Configuration
THRESHOLD = 0.3  # Motion threshold in pixels
CAMERA = 'cam1'
ANALYZE_SCRIPT = os.path.expanduser("~/grasp_dataset/standalone/analyze_slip_onset.py")
OUTPUT_CSV = os.path.expanduser("~/grasp_dataset/ground_truth_labels.csv")

def find_grape_trials():
    """Find all grape_unified_* directories in home and dataset_storage"""
    pattern1 = os.path.expanduser('~/grape_unified_*/')
    pattern2 = '/media/dataset_storage/grape_unified_*/'

    trials1 = glob.glob(pattern1)
    trials2 = glob.glob(pattern2)

    all_trials = trials1 + trials2
    return [Path(t) for t in sorted(all_trials)]

def run_slip_detection(trial_dir):
    """Run slip onset detection on a trial"""
    cmd = [
        '/usr/share/bin/xensegui/bin/python3',
        ANALYZE_SCRIPT,
        str(trial_dir),
        CAMERA,
        str(THRESHOLD)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return parse_result(result.stdout, trial_dir)
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}
    except Exception as e:
        return {'error': str(e)}

def parse_result(stdout, trial_dir):
    """Extract slip frame and time from output"""
    lines = stdout.split('\n')

    result = {
        'trial_dir': trial_dir.name,
        'slip_detected': False,
        'slip_frame': None,
        'slip_time_s': None,
        'motion_max': None,
        'motion_mean': None
    }

    for line in lines:
        if 'Frame:' in line:
            result['slip_detected'] = True
            result['slip_frame'] = int(line.split('Frame:')[1].strip())
        elif 'Time:' in line and 'seconds' in line:
            result['slip_time_s'] = float(line.split('Time:')[1].replace('seconds', '').strip())
        elif 'Motion range:' in line:
            parts = line.split('to')
            if len(parts) == 2:
                result['motion_max'] = float(parts[1].split('pixels')[0].strip())
        elif 'Motion mean:' in line:
            result['motion_mean'] = float(line.split('mean:')[1].split(',')[0].strip())

    # Get tactile force at slip if detected
    if result['slip_detected'] and result['slip_time_s']:
        tactile_data = get_tactile_at_slip(trial_dir, result['slip_time_s'])
        result.update(tactile_data)

    return result

def get_tactile_at_slip(trial_dir, slip_time):
    """Get tactile readings at slip onset time"""
    tactile_file = trial_dir / 'tactile_data.csv'

    if not tactile_file.exists():
        return {'index_force_N': None, 'thumb_force_N': None}

    try:
        df = pd.read_csv(tactile_file)
        # Find closest sample to slip time
        df['time_diff'] = abs(df['elapsed_s'] - slip_time)
        closest = df.loc[df['time_diff'].idxmin()]

        return {
            'index_force_N': round(closest['index_force_N'], 3),
            'thumb_force_N': round(closest['thumb_force_N'], 3)
        }
    except Exception as e:
        return {'index_force_N': None, 'thumb_force_N': None, 'tactile_error': str(e)}

def main():
    print("[INFO] Finding grape grasp trials...")
    trials = find_grape_trials()
    print(f"[INFO] Found {len(trials)} trials")

    if len(trials) == 0:
        print("[ERROR] No trials found matching ~/grape_unified_*/")
        sys.exit(1)

    results = []

    for i, trial_dir in enumerate(trials, 1):
        print(f"\n[{i}/{len(trials)}] Processing {trial_dir.name}...")
        result = run_slip_detection(trial_dir)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        elif result['slip_detected']:
            print(f"  Slip detected: frame {result['slip_frame']}, t={result['slip_time_s']:.2f}s")
        else:
            print(f"  No slip detected (max motion: {result.get('motion_max', 'N/A')} pixels)")

        results.append(result)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"[COMPLETE] Results saved to: {OUTPUT_CSV}")
    print(f"  Total trials: {len(results)}")
    print(f"  Slip detected: {df['slip_detected'].sum()}")
    print(f"  No slip: {(~df['slip_detected']).sum()}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
