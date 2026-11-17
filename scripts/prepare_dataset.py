#!/usr/bin/env python3
"""
Dataset preparation script for slip detection training.
Filters clean trials, extracts features, and prepares training data.
"""
import csv
import os
import glob
import json
import numpy as np
from pathlib import Path

# Clean trials identified by check_all_trials.py
CLEAN_STABLE = [
    'grape_unified_20260922_160537',
    'grape_unified_20260922_160859',
    'grape_unified_20260922_161356',
    'grape_unified_20260922_161634',
    'grape_unified_20260922_162012',
    'grape_unified_20260922_164858',
    'grape_unified_20260922_172705',
    'grape_unified_20260922_173141',
    'grape_unified_20260923_112555',
    'grape_unified_20260923_113533',
    'grape_unified_20260923_114046',
    'grape_unified_20260923_114414',
]

CLEAN_SLIP = [
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
    """Find trial directory in either home or storage."""
    home_path = os.path.join(os.path.expanduser('~'), trial_name)
    storage_path = f'/media/dataset_storage/{trial_name}'

    if os.path.exists(home_path):
        return home_path
    elif os.path.exists(storage_path):
        return storage_path
    else:
        return None

def load_tactile_data(trial_dir):
    """Load tactile sensor data from trial."""
    tactile_csv = os.path.join(trial_dir, 'tactile_data.csv')

    if not os.path.exists(tactile_csv):
        return None

    with open(tactile_csv, 'r') as f:
        reader = list(csv.DictReader(f))

    data = {
        'timestamp': np.array([float(row['timestamp']) for row in reader]),
        'index_force': np.array([float(row['index_force_N']) for row in reader]),
        'thumb_force': np.array([float(row['thumb_force_N']) for row in reader]),
        'index_x': np.array([float(row['index_shear_x']) for row in reader]),
        'index_y': np.array([float(row['index_shear_y']) for row in reader]),
        'thumb_x': np.array([float(row['thumb_shear_x']) for row in reader]),
        'thumb_y': np.array([float(row['thumb_shear_y']) for row in reader]),
    }

    return data

def compute_features(data, window_size=10):
    """
    Compute time-series features for slip detection.
    Returns features for each timestep (using past window_size samples).
    """
    n = len(data['timestamp'])
    features = []

    for i in range(window_size, n):
        window = slice(i - window_size, i)

        # Force statistics
        idx_force_mean = np.mean(data['index_force'][window])
        idx_force_std = np.std(data['index_force'][window])
        idx_force_delta = data['index_force'][i-1] - data['index_force'][i-window_size]

        thumb_force_mean = np.mean(data['thumb_force'][window])
        thumb_force_std = np.std(data['thumb_force'][window])
        thumb_force_delta = data['thumb_force'][i-1] - data['thumb_force'][i-window_size]

        # Shear statistics
        idx_shear_mag = np.sqrt(data['index_x'][i-1]**2 + data['index_y'][i-1]**2)
        thumb_shear_mag = np.sqrt(data['thumb_x'][i-1]**2 + data['thumb_y'][i-1]**2)

        # Shear velocity (change over window)
        idx_shear_delta = np.sqrt(
            (data['index_x'][i-1] - data['index_x'][i-window_size])**2 +
            (data['index_y'][i-1] - data['index_y'][i-window_size])**2
        )

        thumb_shear_delta = np.sqrt(
            (data['thumb_x'][i-1] - data['thumb_x'][i-window_size])**2 +
            (data['thumb_y'][i-1] - data['thumb_y'][i-window_size])**2
        )

        features.append([
            idx_force_mean, idx_force_std, idx_force_delta,
            thumb_force_mean, thumb_force_std, thumb_force_delta,
            idx_shear_mag, thumb_shear_mag,
            idx_shear_delta, thumb_shear_delta,
        ])

    return np.array(features)

def prepare_dataset(output_dir='~/slip_dataset_prepared'):
    """Prepare clean dataset for training."""
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("DATASET PREPARATION")
    print("=" * 80)

    all_features = []
    all_labels = []
    trial_metadata = []

    # Process STABLE trials (label = 0)
    print(f"\nProcessing {len(CLEAN_STABLE)} STABLE trials...")
    for trial_name in CLEAN_STABLE:
        trial_path = find_trial_path(trial_name)
        if not trial_path:
            print(f"  ⚠️  {trial_name}: NOT FOUND")
            continue

        data = load_tactile_data(trial_path)
        if data is None:
            print(f"  ⚠️  {trial_name}: NO TACTILE DATA")
            continue

        features = compute_features(data)
        labels = np.zeros(len(features), dtype=int)  # All stable

        all_features.append(features)
        all_labels.append(labels)
        trial_metadata.append({
            'name': trial_name,
            'type': 'STABLE',
            'n_samples': len(features)
        })

        print(f"  ✓ {trial_name}: {len(features)} samples")

    # Process SLIP trials (label = 1, but needs manual labeling)
    print(f"\nProcessing {len(CLEAN_SLIP)} SLIP trials...")
    print("NOTE: Slip onset frames need manual labeling from camera footage")

    for trial_name in CLEAN_SLIP:
        trial_path = find_trial_path(trial_name)
        if not trial_path:
            print(f"  ⚠️  {trial_name}: NOT FOUND")
            continue

        data = load_tactile_data(trial_path)
        if data is None:
            print(f"  ⚠️  {trial_name}: NO TACTILE DATA")
            continue

        features = compute_features(data)
        # For now, label everything as slip (will refine with labeling tool)
        labels = np.ones(len(features), dtype=int)

        all_features.append(features)
        all_labels.append(labels)
        trial_metadata.append({
            'name': trial_name,
            'type': 'SLIP',
            'n_samples': len(features),
            'slip_onset_frame': None  # To be labeled
        })

        print(f"  ✓ {trial_name}: {len(features)} samples (needs slip onset labeling)")

    # Save dataset
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    np.save(os.path.join(output_dir, 'features.npy'), X)
    np.save(os.path.join(output_dir, 'labels.npy'), y)

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(trial_metadata, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Dataset saved to: {output_dir}")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Stable samples: {np.sum(y == 0)}")
    print(f"  Slip samples: {np.sum(y == 1)}")
    print(f"{'=' * 80}")

    return output_dir

if __name__ == '__main__':
    prepare_dataset()
