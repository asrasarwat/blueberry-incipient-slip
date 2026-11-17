#!/usr/bin/env python3
"""
Train slip detection model on labeled dataset.
Uses LSTM or 1D CNN on tactile time-series features.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_dataset(dataset_dir='~/slip_dataset_prepared'):
    """Load prepared dataset."""
    dataset_dir = os.path.expanduser(dataset_dir)

    X = np.load(os.path.join(dataset_dir, 'features.npy'))
    y = np.load(os.path.join(dataset_dir, 'labels.npy'))

    with open(os.path.join(dataset_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    return X, y, metadata

def apply_slip_labels(y, metadata, labels_file='~/slip_onset_labels.json'):
    """
    Update labels with manually marked slip onset frames.
    Before slip onset: label=0 (stable)
    After slip onset: label=1 (slip)
    """
    labels_file = os.path.expanduser(labels_file)

    if not os.path.exists(labels_file):
        print("⚠️  No slip onset labels found. Using default labels.")
        return y

    with open(labels_file, 'r') as f:
        slip_labels = json.load(f)

    print("\nApplying slip onset labels...")
    y_refined = y.copy()
    idx = 0

    for trial in metadata:
        n = trial['n_samples']
        trial_name = trial['name']

        if trial['type'] == 'SLIP' and trial_name in slip_labels:
            slip_onset = slip_labels[trial_name]['slip_onset_frame']
            # Before slip: stable (0), after slip: slip (1)
            y_refined[idx:idx+slip_onset] = 0
            y_refined[idx+slip_onset:idx+n] = 1
            print(f"  ✓ {trial_name}: slip onset at frame {slip_onset}")

        idx += n

    return y_refined

def build_lstm_model(input_shape, units=64, dropout=0.3):
    """Build LSTM model for slip detection."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model

def build_cnn_model(input_shape, filters=32, dropout=0.3):
    """Build 1D CNN model for slip detection."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout),
        layers.Conv1D(filters * 2, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model

def prepare_sequences(X, y, sequence_length=30):
    """
    Create sliding window sequences for LSTM/CNN.
    Each sample becomes a sequence of past `sequence_length` timesteps.
    """
    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

def train_model(model_type='lstm', sequence_length=30):
    """Train slip detection model."""
    print("="*80)
    print(f"TRAINING SLIP DETECTION MODEL ({model_type.upper()})")
    print("="*80)

    # Load dataset
    X, y, metadata = load_dataset()
    print(f"\nLoaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Apply slip onset labels
    y = apply_slip_labels(y, metadata)

    # Create sequences
    print(f"\nCreating sequences (length={sequence_length})...")
    X_seq, y_seq = prepare_sequences(X, y, sequence_length)
    print(f"Sequence dataset: {X_seq.shape}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    print(f"\nTrain samples: {len(X_train)} (stable={np.sum(y_train==0)}, slip={np.sum(y_train==1)})")
    print(f"Test samples: {len(X_test)} (stable={np.sum(y_test==0)}, slip={np.sum(y_test==1)})")

    # Build model
    input_shape = (sequence_length, X.shape[1])

    if model_type == 'lstm':
        model = build_lstm_model(input_shape)
    elif model_type == 'cnn':
        model = build_cnn_model(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"\nModel architecture:")
    model.summary()

    # Train
    print(f"\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ],
        verbose=1
    )

    # Evaluate
    print(f"\nEvaluating on test set...")
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stable', 'Slip']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model_dir = os.path.expanduser('~/slip_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'slip_detector_{model_type}.keras')
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Precision')
    plt.plot(history.history['recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision/Recall')

    plt.tight_layout()
    plot_path = os.path.join(model_dir, f'training_history_{model_type}.png')
    plt.savefig(plot_path)
    print(f"✓ Training plot saved to: {plot_path}")
    plt.show()

    return model, history

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lstm', 'cnn'], default='lstm',
                        help='Model architecture')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Sequence length for time-series input')
    args = parser.parse_args()

    train_model(model_type=args.model, sequence_length=args.sequence_length)
