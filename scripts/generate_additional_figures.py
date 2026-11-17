import os
#!/usr/bin/env python3
"""
Generate Additional Publication Figures for Slip Detection
- t-SNE feature visualization
- ROC curves (one-vs-rest)
- Learning curves
- Class distribution
- Per-class precision/recall/F1 bar charts
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.manifold import TSNE
import seaborn as sns

# Set publication-quality style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
sns.set_palette("husl")

print("="*70)
print("GENERATING ADDITIONAL PUBLICATION FIGURES")
print("="*70)

# Load data and model
print("\n[1/6] Loading dataset and model...")
df_3class = pd.read_csv('/media/dataset_storage/toy/sept/dataset_3class.csv')
with open(os.path.expanduser("~/slip_model_3class.pkl"), 'rb') as f:
    model_3class = pickle.load(f)

X = df_3class.drop('label', axis=1)
y = df_3class['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

y_pred = model_3class.predict(X_test)
y_pred_proba = model_3class.predict_proba(X_test)

class_names = ['Stable', 'Incipient Slip', 'Full Slip']
colors = ['#2ecc71', '#f39c12', '#e74c3c']

print("✓ Data loaded")
print(f"  Training samples: {len(y_train)}")
print(f"  Test samples: {len(y_test)}")

# ============================================================================
# 1. t-SNE FEATURE VISUALIZATION
# ============================================================================
print("\n[2/6] Generating t-SNE visualization...")

# Use test set for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_test_tsne = tsne.fit_transform(X_test)

fig, ax = plt.subplots(figsize=(10, 8))

for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
    mask = y_test == cls_idx
    ax.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1],
              c=color, label=cls_name, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE Component 1', fontsize=13, fontweight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=13, fontweight='bold')
ax.set_title('t-SNE Visualization of Feature Space\n3-Class Slip Detection',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_tsne_visualization.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_tsne_visualization.png")
plt.close()

# ============================================================================
# 2. ROC CURVES (ONE-VS-REST)
# ============================================================================
print("\n[3/6] Generating ROC curves...")

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

fig, ax = plt.subplots(figsize=(10, 8))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2.5,
           label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

# Plot diagonal
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax.set_title('ROC Curves - One-vs-Rest Classification\n3-Class Slip Detection Model',
            fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_roc_curves.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_roc_curves.png")
plt.close()

# ============================================================================
# 3. PER-CLASS PERFORMANCE BAR CHART
# ============================================================================
print("\n[4/6] Generating per-class performance metrics...")

precision_per_class = precision_score(y_test, y_pred, average=None)
recall_per_class = recall_score(y_test, y_pred, average=None)
f1_per_class = f1_score(y_test, y_pred, average=None)

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width, precision_per_class*100, width, label='Precision',
              color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, recall_per_class*100, width, label='Recall',
              color='#e74c3c', edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, f1_per_class*100, width, label='F1-Score',
              color='#2ecc71', edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Grasp State', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Performance Metrics\n3-Class Slip Detection Model',
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=12)
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.set_ylim([0, 105])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_per_class_metrics.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_per_class_metrics.png")
plt.close()

# ============================================================================
# 4. CLASS DISTRIBUTION
# ============================================================================
print("\n[5/6] Generating class distribution plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training set distribution
train_counts = [sum(y_train == i) for i in range(3)]
axes[0].bar(range(3), train_counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(class_names, fontsize=11)
axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
axes[0].set_title('Training Set Distribution', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Add count labels
for i, count in enumerate(train_counts):
    axes[0].text(i, count, str(count), ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# Test set distribution
test_counts = [sum(y_test == i) for i in range(3)]
axes[1].bar(range(3), test_counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(class_names, fontsize=11)
axes[1].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
axes[1].set_title('Test Set Distribution', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add count labels
for i, count in enumerate(test_counts):
    axes[1].text(i, count, str(count), ha='center', va='bottom',
                fontsize=12, fontweight='bold')

fig.suptitle('Dataset Class Distribution\n70% Training / 30% Testing Split',
            fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_class_distribution.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_class_distribution.png")
plt.close()

# ============================================================================
# 5. LEARNING CURVE
# ============================================================================
print("\n[6/6] Generating learning curve...")

train_sizes, train_scores, val_scores = learning_curve(
    model_3class, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42, scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(train_sizes, train_mean, 'o-', color='#3498db', linewidth=2.5,
        markersize=8, label='Training Score')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.2, color='#3498db')

ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', linewidth=2.5,
        markersize=8, label='Cross-validation Score')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.2, color='#e74c3c')

ax.set_xlabel('Training Set Size', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy Score', fontsize=13, fontweight='bold')
ax.set_title('Learning Curve - Random Forest Classifier\n3-Class Slip Detection',
            fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_ylim([0.5, 1.05])

plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_learning_curve.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_learning_curve.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ALL ADDITIONAL FIGURES GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Figures:")
print("  1. fig_tsne_visualization.png - Feature space clustering")
print("  2. fig_roc_curves.png - ROC curves with AUC scores")
print("  3. fig_per_class_metrics.png - Precision/Recall/F1 per class")
print("  4. fig_class_distribution.png - Training/test set distribution")
print("  5. fig_learning_curve.png - Model learning curve")
print("\nAll figures saved to: ~/")
print("="*70)
