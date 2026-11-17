import os
#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Slip Detection Models
Generates all figures for final report:
- Confusion matrices (binary + 3-class)
- ROC curves
- Feature importance
- Force-time comparison plots
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Import train_test_split at top
from sklearn.model_selection import train_test_split

print("="*70)
print("SLIP DETECTION MODEL EVALUATION")
print("="*70)

# ============================================================================
# 1. 3-CLASS MODEL EVALUATION (Primary focus - only available model)
# ============================================================================
print("\n[1/5] Loading 3-class model and dataset...")

df_3class = pd.read_csv('/media/dataset_storage/toy/sept/dataset_3class.csv')
with open(os.path.expanduser("~/slip_model_3class.pkl"), 'rb') as f:
    model_3class = pickle.load(f)

X_3c = df_3class.drop('label', axis=1)
y_3c = df_3class['label']

_, X_test_3c, _, y_test_3c = train_test_split(
    X_3c, y_3c, test_size=0.3, random_state=42, stratify=y_3c
)

y_pred_3c = model_3class.predict(X_test_3c)

print("✓ 3-class model loaded")
print(f"  Test samples: {len(y_test_3c)}")

# Confusion matrix
cm_3c = confusion_matrix(y_test_3c, y_pred_3c)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_3c, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Stable', 'Incipient Slip', 'Full Slip'],
            yticklabels=['Stable', 'Incipient Slip', 'Full Slip'],
            cbar_kws={'label': 'Number of Samples'},
            annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=2, linecolor='white')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('3-Class Slip Detection Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_3class_confusion.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_3class_confusion.png")
plt.close()

# Classification report
print("\n" + "="*70)
print("3-CLASS CLASSIFICATION REPORT")
print("="*70)
print(classification_report(
    y_test_3c, y_pred_3c,
    target_names=['Stable', 'Incipient', 'Full Slip']
))

# ============================================================================
# 3. FEATURE IMPORTANCE
# ============================================================================
print("\n[2/5] Generating feature importance plots...")

# 3-class model features
feat_importance_3c = pd.DataFrame({
    'feature': X_3c.columns,
    'importance': model_3class.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
top_10_3c = feat_importance_3c.head(10)
bars = ax.barh(range(len(top_10_3c)), top_10_3c['importance'], color='darkorange', edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(top_10_3c)))
ax.set_yticklabels(top_10_3c['feature'], fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Feature Importance Score', fontsize=14, fontweight='bold')
ax.set_title('Top 10 Most Important Features - 3-Class Slip Detection Model', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, top_10_3c['importance'])):
    ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_feature_importance_3class.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_feature_importance_3class.png")
plt.close()

# ============================================================================
# 4. FORCE-TIME COMPARISON PLOTS
# ============================================================================
print("\n[4/6] Generating force-time comparison plots...")

# Sample trajectories from each class
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

class_names = ['Stable Grasp', 'Incipient Slip', 'Full Slip']
colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
    # Get sample from this class
    sample_data = df_3class[df_3class['label'] == cls_idx].iloc[:100]  # First 100 frames

    if len(sample_data) > 0:
        time_axis = np.arange(len(sample_data)) / 30.0  # Assuming ~30 Hz

        # Plot normal force
        axes[cls_idx].plot(time_axis, sample_data['thumb_force'],
                          label='Thumb Force', color=color, alpha=0.9, linewidth=2.5)
        axes[cls_idx].plot(time_axis, sample_data['index_force'],
                          label='Index Force', color=color, alpha=0.6, linewidth=2.5, linestyle='--')

        axes[cls_idx].set_ylabel('Normal Force (N)', fontsize=13, fontweight='bold')
        axes[cls_idx].set_title(f'{cls_name}', fontsize=14, fontweight='bold', pad=10)
        axes[cls_idx].legend(loc='upper right', fontsize=11, framealpha=0.9)
        axes[cls_idx].grid(alpha=0.4, linestyle='--', linewidth=0.8)
        axes[cls_idx].set_axisbelow(True)

        # Set y-axis limits for consistency
        axes[cls_idx].set_ylim(-0.5, 4.0)

axes[2].set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
fig.suptitle('Force Trajectories Across Different Grasp States', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/fig_force_time_comparison.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_force_time_comparison.png")
plt.close()

# ============================================================================
# 5. MODEL COMPARISON SUMMARY
# ============================================================================
print("\n[3/5] Generating model comparison summary...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics_data = {
    'Model': ['3-Class Random Forest'],
    'Accuracy': [accuracy_score(y_test_3c, y_pred_3c)],
    'Precision (weighted)': [precision_score(y_test_3c, y_pred_3c, average='weighted')],
    'Recall (weighted)': [recall_score(y_test_3c, y_pred_3c, average='weighted')],
    'F1-Score (weighted)': [f1_score(y_test_3c, y_pred_3c, average='weighted')]
}

metrics_df = pd.DataFrame(metrics_data)

# Format percentages
metrics_df_display = metrics_df.copy()
for col in ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1-Score (weighted)']:
    metrics_df_display[col] = metrics_df_display[col].apply(lambda x: f'{x*100:.2f}%')

fig, ax = plt.subplots(figsize=(12, 2.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df_display.values, colLabels=metrics_df_display.columns,
                cellLoc='center', loc='center',
                bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color header
for i in range(len(metrics_df_display.columns)):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# Color data row
for i in range(len(metrics_df_display.columns)):
    table[(1, i)].set_facecolor('#ecf0f1')
    table[(1, i)].set_text_props(weight='bold', fontsize=11)

# Add borders
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(1.5)

plt.title('3-Class Slip Detection Model - Performance Metrics', fontsize=14, weight='bold', pad=15)
plt.savefig(os.path.expanduser("~/fig_model_comparison.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: fig_model_comparison.png")
plt.close()

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================
print("\n[4/5] Generating summary report...")

summary_text = f"""
{'='*70}
SLIP DETECTION PROJECT - EVALUATION SUMMARY
{'='*70}

3-CLASS MODEL (Stable / Incipient / Full Slip)
  • Accuracy: {accuracy_score(y_test_3c, y_pred_3c)*100:.2f}%
  • Weighted Precision: {precision_score(y_test_3c, y_pred_3c, average='weighted')*100:.2f}%
  • Weighted Recall: {recall_score(y_test_3c, y_pred_3c, average='weighted')*100:.2f}%
  • Weighted F1-Score: {f1_score(y_test_3c, y_pred_3c, average='weighted')*100:.2f}%

GENERATED FIGURES:
  1. fig_3class_confusion.png - 3-class confusion matrix
  2. fig_feature_importance_3class.png - Feature importance
  3. fig_force_time_comparison.png - Force trajectories by class
  4. fig_model_comparison.png - Performance metrics table

All figures saved to: ~/
"""

print(summary_text)

# Save summary to file
with open(os.path.expanduser("~/evaluation_summary.txt"), 'w') as f:
    f.write(summary_text)

print("\n✓ Summary saved: evaluation_summary.txt")
print("\n" + "="*70)
print("EVALUATION COMPLETE! All figures ready for final report.")
print("="*70)
