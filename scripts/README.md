# Analysis code

Code for the paper *Tactile Detection of Incipient Slip and Early Warning of
Grasp Loss in Robotic Blueberry Grasping* (Sarwat & Lu, RA-L).

Dataset: <ZENODO DOI>  ·  Place or symlink the unpacked trials at
`~/grasp_data/blueberry/` (the layout is described in the dataset README).

## Requirements

    python >= 3.8, numpy, scipy, pandas, scikit-learn, matplotlib
    statsmodels   (berry-clustered GEE tests only)

## Where things go

Scripts read trials from `~/grasp_data/blueberry/` and write results into
`~/claim1_results/` and `~/model_results/`. Those two folders are included in
this repository under `results/`, so every figure and table can be regenerated
without recomputing from the raw trials.

## Pipeline

    1. per-trial processing    postprocess_trial.py, convert_raw.py, slip_features.py
    2. camera ground truth     track_camera_v3.py, check_drop.py, ruler_calibrate.py
    3. onsets and baselines    analyze_slip_onset.py, entropy_baseline.py, dong_baseline.py
    4. claim 1 statistics      claim1_check_7.py, gross_slip_check_7.py, paper_tests.py
    5. learned predictor       train_deploy_model_v4.py, export_model_np_v4.py
    6. figures                 fig_*.py

## Script map

### Main-text figures

| script | output | paper |
|---|---|---|
| `fig_slip_field_v2.py` | `fig_slip_field.{png,pdf}` | **Fig. 3** partial slip in the marker field |
| `fig_lead_thresholds_v2.py` | `fig_lead_thresholds_v2.{png,pdf}` | **Fig. 4** onset timing vs. gross slip |
| `fig_roc_ablation.py` | `fig_roc_ablation.{png,pdf}` | **Fig. 5** held-out ROC and ablations |
| `fig_closed_loop.py` | `fig_closed_loop.{png,pdf}` | **Fig. 6** closed-loop regrasp |

Figs. 1 and 2 are a photograph and a diagram; no script produces them.

**Note on Fig. 4.** The script must apply the −0.125 s camera-delay correction
to every `cam_*` column before computing leads, or the counts and medians will
not match Tables III and the text. See §7 of the dataset README.

### Supplementary figures

| script | output | paper |
|---|---|---|
| `fig_importance_learning.py` | `fig_importance_learning.{png,pdf}` | **Fig. S1** feature importance, learning curve |
| `threshold_heatmap_7.py` | `threshold_grid.csv`, `threshold_heatmap.*` | **Fig. S2** onset-rule robustness |
| `fig_rot_warn.py` | `fig_rotation_warning.{png,pdf}` | **Fig. S4** rotation and warning per size |
| `fig_model_extra.py` | `figS_model_extra.png` | **Fig. S7** event-level performance, calibration |

### Incipient slip — measures and statistics

| script | writes | used for |
|---|---|---|
| `analyze_slip_onset.py` | — | stick-ratio onsets under the `z*=5`, `T_min=0.25 s` rule |
| `entropy_baseline.py` | `claim1_results/entropy_baseline_per_trial.csv` | displacement entropy (Yuan *et al.*) |
| `dong_baseline.py` | `claim1_results/dong_baseline_per_trial.csv` | rigid-body residual (Dong *et al.*) |
| `gross_slip_check_7.py` | `gross_slip_per_trial.csv` | camera gross slip at 0.1–0.5 mm; **the 70-trial list** |
| `claim1_check_7.py` | `claim1_per_trial.csv` | primary hypothesis test |
| `paper_tests.py` | — | sign tests, Wilcoxon, Spearman reported in Results |
| `review_per_finger.py` | `claim1_results/per_finger_aggregation.csv` | **Table S10** index vs. thumb vs. min vs. mean |
| `review_rigid_sweep.py` | — | **Table S11** 28-setting rigid-body threshold sweep |
| `supported_motion_check.py` | `claim1_results/supported_vs_release_motion.csv` | supported controls, berry motion at onset |
| `control_stick_check.py` | — | no-release controls, hold false alarms |
| `stiffness_proxy.py` | `claim1_results/stiffness_proxy.csv` | **Table S3** grasp stiffness by size |
| `berry_rotation.py`, `tactile_rotation.py` | `claim1_results/tactile_rotation.csv` | roll vs. slide; **Fig. S3** |
| `failure_analysis.py` | `claim1_results/failure_analysis.csv` | which trials each measure missed |
| `lead_time_protocol.py` | — | lead time vs. release rate and grip force |

### Learned grasp-loss predictor

| script | writes | used for |
|---|---|---|
| `train_deploy_model_v4.py` | trained model | gradient-boosted trees, 200 × depth 3, lr 0.05 |
| `export_model_np_v4.py` | NumPy evaluator | deployment form used online |
| `roc_preds_v4.py` | `model_results/roc_preds_v4.csv` | **Fig. 5a,b** LOBO / LOSO ROC |
| `ablation_v4.py` | `model_results_ablation_v4.csv` | **Table S8** feature ablation |
| `importance_learning_v4.py` | `model_results/learning_curve.csv`, `perm_importance*.csv` | **Fig. S1** |
| `warn_per_trial_v4.py` | `model_results/warn_per_trial_loso.csv` | warning times, drops detected |
| `auc_ci_v4.py` | — | berry-level bootstrap AUC intervals |
| `nopull_fa_v4.py` | `model_results/nopull_fa_v4.csv` | false alarms on no-release controls |
| `model_extra_compute.py`, `model_extra_plot.py` | `model_results/extra_*.csv` | **Fig. S7** |
| `protocol_effects.py` | `model_results/protocol_effects.csv` | **Table S2** protocol effects |
| `review_force_baselines.py` | — | **Table S12** force-only baseline comparison |

### Tables

| script | paper |
|---|---|
| `make_table_by_size_v2.py` | **Table III** onsets per berry size |
| `make_dataset_table.py` | **Table S4** dataset composition |

### Data collection and online loop

| script | role |
|---|---|
| `unified_grasp_capture_v6_28.py` | trial capture: grip setting, hold, timed release, camera trigger |
| `realtime_slip_detector_v5.py` | online detector used in the closed-loop trials |
| `make_detector_v5.py` | builds the deployed detector (pinch pose matched to data collection) |
| `contact_sheet.py` | the per-trial contact sheets in the dataset record |
| `postprocess_trial.py`, `convert_raw.py`, `slip_features.py` | per-trial processing into `tactile_data.csv` |
| `track_camera_v3.py`, `check_drop.py` | side-camera berry tracking and drop time |
| `ruler_calibrate.py` | the 29.45 px/mm scale |

`trials.csv` and `control_trials.csv` list the trials each batch script processes.

## Reproducing the headline number

    python scripts/standalone/gross_slip_check_7.py      # camera gross slip per trial
    python scripts/standalone/analyze_slip_onset.py      # stick-ratio onsets
    python scripts/standalone/claim1_check_7.py          # 45 of 68, p = 0.010
    python scripts/standalone/fig_lead_thresholds_v2.py  # Fig. 4

Or, from the shipped `results/`, skip straight to the figure scripts.

## Not included

`scripts/_quarantine/` is not part of the release: superseded script versions
kept only for provenance. The versions above are the ones the paper used.
