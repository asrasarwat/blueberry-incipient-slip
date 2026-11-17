# Tactile Detection of Incipient Slip in Robotic Blueberry Grasping — Dataset

Per-trial tactile, kinematic and camera-timing data for the paper:

> A. Sarwat and W. Lu, "Tactile Detection of Incipient Slip and Early Warning of
> Grasp Loss in Robotic Blueberry Grasping," *IEEE Robotics and Automation
> Letters*, under review, 2026.

Analysis code: <GITHUB URL>   ·   Licence: data CC BY 4.0, code MIT

---

## 1. Contents

| archive | size | what |
|---|---|---|
| `bbslip_trials.zip` | ~804 MB | 88 trial folders (tactile, kinematics, camera timing, events) |
| `bbslip_regrasp_closedloop.zip` | ~20 MB | closed-loop regrasp trials |
| `bbslip_contact_sheets.zip` | ~68 MB | one time-lapse sheet per trial, side camera |

Raw high-speed camera frames (~285 GB) are **not** included. They are available
from the corresponding author on reasonable request. Nothing in the paper
requires them: every reported number is computable from the files here.

Verify downloads against `SHA256SUMS.txt`:

    sha256sum -c SHA256SUMS.txt

---

## 2. Layout

    blueberry/
      size_14.0mm/
        trial_map.csv
        release01_trial_20261101_113604/
          tactile_data.csv            per-frame tactile features (~96 Hz)
          taxels.npz                  full force maps and marker flow fields
          slip_curve.csv              release ramp: joint opening vs. force
          squeeze_curve.csv           grip-setting ramp
          events.csv                  named events with timestamps
          joint_log.csv               16 commanded and 16 actual joint angles
          trigger_log.csv             hardware camera trigger log
          camera_cam1_timestamps.csv  per-frame times, top camera
          camera_cam2_timestamps.csv  per-frame times, side camera
          trial_info.json             trial configuration
          roi.json                    camera region of interest
          drop_camera.json            drop time read from the side camera
          slip_analysis.png           per-trial summary plot
      size_15.0mm/ ... size_21.0mm/

All times in seconds. `elapsed_s` is time from the start of the trial and is the
common clock across every file in a folder. `wall_time` is a Unix timestamp.

---

## 3. Which trials are analysed  (IMPORTANT)

The archive holds **88** trial folders. The paper analyses **70**. Both numbers
are correct: 18 folders are present but excluded, and all of them are listed
below so the counts can be checked by inspection.

| category | n |
|---|---|
| analysed (the paper's 70) | 70 |
| excluded by quality control (`flagged*`) | 13 |
| rejected during recording (`rejected01`) | 1 |
| spare trial, never analysed (16 mm) | 1 |
| excluded but **named `release`** (21 mm) | 3 |
| **total folders** | **88** |

The authoritative list of the 70 is the `trial` column of
`claim1_results/gross_slip_per_trial.csv` in the code repository.
`trial_map.csv` in each size folder also carries a `status` column.

**Warning — the 21 mm folder.** At every other size the excluded trials were
renamed `flagged*`. At 21 mm they were not. These three carry a `release`
prefix but are **not** part of the 70:

    release05_trial_20261106_130030
    release06_trial_20261106_130402
    release12_trial_20261106_135237

The other 18 non-analysed folders:

    flagged01_trial_20261029_203838     flagged02_trial_20261103_125554
    flagged01_trial_20261101_121449     flagged03_trial_20261101_131713
    flagged01_trial_20261101_130003     flagged03_trial_20261102_121344
    flagged01_trial_20261102_114025     flagged03_trial_20261103_132633
    flagged01_trial_20261102_130637     flagged04_trial_20261101_133817
    flagged02_trial_20261101_130900     flagged05_trial_20261101_135558
    flagged02_trial_20261102_114354     rejected01_trial_20261101_124648
    release10_trial_20261106_121831  (16 mm spare, logged as a spare, not analysed)

Exclusion criteria (paper, Section IV-C): the berry did not fall, the operator
touched it, or a fingertip delivered no fresh frame for more than 0.4 s between
the hold and the drop. Use the `*_stale` / `*_age_ms` columns to reproduce the
third criterion.

---

## 4. `taxels.npz` — the raw fields

`np.load(path)` gives, for a trial of `N` samples:

| key | shape | meaning |
|---|---|---|
| `index`, `thumb` | (N, 35, 20, 3) | normal-force map per fingertip |
| `index_flow`, `thumb_flow` | (N, 26, 14, 2) | marker displacement field, px |
| `index_field_init`, `thumb_field_init` | (26, 14, 2) | marker grid reference positions, px |
| `t` | (N,) | time, s |
| `phase` | (N,) str | trial phase |
| `seq`, `fseq` | (N, 2) | sensor frame sequence numbers (index, thumb) |
| `new`, `force_new` | (N, 2) | 1 if this sample carried a fresh frame |
| `stale` | (N, 2) | samples since the last fresh frame |

Only samples with `new == 1` are fresh; repeated frames must be dropped before
any rate is computed. This is how the paper's "fresh, non-repeated frames only"
rule is applied.

The stick ratio of Eq. (3) is recomputed from `*_flow` and `*_field_init` as the
fraction of in-contact markers displaced by less than `u0 = 0.5 px` from the
hold reference.

---

## 5. `tactile_data.csv` — derived per-frame signals

Already computed, at ~96 Hz. The columns the paper uses directly:

| column | paper symbol | meaning |
|---|---|---|
| `min_stick_frac` | `S(t)`, Eq. (3) | **stick ratio, the primary measure** — minimum over fingers |
| `index_stick_frac`, `thumb_stick_frac` | `S_k(t)` | stick ratio per fingertip |
| `min_dT_edge_centre` | `L(t)`, Eq. (4) | edge lag, minimum over fingers |
| `index_dT_edge_centre`, `thumb_dT_edge_centre` | `L_k(t)` | edge lag per fingertip |
| `index_dT_edge`, `index_dT_centre` | `T^e_k`, `T^c_k` | edge and centre translation |
| `index_Fz`, `thumb_Fz` | `f_k(t)` | normal force per fingertip, N |
| `index_force_N`, `thumb_force_N` | | total normal force per fingertip, N |
| `elapsed_s` | | trial clock, s |
| `phase` | | trial phase |

Frame-freshness columns, used by the quality-control rule:
`index_new`, `thumb_new`, `index_stale`, `thumb_stale`, `index_seq`,
`thumb_seq`, `index_age_ms`, `thumb_age_ms`, `index_read_ms`, `thumb_read_ms`,
and the `*_force_*` equivalents for the force stream.

Wrench and contact columns: `index_Fx/Fy/Fz`, `index_Tx/Ty/Tz`,
`index_cop_row`, `index_cop_col` (centre of pressure, grid units),
`index_contact_x/y`, `index_contact_area`, `index_area_rel`.

> **AUTHORS: confirm before publishing.** The following columns are present but
> their exact definitions are not stated in the paper, so they are described
> loosely here: `index_pressure`, `index_shear_N`, `index_sn_ratio`,
> `index_cop_drift`, `index_flow_centre`, `index_flow_edge_centre`,
> `motion_magnitude`, `motion_max`, `motion_mean`. Either define each one in
> this table or delete this paragraph and state that they are intermediate
> quantities not used in the paper.

---

## 6. Other files

**`trial_map.csv`** (one per size) — `folder, original, berry_id, diameter_mm,
drop_camera_s, status`. `berry_id` identifies the physical berry (e.g. `B14a`),
which is what the leave-one-berry-out validation groups on. At most three
trials share a `berry_id`.

**`slip_curve.csv`** — `tick, loosen_rad, index_Fz_N, thumb_Fz_N, elapsed_s`.
The release ramp: commanded joint opening against the resulting fingertip
forces.

**`squeeze_curve.csv`** — `step, extra_dip_rad, index_Fz_N, thumb_Fz_N,
elapsed_s`. The grip-setting ramp before the hold; the source of the grasp
stiffness in supplementary Table S3.

**`events.csv`** — `event, wall_time, elapsed_s`. Named events, including the
start of loosening.

**`joint_log.csv`** — `elapsed_s, phase, cmd_0..cmd_15, act_0..act_15`.
Commanded and actual angles for the 16 LEAP Hand joints, radians.

**`trigger_log.csv`** — `trigger, t_cam1, t_cam2, elapsed_s, phase`. The common
hardware trigger: 99.9 Hz, 0.6 ms jitter, 0.27 ms inter-camera offset.

**`camera_cam{1,2}_timestamps.csv`** — `frame, wall_time, elapsed_s, host_ts,
dev_ts, phase, saved`. cam1 is the top view, cam2 the side view. `saved`
indicates whether that frame was written to disk.

**`drop_camera.json`** — the drop time read from the side camera, with the
bracketing range and a note on how it was determined.

---

## 7. Constants needed to reproduce the paper

| quantity | value |
|---|---|
| camera scale | 29.45 px/mm (from the 30 mm sensor-pad width) |
| camera rate | 100 fps, common hardware trigger |
| gross-slip threshold | 0.3 mm sustained for 0.25 s (0.1–0.5 mm as sensitivity checks) |
| **camera-delay correction** | **−0.125 s on every camera event time** |
| stuck-marker threshold | `u0 = 0.5 px` |
| onset rule | `z* = 5`, sustained `T_min = 0.25 s` |
| causal alarm | onset + `T_min`; an online detector cannot fire earlier |
| predictor horizon | `H = 3 s` |
| tactile rate | median 95.8 Hz per fingertip (range 90.0–97.6) |

The camera-delay correction exists because gross slip is detected after a
trailing 0.25 s running median, which delays a crossing by about half its
window (measured: 120–125 ms, SD ≤ 15 ms on synthetic ramps). **Omitting it
changes the reported lead times.** Tracking precision during the hold was
0.6 µm (median), 2.6 µm (95th percentile). Camera-to-tactile synchronisation
was not measured.

---

## 8. Contact sheets

`bbslip_contact_sheets.zip` holds one sheet per trial: side-camera frames every
0.5 s in a grid, elapsed time overlaid on each tile. They join to trial folders
on the timestamp in the filename:

    sheet_trial_20261101_121449.jpg  ->  size_14.0mm/flagged01_trial_20261101_121449/

`as_recorded/` is unmodified. `enhanced/` has brightness and contrast raised for
readability only (`-auto-level -brightness-contrast 15x20`); use `as_recorded/`
for anything measured. Sheets are included for excluded trials too.

---

## 9. Citation

    <BIBTEX ONCE THE PAPER HAS A DOI>

Data licensed CC BY 4.0. Analysis code at <GITHUB URL> under MIT.
