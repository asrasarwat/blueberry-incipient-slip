#!/usr/bin/env python3
"""
unified_grasp_capture_v6.py  -  capture protocol for INCIPIENT-slip detection

Goal: show that the Xense signal changes at or before the camera-labelled incipient-slip onset,
when slip is NOT caused by the robot opening its own grip.

v6_28 (1 Nov): --released-s N sets how long the grip force must stay low before loosening stops ("released"),
  default 15 s as before (logged as released_loosen_s). Trial 112941 (14 mm): "released" stopped at 0.226 rad while
  the berry was still resting between the pads at the end of the recording; use --released-s 60 for small berries.
v6_27 (1 Nov): after the force-based drop detection, loosening continues DROP_CONFIRM_S more (normal speed, still
  recording) before stopping. Trial 111023: "dropped" fired at 52.4 s while camera 2 showed the berry still resting
  between the pads at 54.4 s (near-zero force on both), so the real fall was never recorded.
v6_26 (29 Oct): at the start of every trial, with the hand open and nothing touching, both sensors get
  Sensor.resetReferenceImage() (readers paused meanwhile) before the unloaded |Fz| is measured; result logged as
  reference_reset. --no-reset-ref skips it. Why: the thumb kept a growing force offset (still ~1.5 N with no berry
  after the drop in 201510; 4-39 N on a blueberry in 200120-202026) although the gel was clean and replugged.
v6_25 (29 Oct): glitch guard also inside grip_set (0.5-s mean |Fz| of either finger above FZ_GLITCH_N -> result
  "glitch" -> abort, hand opened). Trial 200842: the mean force jumped 2.1 -> 9.7 N in 2 s during grip_set on a
  0.005 rad step, the controller then opened 0.24 rad in big steps, hunted for 30 s and timed out at 0.67 N.
v6_24 (29 Oct): glitch guard also during the hold (1-s mean |Fz| of either finger above FZ_GLITCH_N -> abort, hand
  opened) and FZ_GLITCH_N 8 -> 5 N (grip targets are 1-2.5 N). Trial 200454: thumb 7.99 N at grip_set (just under 8),
  then 8-19 N through the hold while the index read 0.5-1 N - a two-finger pinch cannot be that unbalanced.
v6_23 (29 Oct): two sensor-glitch guards. (1) Pre-flight also needs >= MIN_CHANGE_HZ changing samples per second on
  each sensor (open hand and pinch pose; --allow-slow overrides): the thumb read 9.5 Hz at open hand in trial 200120
  and then reported 20-39 N on a blueberry. (2) After grip_set, if either finger reads more than FZ_GLITCH_N, the
  trial is aborted and the hand opened (a blueberry cannot carry that; it is a sensor fault, replug that sensor).
v6_22 (29 Oct): low-force / released rules use the force ABOVE each sensor's unloaded reading. In 20 mm trial 194839
  the berry left the fingers ~25 s into loosening, but the thumb then read 0.1-0.25 N above its unloaded value and the
  index sat at its own 0.36 N offset, so the raw sum never fell below 25 % of its start value: no low-force event, no
  "released" stop, full-rate saving for 75 s more and two 0.9 s computer stalls. Now: net = mean(sum Fz) - sum(unloaded),
  low when net < LOW_FORCE_FRAC x net at loosening start. RELEASED_LOOSEN_S 10 -> 15 s so loosening goes on well past the
  point where the force first disappears (trial 193746 stopped while the berry was still wedged).
v6_21 (29 Oct): --loosen-weights "j:w,j:w,..." overrides LOOSEN_WEIGHTS for this trial (logged in trial_info as
  loosen_weights). Default unchanged ({2:0.5, 3:1.0, 14:0.5, 15:1.0}, used for all 16 mm trials). Why: with 20 mm
  berries the pinch pose leaves the index DIP almost straight (act 0.14 rad at hold, 0.02 rad after 0.28 rad of
  loosening, lower limit about -0.37), so DIP/PIP loosening runs out of range, the pads only tilt into a V and the
  berry stays wedged at the tips (trials 192815, 193746). Adding the index MCP flexion joint (j1, 1.8 rad at the pinch)
  moves the index tip away from the thumb. Recommended for 20 mm: --loosen-weights 1:0.5,2:0.5,3:1.0,14:0.5,15:1.0

v6_20 (29 Oct): CLOSED-LOOP REGRASP (release mode only). --regrasp edgelag | force runs the pre-registered offline
  detector live (OnlineSlipDetector): per finger, every new Marker2D sample gives the edge lag (contact centre vs edge
  displacement, masks from the first REF_WIN s of the hold, 5-sample smoothing - the same definition as
  analyze_slip.py) and the summed normal force; both are z-scored against the hold. When the chosen signal (edgelag:
  edge lag; force: the drop in normal force, a baseline trigger) has z > 5 for 0.25 s on either finger, loosening
  stops and ONE command closes the loosened joints to REGRASP_EXTRA rad beyond the pre-loosening pose. The berry
  must then stay held for REGRASP_HOLD_S (result "caught"; "dropped" otherwise; "missed" if it fell before the
  detector fired). Both signals are written to regrasp_detector.csv in every regrasp trial, so the two triggers can
  be compared offline on the same data. The grape controller (slip_regrasp_10cycles.py) is NOT used: it streamed
  commands at 30-60 Hz into the node's backlog, regrasped on a 1-s timer even without a detection, used a
  grape-specific fixed pose and logged nothing. --regrasp none (default) = v6_19.

v6_19 (28 Oct): the low-force checks used the instantaneous sum of the two forces, whose noise spikes (0.3-0.7 N on
  the index) kept resetting the timer. In S16c release 2 the grip was ~0 for 75 s, but neither the low-force camera
  rule nor any stop fired, so loosening ran to 1.27 rad (130 s trial, 8800 camera-2 frames). Now the checks use the
  1-s moving average of the summed force, and loosening also stops once that average has stayed below
  LOW_FORCE_FRAC of its loosening-start value for RELEASED_LOOSEN_S (event "released").

v6_18 (28 Oct): with slow loosening the berry stays held at low force for many seconds and its gross slip falls
  inside the "low force" period, where camera 2 saved only 1 frame in 5 (20 fps). LOW_FORCE_EVERY is now 2 (50 fps).

v6_17 (28 Oct): SLOWER LOOSENING. With the command lag gone (v6_16) the loosening acts at its real speed: at
  0.05-0.08 rad/s the grip halved within ~2 s of loosening start and the berry moved 1 mm against the upper
  fingertip 1.7-2.2 s after it (S16a, 4 trials), before the edge-lag detector fired in 3 of 4. In the v6_14 data the
  backlog replayed the loosening late and at roughly half speed. RATE_CHOICES is now 0.01 / 0.015 / 0.02 rad/s, which
  stretches the same transition over roughly 5-10 s. Everything else is unchanged from v6_16.

v6_16 (28 Oct): HAND COMMAND LAG FIXED. Up to v6_15 every loop published the pose at CMD_RATE = 30 Hz,
  but the LEAP node (leaphand_node.py) applies each /leaphand_node/cmd_leap message under the same lock as its
  60 Hz read loop and sleeps 1/60 s inside it, so it only gets through roughly 15-20 messages per second. Its
  subscriber queue is unbounded, so the backlog - and with it the delay between a command and the finger
  moving - grew by ~0.4-0.5 s per second of running: ~3 s at the squeeze, ~25-35 s by loosening
  (28 Oct test trials: grip_set commands applied 26 s late; loosening acted 34 s late). This is why the
  fingers "broke free" 14-35 s after loosening began whatever the speed, and later after longer holds.
  Now every publish goes through ThrottledPublisher: a new message only when the pose has changed, at most
  CMD_MAX_HZ (10 Hz), plus a CMD_KEEPALIVE_S re-send of an unchanged pose. Before recording, the script
  checks that the free joints reach the pinch pose within LATENCY_MAX_S (aborts if the hand lags).
  No change to the LEAP node or its gains (kP 800, kI 0, kD 200, current limit 550 mA).

v6_15 (28 Oct, professor's review): GRIP FORCE IS NOW RANDOMISED AND ACTUALLY SET.
  Up to v6_14 --target-force / FORCE_CHOICES were printed but had no effect: the squeeze stopped on the target
  only while the operator was still holding the berry (forces ~0.1-0.4 N), so it always ran the full +0.5 rad,
  and after the hand-off the grip settled at whatever the berry and the motor current limit gave (1-3.4 N).
  Now:
    - the squeeze always closes the full +0.5 rad (identical for every trial; no early target stop);
    - after the hand-off and settling, a new phase "grip_set" opens/closes the two squeezed DIP joints (index 3,
      thumb 15) together in small steps until the 0.5-s mean |Fz| of the live sensors is within GRIP_TOL of
      the target, held for GRIP_IN_BAND_S; then the settle gate runs again before the hold;
    - target drawn from FORCE_CHOICES = 1.0 / 1.5 / 2.0 / 2.5 N (or --target-force); achieved force, steps and
      result are logged ("grip_set" in trial_info.json). A missed target is a QC note, not a failure;
      the analysis uses the achieved force;
    - --max-force (safety) default 4.0 N (grips of 2-3.4 N were normal and only produced warnings);
    - --no-grip-control gives the v6_14 behaviour.
  Also: --berry-id and --diameter-mm are REQUIRED (for leave-one-berry-out / leave-one-size-out), and the
  placement prompt reminds the operator that the mm ruler must be in camera 2's view (pixel scale is
  measured from the ruler with ruler_calibrate.py, no longer from the berry's own width).

v6_14 (25 Oct evening): if loosening reaches LOOSEN_MAX with the berry still held, the script keeps the
pose and records for up to WAIT_AFTER_LOOSEN (30 s, was 5 s) until the berry drops or the grip force has
stayed below LOW_FORCE_FRAC of its loosening-start value for RELEASED_S. Release 6 (25 Oct 21:08, 0.08
rad/s) stopped 5 s after 1.5 rad with ~1 N still on each finger, so its slip was not recorded.

v6_13 (25 Oct evening): the finger-position reader keeps retrying /leap_position (every 1 s) instead of
giving up after one failed call at start-up; the ROS master was briefly unreachable on 25 Oct 20:45 and
two trials lost their joint_log.csv that way. Before placement the script waits up to 10 s for it.

v6_12 (25 Oct evening): fixed end-of-trial order. The v6_11 test crashed (segmentation fault) the moment
the PNG conversion started, while the Xense sensors were still shutting down; the v6_10 trial before it
lost a Xense camera ("Camera 2 disconnected") during the conversion. Now: stop the sensor readers ->
release the Xense sensors -> save events.csv / taxels.npz / trial_info.json -> only then do the cameras
convert to PNG (they wait for CONVERT_GO).

v6_11 (25 Oct evening): smaller trials. A v6_10 trial with a long hold and slow loosening was 21 GB (15,500 PNGs,
8 min of conversion). Now camera 1 (not used by the analysis) saves 1 frame in CAM1_EVERY, and once the
grip force has fallen below LOW_FORCE_FRAC of its value at loosening start for LOW_FORCE_S, camera 2 also
saves 1 in LOW_FORCE_EVERY (event "low_force"); the slip onset and early slip are long over by then.
Every frame is still time-stamped. Hand motion is unchanged.

v6_10 (25 Oct evening): events.csv, taxels.npz and trial_info.json are saved BEFORE the camera frames are
finished and converted to PNG, so closing the terminal during the conversion no longer loses the tactile
data (test 1 of v6_9 was lost that way). trial_info.json has "status": "complete" once everything is done.

v6_9 (25 Oct evening): the hold before loosening is randomised over 2-20 s (was 1.5-5 s). In the three
pilot trials the fingers broke free 14.1 +/- 0.4 s after loosening started, at almost the same frame
number every time, so "time in trial" could predict slip. Camera frames in the middle of a long hold are
saved 1-in-10 (full rate for the first 1.5 s and the last 3 s). The console now reports when the fingers
actually start to move during loosening. Hand motion is unchanged from v6_8 / v2.

v6_8: colour PNGs made with the camera SDK's own conversion (MV_CC_ConvertPixelTypeEx -> BGR8, as in the
user's v2) right after recording, in this process with the camera still open; OpenCV demosaicing (which
gave flat, almost grey images) is only the fallback.

v6_7: the camera grab loop never waits for the disk; if the write queue backs up, only every 2nd/4th
frame is saved (all frames stay time-stamped, saved=0/1). 25 Oct 10:45: a 37 s full-rate loosening filled
the queue, the grab loop blocked, frames were missed and the cameras desynchronised.

v6_6: records until the berry has actually fallen (loosening up to 1.5 rad, then up to 5 s more; 2 s after
the drop). The 25 Oct 10:33 trial ended 1 s after loosening with the berry still held.

v6_5: hand behaviour back to the user's v2 (the one where the berry always drops): squeeze 0.05 rad every
0.15 s up to +0.5 rad, short settle (max 8 s, never blocks), no trims / wind-up removal / stall stop
(those are now opt-in with --hand-control), loosening speeds 0.03/0.05/0.08 rad/s. Finger positions are
still logged to joint_log.csv, and loosening continues past 0.48 rad only if the berry has not dropped.

v6_4: reads the ACTUAL finger positions (/leap_position) into joint_log.csv; the squeeze stops when the
fingers stall against the object; after the hand-off the command wind-up is removed (commands set to
actual + KEEP_GAP) so trims and loosening act immediately; loosening continues until the object drops
(max LOOSEN_MAX). Also aborts before placement if the LEAP hand node is not listening on /leaphand_node/cmd_leap (25 Oct
trial: the node was down, so squeeze/trim/loosen commands were silently lost); camera frame cap raised to 60000.

v6_3: the PNG conversion at the end of each trial runs in a separate fresh Python process (the in-process
conversion could hang after --gpu trials). Conversion stays automatic; --no-convert still skips it.

v6_2 (after the 25 Oct pilot): stick_frac uses the median translation of the contact and thresholds
scaled to the field noise measured in the reference window (NOISE_K); settling is judged on the drift of
2-s mean forces (the per-sample noise is 0.15-0.25 N); a settled grip above --max-force is trimmed by
opening that finger's DIP in TRIM_STEP steps. Recommended field: --gpu with the default marker field
(Mesh3DFlow rounds small in-plane movements to exactly zero).

What v6 changes on top of v5 (speed: v5 ran at only 6-7 Hz because the Xense network runs on the CPU)

  A. FAST MARKER FIELD (default, --field marker). Each sensor thread reads the Xense Marker2D grid
     (26 x 14 marker positions, ~30 Hz on the CPU, no neural network) on EVERY loop, and the
     network outputs (Force + ForceResultant, ~65 ms on the CPU) only every --force-every loops
     (default 6) in the same call. The incipient-slip features (stick_frac, flow_edge_centre)
     come from marker displacement since the hold reference, at the full marker rate; the force
     field is used for the contact mask, normal force, drop detection and dT features at a lower
     rate. --field mesh gives the v5 behaviour (Mesh3DFlow; slow without a GPU).
     The no-contact marker positions (Marker2DInit) are saved once per sensor for offline use.

  B. --gpu. Loads the CUDA 11 libraries from ~/cuda11_libs (pip --target install; override with
     the XENSE_CUDA_LIBS environment variable) inside this process only, and asks onnxruntime for
     the CUDA provider. Nothing on the system is changed. The providers actually used are printed
     and saved in trial_info.json; if CUDA cannot load, the run continues on the CPU. With a working
     GPU use --force-every 1 to get force at the full rate too.

  C. Separate freshness for the two streams: <sensor>_new (marker/field sample), <sensor>_force_new
     and <sensor>_force_age_ms in every row. Pre-flight reports both rates; --min-rate applies to
     the field stream, --min-force-rate (default 2 Hz) to the force stream.

What v5 changed (on top of v4: phases, settling gate, randomisation, controls, camera sync)

  1. CONSTANT-GRIP SLIP MODES (the main change).
       --mode pull     grip is held at a constant pose (no loosening). After a randomised quiet
                       baseline (15-25 s) the script cues the operator ("PULL NOW"), who applies a
                       slow tangential load (thread pull / added weight / tilt; say which with
                       --load-method). Slip is then caused by tangential load, not by unloading.
       --mode nopull   identical timeline and cue, but nobody touches anything. This is the
                       false-alarm control for the pull trials.
     The old modes are kept:  --mode release  (robot loosens)  and  --mode supported  (release
     control with a support under the berry).  --mode is now REQUIRED so no trial is run in the
     wrong mode by accident.

  2. SHEAR / DISPLACEMENT FIELD FEATURES. Incipient slip is "edge of the contact moves, centre
     stays stuck". Summed force and contact area cannot show that, so v5:
       - reads the Xense displacement field (default output Mesh3DFlow; see --flow-output and
         --list-outputs) together with the per-taxel force field;
       - takes a reference field during the first REF_WIN s of the quiet hold;
       - logs per finger, every row:
           stick_frac        fraction of the reference contact patch that still moves together with
                             its centre (1 = fully stuck, falls as incipient slip spreads inward)
           flow_edge_centre  mean edge displacement / mean centre displacement (1 = stuck)
           dT_edge_centre    same ratio for the change in tangential force (shear) since reference
           cop_drift         centre-of-pressure drift since reference (taxels)
           area_rel          contact area with a relative threshold (the old area counted noise)
           sn_ratio          |total shear| / normal force
         plus min_stick_frac / min_dT_edge_centre over both fingers (finger-agnostic);
       - saves the full force and displacement fields in taxels.npz so every feature can be
         recomputed offline.
     The pre-registered feature (--prereg, default min_stick_frac) is written to trial_info.json
     BEFORE the trial starts.

  3. STALE / FRESH SAMPLES. Every CSV row now says which sensor actually delivered a NEW reading
     (index_new / thumb_new) and whether it was frozen (index_stale / thumb_stale). Analysis must use
     only rows where <sensor>_new == 1 and <sensor>_stale == 0. Pre-flight measures each sensor's
     rate and aborts below --min-rate Hz (default 20) unless --allow-slow is given.

  4. CAMERA DISK BUDGET + ROI. Full-rate frames are saved from PRE_CUE_FULL_S before the cue (pull /
     nopull) or from the start of the hold (release / supported); quiet settle/baseline frames are
     saved 1-in-BASELINE_DECIMATE. Every frame is still time-stamped in camera_camX_timestamps.csv.
     --roi WxH+X+Y crops the sensor (smaller files, higher frame rate). Free disk space is checked
     before the trial.

  5. QC VERDICT. trial_info.json gets a "qc" block (sensor rates, fresh fraction in the analysis
     window, settling, cameras, frame-count match) with pass/fail reasons, so bad trials are
     rejected by rule rather than by eye.

  6. DROP DETECTION relative to each sensor's own unloaded reading (the Xense reads ~0.1-0.3 N with
     nothing touching, so a fixed 0.05 N threshold never fired).

Operator instructions for pull trials (also printed by the script)
  - Tie the thread before the trial. From the hand-off on, hold the thread end OUTSIDE the camera
    view, slack, and keep your hand still (no hand entering the frame = no camera-side confound).
  - At "PULL NOW": one slow, smooth pull in a fixed direction. Do not jerk or pull repeatedly.
  - Press ENTER when the pull is finished (or the berry has dropped), then keep still.

Usage (v6_15)
  python3 unified_grasp_capture_v6_19.py blueberry --mode release   --gpu --force-every 1 --berry-id B17a --diameter-mm 16.9
  python3 unified_grasp_capture_v6_19.py blueberry --mode supported --gpu --force-every 1 --berry-id B17a --diameter-mm 16.9
  python3 unified_grasp_capture_v6_19.py blueberry --mode nopull    --gpu --force-every 1 --berry-id B17a --diameter-mm 16.9
Usage (v6_20, closed-loop regrasp)
  python3 unified_grasp_capture_v6_20.py blueberry --mode release --regrasp edgelag --gpu --force-every 1 --berry-id B16f --diameter-mm 16.1
  python3 unified_grasp_capture_v6_20.py blueberry --mode release --regrasp force   --gpu --force-every 1 --berry-id B16f --diameter-mm 16.1
Older usage
  python3 unified_grasp_capture_v6.py --list-outputs
  python3 unified_grasp_capture_v6.py blueberry --mode pull   --load-method thread --note "berry 07, 17.7 mm"
  python3 unified_grasp_capture_v6.py blueberry --mode nopull --note "berry 07 control"
  python3 unified_grasp_capture_v6.py blueberry --mode pull --gpu --force-every 1     (after the CUDA 11 install)
  python3 unified_grasp_capture_v6.py blueberry --mode release
(run with the Xense GUI python: /usr/share/bin/xensegui/bin/python3)
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import deque
from queue import Queue, Full

import numpy as np


# ===================== Optional GPU (must run before xensesdk is imported) =====================
GPU_STATUS = {"requested": "--gpu" in sys.argv, "libs_loaded": 0, "sessions": [], "error": None}


def _enable_gpu():
    """Load CUDA 11 libraries from a private folder into THIS process and make every onnxruntime
    session ask for CUDA first. Nothing outside this process is changed."""
    import ctypes
    import glob
    root = os.path.expanduser(os.environ.get("XENSE_CUDA_LIBS", "~/cuda11_libs"))
    libs = sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib", "lib*.so*")))
    if not libs:
        GPU_STATUS["error"] = f"no CUDA libraries found under {root}"
        print(f"[gpu] {GPU_STATUS['error']} - running on CPU")
        return
    pending, loaded = list(libs), 0
    for _ in range(6):                       # retry until dependencies are satisfied
        still = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL); loaded += 1
            except OSError:
                still.append(path)
        if len(still) == len(pending):
            break
        pending = still
    GPU_STATUS["libs_loaded"] = loaded
    print(f"[gpu] loaded {loaded}/{len(libs)} CUDA libraries from {root}")
    try:
        import onnxruntime as ort
    except ImportError as e:
        GPU_STATUS["error"] = f"onnxruntime import failed: {e}"
        return
    orig = ort.InferenceSession

    class CudaSession(orig):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
            super().__init__(path_or_bytes, sess_options,
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"], **kw)
            used = self.get_providers()
            GPU_STATUS["sessions"].append(used)
            print(f"[gpu] onnxruntime session providers: {used}")

    ort.InferenceSession = CudaSession


if GPU_STATUS["requested"]:
    _enable_gpu()

import rospy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from xensesdk import Sensor  # noqa: E402
from ctypes import memmove, byref, c_ubyte  # noqa: E402

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *  # noqa: E402,F401,F403

SCRIPT_NAME = "unified_grasp_capture_v6_28.py"

# ===================== Hardware IDs =====================
INDEX_SENSOR_ID = "OG000961"
THUMB_SENSOR_ID = "OG000324"
CAMERA_SERIALS = {"cam1": "DB1216927", "cam2": "DB1216964"}
OUT_ROOT = "/media/dataset_storage/toy"

# ===================== Camera settings =====================
CAM_WIDTH, CAM_HEIGHT = 1280, 1024
CAM_EXPOSURE_TIME = 1000.0     # us
MAX_CAMERA_FRAMES = 60000      # safety cap only (10 min at 100 fps; quiet phases are saved 1-in-10)
CAM_QUEUE_SIZE = 1500          # raw frames buffered per camera before the writer catches up
TRIGGER_FPS = 100.0            # software-trigger rate during recording
WARMUP_FPS = 20.0              # trigger rate before recording (lets auto white balance settle)
BASELINE_DECIMATE = 10         # quiet settle/hold frames: save 1 in N (all are time-stamped)
PRE_CUE_FULL_S = 3.0           # pull/nopull: save every frame from this long before the cue
FULL_SAVE_PHASES = {"load_window", "loosen", "regrasp", "drop", "post"}
CAM1_EVERY = 5                 # camera 1 (second view, not analysed) saves 1 frame in this many
LOW_FORCE_FRAC = 0.25          # grip force below this fraction of its loosening-start value ...
LOW_FORCE_S = 3.0              # ... for this long counts as "released"
LOW_FORCE_EVERY = 2            # after that, camera 2 saves 1 frame in this many
MIN_FREE_GB = 20.0             # abort if the output disk has less free space than this

# ===================== Tactile settings =====================
STALE_WARN_S = 1.5             # s with an identical, non-zero output -> warn: sensor probably frozen
STALE_DEAD_S = 3.0             # s with an identical, non-zero output -> treat sensor as dead
FRESH_AGE = 1.0                # s; a sensor with no new sample for this long is treated as dead
MIN_RATE_HZ = 20.0             # pre-flight: per-sensor rate must reach this (see --min-rate)
AREA_EPS = 1e-4                # old absolute contact threshold (kept for the old *_contact_area column)
CONTACT_FRAC = 0.10            # new contact mask: taxel Fz > CONTACT_FRAC * max Fz in that frame ...
FZ_MIN_TAXEL = 1e-3            # ... and above this absolute floor
MIN_CONTACT_TAXELS = 8         # reference patch smaller than this -> field features are NaN
CENTRE_FRAC = 0.5              # centre region: distance from patch centre < CENTRE_FRAC * r_eq
EDGE_FRAC = 0.7                # edge region:   distance from patch centre >= EDGE_FRAC * r_eq
STICK_TOL = 0.3                # taxel is "stuck" if |d_i - d_centre| < STICK_TOL * |d_centre|
DISP_MIN = {"marker": 0.5, "mesh": 0.02}   # |centre displacement| below this = no load -> stick_frac 1
                                           # (marker: pixels of Marker2D; mesh: Mesh3DFlow units, mm)
REF_WIN = 1.0                  # s of fresh samples at the start of the hold averaged into the reference
NOISE_K = 5.0                  # adaptive thresholds = NOISE_K x the field noise measured in the reference window
FIELD_OUTPUTS = {"marker": "Marker2D", "mesh": "Mesh3DFlow"}
FORCE_EVERY = 6                # marker mode: read Force + ForceResultant every N sensor loops (1 = every loop)
MIN_FORCE_RATE_HZ = 2.0        # pre-flight: force stream must reach this
DEFAULT_PREREG = "min_stick_frac"
FLOW_FEATURES = {"stick_frac", "flow_centre", "flow_edge_centre"}
QC_MIN_FRESH = 0.8             # QC: fraction of rows in the analysis window with a fresh sample, per sensor

# ===================== Hand settings =====================
CMD_RATE = 30                  # control-loop rate of this script (commands are throttled, see below)
CMD_MAX_HZ = 10.0              # v6_16: at most this many /cmd_leap messages per second (the node manages ~15-20)
CMD_KEEPALIVE_S = 1.0          # re-send an unchanged pose this often
LATENCY_JOINTS = (1, 12)       # free joints (index MCP, thumb base) used for the lag check at the pinch pose
LATENCY_TOL = 0.05             # rad
LATENCY_MAX_S = 5.0            # s allowed for the free joints to reach the pinch pose
JOINT_NAMES = [f"joint_{i}" for i in range(16)]
DIP_SQUEEZE_STEP = 0.05        # rad per squeeze step (index DIP 3, thumb DIP 15) - same as v2
SQUEEZE_MIN_WAIT = 0.15        # s per step (v2: 0.15 s), plus one fresh force sample when available
MAX_DIP_EXTRA = 0.5
LOOSEN_WEIGHTS = {2: 0.5, 3: 1.0, 14: 0.5, 15: 1.0}   # release mode: subtract from these joints
LOOSEN_CAP = 0.48              # rad of total loosening on the main joints (v4 default; logged)
LOOSEN_MAX = 1.5               # keep loosening past LOOSEN_CAP until the object drops, up to this
SQUEEZE_STALL_GAP = 0.08       # rad: stop the squeeze when a squeezed joint lags its command by more than this
KEEP_GAP = 0.03                # rad: after the hand-off, commands are pulled back to (actual + KEEP_GAP)
UNWIND_JOINTS = (2, 3, 14, 15) # joints whose command-vs-actual wind-up is removed (same as loosening)
HAND_POLL_HZ = 10.0            # /leap_position read rate (joint_log.csv)
DROP_CONFIRM_S = 5.0          # v6_27: keep loosening this long after a force-based drop detection
LOOSEN_TICK = 0.1              # s between small loosening increments
DROP_MARGIN = 0.10             # N above each sensor's own unloaded |Fz| ...
DROP_TIME = 0.5                # ... for this long on both live sensors = object dropped

# Randomisation (used when the matching option is not given)
HOLD_RANGE = (2.0, 20.0)                # release/supported: s before loosening starts (v6_9: wide)
HOLD_FULL_HEAD = 1.5                    # s of full-rate camera saving at the start of the hold (reference)
HAND_MOVE_RAD = 0.01                    # total movement of joints 2/3/14/15 that counts as "fingers moved"
BASELINE_RANGE = (15.0, 25.0)           # pull/nopull: quiet baseline before the cue
RATE_CHOICES = [0.01, 0.015, 0.02]      # release/supported: loosening speed, rad/s (v6_17; was 0.03/0.05/0.08)
FORCE_CHOICES = [1.0, 1.5, 2.0, 2.5]    # grip target, N (v6_15: actually applied in the grip_set phase)

# Grip setting after the hand-off (v6_15)
GRIP_JOINTS = (3, 15)          # the squeezed DIP joints, moved together (+ closes, - opens)
GRIP_WIN = 0.5                 # s, mean |Fz| window
GRIP_TOL = 0.10                # N, |mean - target| counted as on target ...
GRIP_IN_BAND_S = 1.0           # ... continuously for this long
GRIP_STEP_BIG = 0.02           # rad per step while far from the target (|err| > GRIP_BIG_ERR) ...
GRIP_WAIT_BIG = 0.3            # ... every this many s (opens the squeeze wind-up quickly)
GRIP_STEP_SMALL = 0.005        # rad per step near the target ...
GRIP_WAIT_SMALL = 0.6          # ... every this many s (lets the 0.5-s mean catch up)
GRIP_BIG_ERR = 0.5             # N
GRIP_DEAD_N = 0.05             # N: opening and the mean force changed less than this since the last step
                               # = still in the squeeze wind-up (motor at its current limit) -> big steps
GRIP_MAX_OPEN = 0.9            # rad, never open the DIPs more than this in grip_set
GRIP_MAX_CLOSE = 0.3           # rad, never close the DIPs more than this in grip_set
GRIP_SET_MAX = 30.0            # s, then continue with whatever force was reached (logged)
LOAD_WINDOW_MAX = 20.0                  # pull: s to wait for ENTER after the cue
CONTROL_WINDOW = 10.0                   # nopull (and pull with --no-confirm): fixed window after the cue
POST_PULL = 10.0                        # pull/nopull: untouched recording after the window
POST_RELEASE = 2.0                      # release/supported: recording after the drop / loosening end
WAIT_AFTER_LOOSEN = 30.0                # if not dropped at LOOSEN_MAX: keep recording up to this long (v6_14: was 5)
MIN_CHANGE_HZ = 15.0                    # v6_23: pre-flight minimum changing samples per second, per sensor
FZ_GLITCH_N = 5.0                       # v6_23: |Fz| above this after grip_set = sensor fault -> abort
RELEASED_CFG = {}                       # v6_28: {"s": value of --released-s}
RELEASED_LOOSEN_S = 15.0                # v6_19: during loosening, stop once the 1-s mean force has been low this long

# ---- v6_20: closed-loop regrasp (--regrasp edgelag | force; release mode) ----
REGRASP_Z = 5.0                         # pre-registered onset rule (as analyze_slip.py): z > REGRASP_Z ...
REGRASP_DUR = 0.25                      # ... for this many seconds, on either finger
REGRASP_SMOOTH_N = 5                    # samples, moving average of the marker displacement (as analyze_slip.py)
REGRASP_MIN_BASE = 15                   # fresh samples per finger needed in the hold baseline to arm that finger
REGRASP_MIN_HOLD_S = 5.0                # regrasp trials hold at least this long before loosening (z baseline)
REGRASP_EXTRA = 0.05                    # rad closed beyond the pre-loosening pose on the loosened joints
REGRASP_HOLD_S = 5.0                    # berry must stay held this long after the regrasp command = "caught"
REGRASP_RECOVER_FRAC = 0.8              # grip counts as restored when the summed |Fz| is back to this x loosen-start
REGRASP_BACKOFF = 0.01                  # rad opened per step if a finger stays above --max-force for 0.3 s
RELEASED_S = 5.0                        # ... but stop once the grip force has stayed below LOW_FORCE_FRAC this long

# Hand-off / settling gate
# (pilot 25 Oct: after the hand-off the grip force kept rising for ~12 s, and the reading noise is
#  0.15-0.25 N per sample, so a standard-deviation test can never pass; the test now uses drift of 2-s means)
SETTLE_MIN = 2.0               # s, minimum wait after the operator lets go
SETTLE_MAX = 8.0               # s, then continue anyway (settle_stable is logged, the trial is not blocked)
SETTLE_WIN = 2.0               # s, length of each of the two averaging windows compared
SETTLE_DRIFT = 0.08            # N, |mean(last 2 s) - mean(previous 2 s)| must stay below this ...
SETTLE_HOLD = 1.0              # s, ... continuously for this long, on every live sensor
TRIM_STEP = 0.01               # rad, open a finger's DIP by this when its settled force exceeds --max-force
TRIM_MAX = 15                  # at most this many trim steps
TRIM_JOINT = (3, 15)           # index DIP, thumb DIP (the joints the squeeze closes)

# ===================== Shared state =====================
stop_flag = threading.Event()
reset_pause = threading.Event()   # v6_26: readers idle while the reference images are reset
SENSOR_OBJS = []                  # v6_26: [index, thumb] Sensor objects
t0_global = None
phase = "init"                 # current protocol phase, written into every tactile row / camera frame
phase_start = {}               # phase name -> wall time it started
events = []                    # (name, wall_time, elapsed_s)
cam_handles = {}               # name -> MvCamera, for the software-trigger thread
full_save_from = None          # elapsed s from which every camera frame is saved (set by the protocol)
low_force_from = None          # elapsed s from which the grip counts as released (camera thinning)
full_save_pause = None         # (start, end) elapsed s inside which frames go back to 1-in-BASELINE_DECIMATE
unloaded_fz = [0.0, 0.0]       # per-sensor |Fz| with nothing touching
live_feat = {}                 # latest computed features, for the console status line
SENSOR_NAMES = ("Index", "Thumb")
SENSOR_KEYS = ("index", "thumb")

# n / t / stale / ms: field (marker or mesh) stream;  nf / tf / fstale: force stream (Force + ForceResultant)
latest = [{"f": None, "rf": None, "fl": None, "t": 0.0, "n": 0, "stale": 0, "ms": 0.0,
           "tf": 0.0, "nf": 0, "fstale": 0, "fms": 0.0,
           "chg": 0.0, "fchg": 0.0, "zero": False, "fzero": False, "nchg": 0} for _ in range(2)]
latest_lock = threading.Lock()
sensor_lock = threading.Lock()   # used only with --serial-sensors


def elapsed():
    return None if t0_global is None else time.time() - t0_global


def mark(name):
    """Start a new protocol phase and log the event time."""
    global phase
    now = time.time()
    phase = name
    phase_start[name] = now
    el = None if t0_global is None else round(now - t0_global, 6)
    events.append((name, round(now, 6), el))
    rospy.loginfo(f"[phase] {name}" + ("" if el is None else f"  (t = {el:.3f} s)"))


def mark_event(name):
    """Log a point event without changing the phase."""
    now = time.time()
    events.append((name, round(now, 6), None if t0_global is None else round(now - t0_global, 6)))


# ===================== Hand helpers =====================
def open_pose():
    return [0.0] * 16


def pinch_pose():
    return [
        0.057, 1.800, 0.059, -0.268,    # index
        -0.028, -0.005, -0.004, -0.000,  # middle
        -0.028, 0.047, -0.086, 0.041,   # ring
        1.533, 0.041, 0.434, -0.426,    # thumb
    ]


def interpolate(a, b, steps=80):
    return [[x + (i / steps) * (y - x) for x, y in zip(a, b)] for i in range(steps + 1)]


last_cmd = None                # last commanded pose (for joint_log.csv)


def make_msg(pos):
    global last_cmd
    last_cmd = list(pos)
    msg = JointState()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.name = JOINT_NAMES
    msg.position = list(pos)
    return msg


def current_fz():
    """|Fz| of index and thumb in newtons from the reader threads."""
    with latest_lock:
        rfs = [s["rf"] for s in latest]
    return tuple(0.0 if rf is None else abs(float(rf[2])) for rf in rfs)


def sensor_alive():
    """[index_ok, thumb_ok]: both streams recently updated and not returning frozen arrays."""
    now = time.time()
    with latest_lock:
        return [s["rf"] is not None and now - s["t"] < FRESH_AGE and now - s["tf"] < FRESH_AGE + 1.0
                and (now - s["chg"] < STALE_DEAD_S or s["zero"])
                and (now - s["fchg"] < STALE_DEAD_S + 1.0 or s["fzero"]) for s in latest]


def sample_counts():
    """Field-stream sample counters."""
    with latest_lock:
        return [s["n"] for s in latest]


def change_counts():
    """Number of field samples that differed from the previous one (= real new sensor images)."""
    with latest_lock:
        return [s["nchg"] for s in latest]


def force_counts():
    """Force-stream sample counters."""
    with latest_lock:
        return [s["nf"] for s in latest]


class HandMonitor:
    """Reads the ACTUAL joint positions from the LEAP node's /leap_position service (10 Hz, own thread) and
    logs them next to the commanded pose in joint_log.csv. offset = actual - command measured at the empty
    pinch pose, so gap(j) = command - actual - offset is the wind-up of joint j (>0: commanded past where
    the finger really is, e.g. pressing on the object)."""

    def __init__(self, trial_dir):
        self.proxy, self.err, self.last, self.t_last, self.offset = None, None, None, 0.0, None
        self.connect()
        self.fh = open(os.path.join(trial_dir, "joint_log.csv"), "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["elapsed_s", "phase"] + [f"cmd_{j}" for j in range(16)] + [f"act_{j}" for j in range(16)])

    def connect(self):
        try:
            import rosservice
            cls = rosservice.get_service_class_by_name("/leap_position")
            self.proxy = rospy.ServiceProxy("/leap_position", cls)
            self._call()
            self.err = None
            return True
        except Exception as e:  # noqa: BLE001
            self.err, self.proxy = f"/leap_position unavailable: {e}", None
            return False

    def wait_ok(self, timeout):
        t_end = time.time() + timeout
        while not self.ok and time.time() < t_end and not rospy.is_shutdown():
            time.sleep(0.2)
        return self.ok

    @property
    def ok(self):
        return self.proxy is not None and self.last is not None

    def _call(self):
        r = self.proxy()
        a = np.asarray(getattr(r, r.__slots__[0]), dtype=float)
        if a.size != 16:
            raise ValueError(f"expected 16 joint values, got {a.size}")
        self.last, self.t_last = a, time.time()
        return a

    def run(self):
        while not stop_flag.is_set() and not rospy.is_shutdown():
            t_a = time.time()
            if self.proxy is None:
                self.connect()
                if self.proxy is None:
                    time.sleep(1.0); continue
            if self.proxy is not None:
                try:
                    a = self._call()
                    if t0_global is not None and last_cmd is not None:
                        self.w.writerow([round(time.time() - t0_global, 4), phase]
                                        + [round(v, 5) for v in last_cmd] + [round(float(v), 5) for v in a])
                except Exception as e:  # noqa: BLE001
                    self.err = str(e)
            time.sleep(max(0.0, 1.0 / HAND_POLL_HZ - (time.time() - t_a)))
        self.fh.close()

    def fresh(self, max_age=0.5):
        return self.ok and time.time() - self.t_last < max_age

    def calibrate(self, cmd):
        if self.fresh():
            self.offset = self.last - np.asarray(cmd, dtype=float)
        return self.offset

    def gap(self, cmd, j):
        if self.offset is None or not self.fresh():
            return None
        return float(cmd[j] - self.last[j] + self.offset[j])


hand = None


class DropDetector:
    """Object counts as dropped when every live sensor reads within DROP_MARGIN of its unloaded |Fz|
    for DROP_TIME s."""

    def __init__(self):
        self.low_since = None

    def update(self):
        fz = current_fz()
        alive = sensor_alive()
        live = [k for k in range(2) if alive[k]] or [0, 1]
        low = all(fz[k] < unloaded_fz[k] + DROP_MARGIN for k in live)
        if not low:
            self.low_since = None
            return False
        self.low_since = self.low_since or time.time()
        return time.time() - self.low_since >= DROP_TIME


# ===================== Tactile: per-sensor reader threads =====================
def centre_of_pressure(fz):
    """Weighted centre of a |Fz| map in taxel units (row, col); NaN if no contact."""
    w = fz.sum()
    if w <= 1e-6:
        return float("nan"), float("nan")
    r, c = np.mgrid[0:fz.shape[0], 0:fz.shape[1]]
    return float((r * fz).sum() / w), float((c * fz).sum() / w)


def sensor_reader(k, sensor, serial_mode, field_type, force_every):
    """Reads ONE Xense sensor in a loop. Every loop returns the displacement field (fast stream);
    every `force_every` loops the same call also returns Force + ForceResultant (slow stream)."""
    O = Sensor.OutputType
    prev_fl, prev_f = None, None
    stale, fstale, warned = 0, 0, False
    last_chg = last_fchg = time.time()
    fzero = False
    field_is_displacement = "Flow" in str(field_type)   # Mesh3DFlow is ~0 at rest; Marker2D positions never are
    i = 0
    while not stop_flag.is_set() and not rospy.is_shutdown():
        if reset_pause.is_set():
            time.sleep(0.01); last_chg = last_fchg = time.time(); continue
        with latest_lock:
            have_force = latest[k]["rf"] is not None
        do_force = force_every <= 1 or i % force_every == 0 or not have_force
        outs = [O.Force, O.ForceResultant, field_type] if do_force else [field_type]
        i += 1
        t_a = time.time()
        try:
            if serial_mode:
                with sensor_lock:
                    res = sensor.selectSensorInfo(*outs)
            else:
                res = sensor.selectSensorInfo(*outs)
            if do_force:
                f = np.asarray(res[0], dtype=np.float32)
                rf = np.asarray(res[1], dtype=np.float64)
                fl = np.asarray(res[2], dtype=np.float32)
            else:   # one output: the SDK returns the array itself
                fl = np.asarray(res[0] if isinstance(res, (tuple, list)) else res, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            rospy.logwarn_throttle(2.0, f"[Xense] {SENSOR_NAMES[k]} read error: {e}")
            time.sleep(0.02)
            continue
        if prev_fl is not None and fl.shape != prev_fl.shape:
            rospy.logwarn_throttle(2.0, f"[Xense] {SENSOR_NAMES[k]} field shape changed {prev_fl.shape} -> {fl.shape}; "
                                        "sample ignored")
            continue
        now = time.time()
        # stale = number of identical readings in a row (a repeat is normal when we poll faster than the
        # sensor camera); frozen = unchanged for STALE_DEAD_S seconds while the output is not all zeros
        # (an idle sensor with nothing touching it may legitimately return exact zeros).
        same = prev_fl is not None and np.array_equal(fl, prev_fl)
        stale = stale + 1 if same else 0
        if not same:
            last_chg, warned = now, False
        prev_fl = fl
        zero = not np.any(fl) if field_is_displacement else False
        if do_force:
            fsame = prev_f is not None and np.array_equal(f, prev_f)
            fstale = fstale + 1 if fsame else 0
            if not fsame:
                last_fchg = now
            prev_f = f
            fzero = not np.any(f)
        unchanged_s = now - last_chg
        if unchanged_s >= STALE_WARN_S and not zero and not warned:
            rospy.logwarn(f"[Xense] {SENSOR_NAMES[k]} sensor unchanged for {unchanged_s:.1f} s - frozen/disconnected?")
            warned = True
        ms = (now - t_a) * 1000.0
        with latest_lock:
            s = latest[k]
            s.update(fl=fl, t=now, n=s["n"] + 1, stale=stale, ms=ms, chg=last_chg, zero=zero,
                     nchg=s["nchg"] + (0 if same else 1))
            if do_force:
                s.update(f=f, rf=rf, tf=now, nf=s["nf"] + 1, fstale=fstale, fms=ms, fchg=last_fchg, fzero=fzero)


# ===================== Field features (incipient-slip signatures) =====================
def contact_mask(fz):
    m = fz.max() if fz.size else 0.0
    if m <= FZ_MIN_TAXEL:
        return np.zeros_like(fz, dtype=bool)
    return (fz > CONTACT_FRAC * m) & (fz > FZ_MIN_TAXEL)


def _vec(field):
    """First two components (tangential) of an (H, W, >=2) field, else None."""
    if field is None or field.ndim != 3 or field.shape[2] < 2:
        return None
    return field[:, :, :2].astype(np.float64)


class FieldReference:
    """Reference patch for one sensor, averaged over the first REF_WIN s of fresh samples in the hold."""

    def __init__(self, name):
        self.name = name
        self.acc_f, self.acc_fl = [], []
        self.t_first = None
        self.ready = False
        self.failed = None
        self.info = {}

    def add(self, f, fl, now):
        """f: a NEW force field or None; fl: a NEW displacement-field sample or None."""
        if self.ready or self.failed:
            return
        self.t_first = self.t_first or now
        if f is not None:
            self.acc_f.append(f)
        if fl is not None:
            self.acc_fl.append(fl)
        if now - self.t_first >= REF_WIN and len(self.acc_f) >= 1 and len(self.acc_fl) >= 3:
            self._freeze()

    def _freeze(self):
        f = np.mean(np.stack(self.acc_f), axis=0)
        fz = np.abs(f[:, :, 2])
        mask = contact_mask(fz)
        n = int(mask.sum())
        if n < MIN_CONTACT_TAXELS:
            self.failed = f"reference patch only {n} taxels"
            rospy.logwarn(f"[ref] {self.name}: {self.failed} - field features will be NaN")
            return
        r0, c0 = centre_of_pressure(np.where(mask, fz, 0.0))
        rr, cc = np.mgrid[0:fz.shape[0], 0:fz.shape[1]]
        d = np.hypot(rr - r0, cc - c0)
        r_eq = np.sqrt(n / np.pi)
        self.mask = mask
        self.centre = mask & (d < CENTRE_FRAC * r_eq)
        self.edge = mask & (d >= EDGE_FRAC * r_eq)
        if self.centre.sum() < 2 or self.edge.sum() < 2:
            self.failed = f"patch too small to split (centre {int(self.centre.sum())}, edge {int(self.edge.sum())})"
            rospy.logwarn(f"[ref] {self.name}: {self.failed} - field features will be NaN")
            return
        self.cop = (r0, c0)
        self.T = _vec(f)
        fl = np.mean(np.stack(self.acc_fl), axis=0) if self.acc_fl else None
        self.D = _vec(fl)
        self.flow_same_grid = self.D is not None and self.D.shape[:2] == mask.shape
        self.ready = True
        self.info = {"n_force_samples": len(self.acc_f), "n_field_samples": len(self.acc_fl), "contact_taxels": n, "centre_taxels": int(self.centre.sum()),
                     "edge_taxels": int(self.edge.sum()), "cop": [round(r0, 3), round(c0, 3)],
                     "r_eq_taxels": round(float(r_eq), 3), "flow_on_force_grid": bool(self.flow_same_grid)}
        if self.D is not None and not self.flow_same_grid:
            rospy.logwarn(f"[ref] {self.name}: flow grid {self.D.shape[:2]} != force grid {mask.shape}; "
                          "flow features use a resized mask")
            self._resize_masks()
        # noise of the displacement field during the quiet reference window (sets the adaptive thresholds)
        self.sig_u, self.sig_tax = 0.0, 0.0
        if self.D is not None and len(self.acc_fl) >= 3:
            fmask = self.mask if self.flow_same_grid else self.fmask
            dDs = np.stack([_vec(x) for x in self.acc_fl]) - self.D          # (n, H, W, 2)
            us = np.median(dDs[:, fmask, :], axis=1)                          # translation per sample
            self.sig_u = float(np.sqrt((us ** 2).sum(1).mean()))
            self.sig_tax = float(np.median(np.sqrt((dDs[:, fmask, :] ** 2).sum(2).mean(0))))
        self.u_thr = max(disp_min_active, NOISE_K * self.sig_u)
        self.info.update(sig_u=round(self.sig_u, 5), sig_taxel=round(self.sig_tax, 5), u_threshold=round(self.u_thr, 5))
        self.acc_f, self.acc_fl = [], []
        rospy.loginfo(f"[ref] {self.name}: reference set ({self.info})")

    def _resize_masks(self):
        """Nearest-neighbour map of the force-grid masks onto the flow grid."""
        H, W = self.D.shape[:2]
        h, w = self.mask.shape
        ri = np.clip((np.arange(H) * h / H).astype(int), 0, h - 1)
        ci = np.clip((np.arange(W) * w / W).astype(int), 0, w - 1)
        self.fmask = self.mask[np.ix_(ri, ci)]
        self.fcentre = self.centre[np.ix_(ri, ci)]
        self.fedge = self.edge[np.ix_(ri, ci)]


NAN = float("nan")
disp_min_active = DISP_MIN["marker"]     # set in main() from --field / --disp-min
FEAT_NAMES =["area_rel", "sn_ratio", "cop_drift", "dT_edge", "dT_centre", "dT_edge_centre",
              "flow_centre", "flow_edge_centre", "stick_frac"]


def field_features(f, rf, fl, ref):
    """Per-sensor features for one sample. Field features are NaN until the reference exists."""
    fz = np.abs(f[:, :, 2])
    out = dict.fromkeys(FEAT_NAMES, NAN)
    out["area_rel"] = int(contact_mask(fz).sum())
    fzt = abs(float(rf[2]))
    shear_vec = np.hypot(float(rf[0]), float(rf[1]))
    out["sn_ratio"] = shear_vec / fzt if fzt > 0.05 else NAN
    if ref is None or not ref.ready:
        return out
    r, c = centre_of_pressure(np.where(ref.mask, fz, 0.0))
    out["cop_drift"] = float(np.hypot(r - ref.cop[0], c - ref.cop[1])) if r == r else NAN
    T = _vec(f)
    if T is not None and ref.T is not None:
        dT = np.hypot(*(T - ref.T).transpose(2, 0, 1))
        e, cen = float(dT[ref.edge].mean()), float(dT[ref.centre].mean())
        out["dT_edge"], out["dT_centre"] = e, cen
        out["dT_edge_centre"] = e / cen if cen > 1e-9 else NAN
    D = _vec(fl)
    if D is not None and ref.D is not None and D.shape == ref.D.shape:
        mask, centre, edge = ((ref.mask, ref.centre, ref.edge) if ref.flow_same_grid
                              else (ref.fmask, ref.fcentre, ref.fedge))
        dD = D - ref.D
        # rigid translation of the contact = median displacement over the reference patch (robust to the
        # outward spreading that a change in normal force produces, and to single noisy taxels)
        u = np.median(dD[mask], axis=0)
        u_mag = float(np.hypot(*u))
        out["flow_centre"] = u_mag
        mag = np.hypot(dD[:, :, 0], dD[:, :, 1])
        mc = float(mag[centre].mean())
        out["flow_edge_centre"] = float(mag[edge].mean()) / mc if mc > 1e-9 else NAN
        if u_mag < ref.u_thr:                      # no translation beyond noise: nothing is slipping
            out["stick_frac"] = 1.0
        else:                                      # stuck = moves with the contact as a whole
            dev = np.hypot(dD[:, :, 0] - u[0], dD[:, :, 1] - u[1])
            tol = max(STICK_TOL * u_mag, NOISE_K * ref.sig_tax)
            out["stick_frac"] = float((dev[mask] < tol).mean())
    return out


# ===================== Tactile logger =====================
def tactile_logger(trial_dir, store, refs):
    """Writes one CSV row (and one taxel frame) every time EITHER sensor has a new sample."""
    path = os.path.join(trial_dir, "tactile_data.csv")
    fh = open(path, "w", newline="")
    w = csv.writer(fh)
    old_cols = ["frame", "wall_time", "elapsed_s",
                "index_force_N", "index_contact_x", "index_contact_y", "index_contact_area", "index_shear_N", "index_pressure",
                "thumb_force_N", "thumb_contact_x", "thumb_contact_y", "thumb_contact_area", "thumb_shear_N", "thumb_pressure",
                "motion_magnitude", "motion_max", "motion_mean"]
    new_cols = [f"{s}_{k}" for s in SENSOR_KEYS for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "cop_row", "cop_col")]
    feat_cols = [f"{s}_{k}" for s in SENSOR_KEYS for k in FEAT_NAMES] + ["min_stick_frac", "min_dT_edge_centre"]
    extra_cols = ["index_new", "thumb_new", "index_stale", "thumb_stale", "index_seq", "thumb_seq",
                  "index_age_ms", "thumb_age_ms", "index_read_ms", "thumb_read_ms",
                  "index_force_new", "thumb_force_new", "index_force_stale", "thumb_force_stale",
                  "index_force_seq", "thumb_force_seq", "index_force_age_ms", "thumb_force_age_ms", "phase"]
    w.writerow(old_cols + new_cols + feat_cols + extra_cols)

    prev_force, motion_hist = None, []
    last_seq, last_fseq = [0, 0], [0, 0]
    frame = 0
    while not stop_flag.is_set() and not rospy.is_shutdown():
        with latest_lock:
            snap = [dict(s) for s in latest]
        seq = [s["n"] for s in snap]
        if t0_global is None or snap[0]["rf"] is None or snap[1]["rf"] is None or seq == last_seq:
            time.sleep(0.003)
            continue
        new = [int(seq[k] != last_seq[k]) for k in range(2)]
        last_seq = seq
        fseq = [s["nf"] for s in snap]
        fnew = [int(fseq[k] != last_fseq[k]) for k in range(2)]
        last_fseq = fseq
        now = time.time()
        f1, f2 = snap[0]["f"], snap[1]["f"]
        rf1, rf2 = snap[0]["rf"], snap[1]["rf"]

        # reference patch: fresh, non-frozen samples from the start of the hold
        if phase == "hold":
            for k in range(2):
                f_add = snap[k]["f"] if fnew[k] and snap[k]["fstale"] == 0 else None
                fl_add = snap[k]["fl"] if new[k] and snap[k]["stale"] == 0 else None
                if f_add is not None or fl_add is not None:
                    refs[k].add(f_add, fl_add, now)

        idx_area = int((np.abs(f1[:, :, 2]) > AREA_EPS).sum())
        thm_area = int((np.abs(f2[:, :, 2]) > AREA_EPS).sum())
        idx_shear = float(np.sum(np.hypot(f1[:, :, 0], f1[:, :, 1])))
        thm_shear = float(np.sum(np.hypot(f2[:, :, 0], f2[:, :, 1])))
        idx_fz, thm_fz = abs(rf1[2]), abs(rf2[2])
        idx_p = idx_fz / idx_area if idx_area else 0.0
        thm_p = thm_fz / thm_area if thm_area else 0.0

        cur = np.array([idx_fz, thm_fz])
        mag = 0.0 if prev_force is None else float(np.linalg.norm(cur - prev_force))
        prev_force = cur
        motion_hist.append(mag); motion_hist = motion_hist[-10:]
        mmax = float(np.max(motion_hist)) if len(motion_hist) >= 5 else 0.0
        mmean = float(np.mean(motion_hist)) if len(motion_hist) >= 5 else 0.0
        c1 = centre_of_pressure(np.abs(f1[:, :, 2]))
        c2 = centre_of_pressure(np.abs(f2[:, :, 2]))

        feats = [field_features(snap[k]["f"], snap[k]["rf"], snap[k]["fl"], refs[k]) for k in range(2)]
        sticks = [x["stick_frac"] for x in feats if x["stick_frac"] == x["stick_frac"]]
        ratios = [x["dT_edge_centre"] for x in feats if x["dT_edge_centre"] == x["dT_edge_centre"]]
        min_stick = min(sticks) if sticks else NAN
        min_ratio = min(ratios) if ratios else NAN

        # index_contact_x / _y keep the old definition (ForceResultant[3], [4]) for train_newdata.py;
        # they are most likely the torques Tx, Ty - see the *_Tx / *_Ty columns.
        row = [frame, round(now, 6), round(now - t0_global, 6),
               round(idx_fz, 4), round(rf1[3], 3), round(rf1[4], 3), idx_area, round(idx_shear, 4), round(idx_p, 5),
               round(thm_fz, 4), round(rf2[3], 3), round(rf2[4], 3), thm_area, round(thm_shear, 4), round(thm_p, 5),
               round(mag, 6), round(mmax, 6), round(mmean, 6)]
        for rf, cop in ((rf1, c1), (rf2, c2)):
            row += [round(float(v), 4) for v in rf[:6]] + [round(cop[0], 3), round(cop[1], 3)]
        for x in feats:
            row += [x[k] if isinstance(x[k], int) else round(x[k], 5) for k in FEAT_NAMES]
        row += [round(min_stick, 5), round(min_ratio, 5)]
        row += [new[0], new[1], snap[0]["stale"], snap[1]["stale"], seq[0], seq[1],
                round((now - snap[0]["t"]) * 1000, 1), round((now - snap[1]["t"]) * 1000, 1),
                round(snap[0]["ms"], 1), round(snap[1]["ms"], 1),
                fnew[0], fnew[1], snap[0]["fstale"], snap[1]["fstale"], fseq[0], fseq[1],
                round((now - snap[0]["tf"]) * 1000, 1), round((now - snap[1]["tf"]) * 1000, 1), phase]
        w.writerow(row)
        frame += 1
        if frame % 25 == 0:
            fh.flush()

        live_feat.update(index=feats[0], thumb=feats[1], min_stick_frac=min_stick, min_dT_edge_centre=min_ratio)
        store["index"].append(f1); store["thumb"].append(f2)
        store["index_flow"].append(snap[0]["fl"]); store["thumb_flow"].append(snap[1]["fl"])
        store["t"].append(now - t0_global)
        store["seq"].append(seq); store["new"].append(new)
        store["fseq"].append(fseq); store["force_new"].append(fnew)
        store["stale"].append([snap[0]["stale"], snap[1]["stale"]])
        store["phase"].append(phase)
    fh.flush(); fh.close()
    rospy.loginfo(f"[Xense] Logger stopped after {frame} rows")


def calibrate_info(samples=30):
    """Mean |Fz| with nothing touching (the Xense reads ~0.1-0.3 N here; used for drop detection)."""
    vals = []
    for _ in range(samples):
        vals.append(current_fz()); time.sleep(0.05)
    v = np.array(vals)
    return [float(v[:, 0].mean()), float(v[:, 1].mean())]


# ===================== Cameras =====================
def find_camera_by_serial(serial):
    dl = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dl)
    if ret != 0:
        raise RuntimeError(f"EnumDevices failed, ret={ret}")
    for i in range(dl.nDeviceNum):
        info = dl.pDeviceInfo[i].contents
        if info.nTLayerType == MV_USB_DEVICE:
            sn = bytes(info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode().rstrip("\x00")
            if sn == serial:
                return info
    raise RuntimeError(f"Camera with serial {serial} not found")


def parse_roi(s):
    """'WxH+X+Y' -> (w, h, x, y), snapped to the usual Hikrobot increments (w/h 16 px, offsets 8 px)."""
    try:
        wh, x, y = s.split("+")
        w, h = wh.lower().split("x")
        w, h, x, y = int(w), int(h), int(x), int(y)
    except ValueError:
        raise argparse.ArgumentTypeError("ROI must look like 640x512+320+256")
    w, h = max(64, w // 16 * 16), max(64, h // 16 * 16)
    x, y = x // 8 * 8, y // 8 * 8
    if x + w > CAM_WIDTH or y + h > CAM_HEIGHT:
        raise argparse.ArgumentTypeError(f"ROI {w}x{h}+{x}+{y} exceeds the {CAM_WIDTH}x{CAM_HEIGHT} sensor")
    return w, h, x, y


def frame_is_saved(n, el, cam_name=None):
    if not _phase_saves(n, el):
        return False
    if cam_name == "cam1" and n % CAM1_EVERY:
        return False
    if (low_force_from is not None and el is not None and el >= low_force_from and n % LOW_FORCE_EVERY
            and phase != "regrasp"):                     # v6_20: every frame while the regrasp is verified
        return False
    return True


def _phase_saves(n, el):
    """Camera save policy: all frames in active phases / after full_save_from, else 1 in BASELINE_DECIMATE."""
    if phase in FULL_SAVE_PHASES:
        return True
    if phase in ("settle", "hold"):
        if full_save_from is not None and el is not None and el >= full_save_from:
            if full_save_pause is None or not (full_save_pause[0] <= el < full_save_pause[1]):
                return True
        return n % BASELINE_DECIMATE == 0
    return False      # placement / squeeze / handoff: time-stamped only


def raw_writer(q, out_dir, cam_name, counter):
    """Writes raw frame bytes to disk; no colour conversion on the capture path."""
    while True:
        item = q.get()
        if item is None:
            break
        data, n = item
        with open(os.path.join(out_dir, f"frame_{n:06d}.raw"), "wb") as fh:
            fh.write(data)
        counter["written"] += 1
    print(f"[{cam_name}] writer finished: {counter['written']} frames")


def record_camera(cam_name, serial, trial_dir, cam_meta, sync, roi):
    out_dir = os.path.join(trial_dir, f"camera_{cam_name}")
    os.makedirs(out_dir, exist_ok=True)
    meta = cam_meta[cam_name] = {"serial": serial, "frames": 0, "saved": 0, "queue_full_waits": 0, "sync": sync}
    try:
        info = find_camera_by_serial(serial)
    except Exception as e:  # noqa: BLE001
        print(f"[{cam_name}] {e}"); meta["error"] = str(e); return
    cam = MvCamera()
    if cam.MV_CC_CreateHandle(info) != 0 or cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
        print(f"[{cam_name}] could not open camera"); meta["error"] = "open failed"; return

    w, h, ox, oy = roi if roi else (CAM_WIDTH, CAM_HEIGHT, 0, 0)
    cam.MV_CC_SetIntValue("OffsetX", 0); cam.MV_CC_SetIntValue("OffsetY", 0)
    if cam.MV_CC_SetIntValue("Width", w) != 0 or cam.MV_CC_SetIntValue("Height", h) != 0:
        print(f"[{cam_name}] could not set ROI size {w}x{h}"); meta["roi_error"] = "size"
    if cam.MV_CC_SetIntValue("OffsetX", ox) != 0 or cam.MV_CC_SetIntValue("OffsetY", oy) != 0:
        print(f"[{cam_name}] could not set ROI offset {ox},{oy}"); meta["roi_error"] = "offset"
    meta["roi"] = [w, h, ox, oy]
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    cam.MV_CC_SetFloatValue("ExposureTime", CAM_EXPOSURE_TIME)
    cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Continuous")   # adapt during warm-up only
    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
    if sync == "software":
        cam.MV_CC_SetEnumValue("TriggerMode", 1)
        if cam.MV_CC_SetEnumValueByString("TriggerSource", "Software") != 0:
            print(f"[{cam_name}] could not set software trigger source"); meta["error"] = "trigger source"
    else:
        cam.MV_CC_SetEnumValue("TriggerMode", 0)
    if cam.MV_CC_StartGrabbing() != 0:
        print(f"[{cam_name}] StartGrabbing failed"); meta["error"] = "grab failed"; return

    q = Queue(maxsize=CAM_QUEUE_SIZE)
    counter = {"written": 0}
    wt = threading.Thread(target=raw_writer, args=(q, out_dir, cam_name, counter), daemon=True)
    wt.start()

    cam_handles[cam_name] = cam
    st = MV_FRAME_OUT()
    print(f"[{cam_name}] ready ({sync} sync), waiting for t0 ...")
    while t0_global is None and not stop_flag.is_set():
        if cam.MV_CC_GetImageBuffer(st, 100) == 0:
            cam.MV_CC_FreeImageBuffer(st)                 # keep buffers drained
        time.sleep(0.005)
    cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Off")   # lock colours for marker tracking

    ts_f = open(os.path.join(trial_dir, f"camera_{cam_name}_timestamps.csv"), "w", newline="")
    tw = csv.writer(ts_f)
    tw.writerow(["frame", "wall_time", "elapsed_s", "host_ts", "dev_ts", "phase", "saved"])
    n, fps_t = 0, time.time()
    t_last = fps_t
    try:
        while not stop_flag.is_set() and n < MAX_CAMERA_FRAMES:
            if cam.MV_CC_GetImageBuffer(st, 1000) != 0:
                continue
            now = time.time()
            fi = st.stFrameInfo
            buf = (c_ubyte * fi.nFrameLen)()
            memmove(byref(buf), st.pBufAddr, fi.nFrameLen)
            if n == 0:
                meta.update(width=int(fi.nWidth), height=int(fi.nHeight), pixel_type=hex(int(fi.enPixelType)),
                            frame_len=int(fi.nFrameLen))
                print(f"[{cam_name}] pixel format {meta['pixel_type']}, {fi.nWidth}x{fi.nHeight}")
            host_ts = int(getattr(fi, "nHostTimeStamp", 0))
            dev_ts = (int(getattr(fi, "nDevTimeStampHigh", 0)) << 32) | int(getattr(fi, "nDevTimeStampLow", 0))
            cam.MV_CC_FreeImageBuffer(st)
            el = now - t0_global
            save = frame_is_saved(n, el, cam_name)
            # never let a slow disk stall the grab loop (that drops triggers and desynchronises the cameras):
            # when the write queue backs up, save every 2nd (>50 % full) or every 4th (>80 % full) frame
            fill = q.qsize() / CAM_QUEUE_SIZE
            if save and ((fill > 0.8 and n % 4) or (fill > 0.5 and n % 2)):
                save = False
                meta["thinned"] = meta.get("thinned", 0) + 1
            if save:
                try:
                    q.put_nowait((bytes(buf), n))
                    meta["saved"] += 1
                except Full:
                    save = False
                    meta["queue_full_waits"] += 1
            tw.writerow([n, round(now, 6), round(el, 6), host_ts, dev_ts, phase, int(save)])
            n += 1
            t_last = now
            if n % 500 == 0:
                ts_f.flush()
                print(f"[{cam_name}] {n} frames @ {n / (time.time() - fps_t):.1f} fps, queue {q.qsize()}")
    finally:
        meta["fps"] = round(n / max(1e-6, t_last - fps_t), 1)
        cam_handles.pop(cam_name, None)
        time.sleep(0.05)                       # let the trigger thread notice before the handle closes
        q.put(None); wt.join()
        ts_f.close()
        cam.MV_CC_StopGrabbing()
        if CONVERT_OPTS["sdk"]:
            if not CONVERT_GO.wait(timeout=300):
                print(f"[{cam_name}] tactile data not saved after 300 s - converting anyway", flush=True)
            try:
                sdk_convert(cam, cam_name, out_dir, meta, CONVERT_OPTS["keep_raw"])
            except Exception as e:  # noqa: BLE001
                print(f"[{cam_name}] SDK colour conversion unavailable ({e}) - falling back to OpenCV")
        cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
        meta["frames"] = n
        print(f"[{cam_name}] done: {n} frames ({meta['fps']} fps), saved {meta['saved']}, "
              f"{meta['queue_full_waits']} queue waits")


CONVERT_OPTS = {"sdk": False, "keep_raw": False}     # set in main()
CONVERT_GO = threading.Event()  # set by main() once the tactile data is saved and the Xense sensors are released


def sdk_convert(cam, cam_name, out_dir, meta, keep_raw):
    """Bayer -> BGR with the camera SDK's own colour conversion (MV_CC_ConvertPixelTypeEx, as in v2), then
    PNG. Runs in this process (threads only, no fork), with the camera still open."""
    import glob
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from ctypes import cast, POINTER
    files = sorted(glob.glob(os.path.join(out_dir, "frame_*.raw")))
    if not files or "width" not in meta:
        return
    w, h = int(meta["width"]), int(meta["height"])
    ptype = int(meta["pixel_type"], 16)
    code = _bayer_code(meta["pixel_type"])
    n_dst = w * h * 3
    print(f"[{cam_name}] converting {len(files)} frames to colour PNG (camera SDK) ...", flush=True)
    t = time.time()
    stats = {"sdk": 0, "cv2": 0, "bad": 0}

    def write(path, img):
        if cv2.imwrite(path[:-4] + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1]) and not keep_raw:
            os.remove(path)

    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as pool:
        futs = []
        for path in files:
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size != w * h:
                stats["bad"] += 1
                continue
            src = (c_ubyte * raw.size).from_buffer(raw)
            dst = (c_ubyte * n_dst)()
            prm = MV_CC_PIXEL_CONVERT_PARAM_EX()
            prm.nWidth, prm.nHeight, prm.enSrcPixelType = w, h, ptype
            prm.pSrcData, prm.nSrcDataLen = cast(src, POINTER(c_ubyte)), raw.size
            prm.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            prm.pDstBuffer, prm.nDstBufferSize = dst, n_dst
            if cam.MV_CC_ConvertPixelTypeEx(prm) == 0:
                img = np.frombuffer(dst, dtype=np.uint8, count=n_dst).reshape(h, w, 3).copy()
                stats["sdk"] += 1
            else:
                img = cv2.cvtColor(raw.reshape(h, w), code) if code is not None else raw.reshape(h, w)
                stats["cv2"] += 1
            futs.append(pool.submit(write, path, img))
        for f in futs:
            f.result()
    meta["png_frames"] = stats["sdk"] + stats["cv2"]
    meta["png_conversion"] = stats
    print(f"[{cam_name}] converted {meta['png_frames']}/{len(files)} frames to colour PNG in {time.time() - t:.0f} s "
          f"(SDK {stats['sdk']}, OpenCV fallback {stats['cv2']})", flush=True)


def trigger_thread(trial_dir, meta):
    """Software trigger: fires both cameras back-to-back from one loop; logs every trigger after t0."""
    fh = open(os.path.join(trial_dir, "trigger_log.csv"), "w", newline="")
    w = csv.writer(fh); w.writerow(["trigger", "t_cam1", "t_cam2", "elapsed_s", "phase"])
    k, next_t = 0, time.time()
    spreads, periods, last = [], [], None
    while not stop_flag.is_set():
        fps = WARMUP_FPS if t0_global is None else TRIGGER_FPS
        now = time.time()
        if now < next_t:
            time.sleep(max(0.0, next_t - now - 0.0005))
            while time.time() < next_t:
                pass
        handles = dict(cam_handles)
        if len(handles) < 2:
            next_t = time.time() + 1.0 / fps
            continue
        ts = {}
        for name in ("cam1", "cam2"):
            ts[name] = time.time()
            handles[name].MV_CC_SetCommandValue("TriggerSoftware")
        if t0_global is not None:
            w.writerow([k, round(ts["cam1"], 6), round(ts["cam2"], 6), round(ts["cam1"] - t0_global, 6), phase])
            spreads.append(ts["cam2"] - ts["cam1"])
            if last is not None:
                periods.append(ts["cam1"] - last)
            last = ts["cam1"]; k += 1
        next_t += 1.0 / fps
        if next_t < time.time() - 0.5:
            next_t = time.time()
    fh.close()
    if spreads:
        meta.update(triggers=k, trigger_rate_hz=round(1.0 / float(np.mean(periods)), 1) if periods else None,
                    trigger_period_jitter_ms=round(float(np.std(periods)) * 1000, 3) if periods else None,
                    cam_spread_mean_ms=round(float(np.mean(spreads)) * 1000, 3),
                    cam_spread_max_ms=round(float(np.max(spreads)) * 1000, 3))


# ===================== Raw -> PNG after the trial =====================
# Hikrobot PixelType -> OpenCV code (Hikrobot BayerGB8 = GBRG sensor = cv2.COLOR_BayerGR2BGR).
def _bayer_code(pixel_type):
    import cv2
    return {"0x1080008": cv2.COLOR_BayerGB2BGR, "0x1080009": cv2.COLOR_BayerBG2BGR,
            "0x108000a": cv2.COLOR_BayerGR2BGR, "0x108000b": cv2.COLOR_BayerRG2BGR}.get(pixel_type.lower())


def _convert_one(job):
    import cv2
    path, w, h, code, keep_raw = job
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != w * h:
        return False
    img = raw.reshape(h, w)
    img = cv2.cvtColor(img, code) if code is not None else img
    ok = cv2.imwrite(path[:-4] + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if ok and not keep_raw:
        os.remove(path)
    return ok


# The conversion runs in a FRESH Python process (not a fork of this one): after a --gpu trial this process
# holds CUDA and SDK threads, and forking worker processes from it can deadlock (seen on 24-25 Oct).
_CONVERT_CHILD = r'''
import glob, os, sys, time, numpy as np, cv2
from multiprocessing import Pool
cam_dir, w, h, code, keep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
def conv(p):
    raw = np.fromfile(p, np.uint8)
    if raw.size != w * h:
        return 0
    img = raw.reshape(h, w)
    img = cv2.cvtColor(img, code) if code >= 0 else img
    ok = cv2.imwrite(p[:-4] + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if ok and not keep:
        os.remove(p)
    return int(ok)
if __name__ == "__main__":
    fs = sorted(glob.glob(os.path.join(cam_dir, "frame_*.raw")))
    t = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        n = sum(pool.imap_unordered(conv, fs, chunksize=16))
    print(f"CONVERTED {n} {len(fs)} {time.time() - t:.0f}", flush=True)
'''


def convert_frames(trial_dir, cam_meta, keep_raw):
    import glob
    import subprocess
    for cam, m in cam_meta.items():
        if "width" not in m:
            continue
        cam_dir = os.path.join(trial_dir, f"camera_{cam}")
        n_raw = len(glob.glob(os.path.join(cam_dir, "frame_*.raw")))
        if not n_raw:
            continue
        code = _bayer_code(m["pixel_type"])
        print(f"[{cam}] converting {n_raw} frames to PNG ...", flush=True)
        res = subprocess.run([sys.executable, "-c", _CONVERT_CHILD, cam_dir, str(m["width"]), str(m["height"]),
                              str(-1 if code is None else int(code)), "1" if keep_raw else "0"],
                             capture_output=True, text=True, timeout=1800)
        line = [x for x in res.stdout.splitlines() if x.startswith("CONVERTED")]
        if res.returncode != 0 or not line:
            m["png_error"] = (res.stderr or res.stdout)[-500:]
            print(f"[{cam}] PNG conversion FAILED - raw frames kept:\n{m['png_error']}")
            continue
        _, ok, total, secs = line[-1].split()
        m["png_frames"] = int(ok)
        print(f"[{cam}] converted {ok}/{total} frames to PNG in {secs} s")


# ===================== Experiment sequence =====================
def preflight(log, p, stage):
    """Both sensors must be live and fast enough before the object is placed. Returns False to abort."""
    n0, m0, c0 = sample_counts(), force_counts(), change_counts(); time.sleep(2.0)
    n1, m1, c1 = sample_counts(), force_counts(), change_counts()
    rates = [(b - a) / 2.0 for a, b in zip(n0, n1)]
    frates = [(b - a) / 2.0 for a, b in zip(m0, m1)]
    crates = [(b - a) / 2.0 for a, b in zip(c0, c1)]
    alive = sensor_alive()
    with latest_lock:
        ms = [round(s["ms"], 1) for s in latest]
    rospy.loginfo(f"[Check:{stage}] {p.field} field rate index={rates[0]:.1f} Hz thumb={rates[1]:.1f} Hz | "
                  f"force rate index={frates[0]:.1f} Hz thumb={frates[1]:.1f} Hz | "
                  f"changing samples index={crates[0]:.1f} Hz thumb={crates[1]:.1f} Hz | alive={alive}")
    log.setdefault("preflight", {})[stage] = {"field_rate_hz": rates, "force_rate_hz": frates,
                                              "changing_rate_hz": crates,
                                              "last_read_ms": ms, "alive": alive}
    dead = [SENSOR_NAMES[k] for k in range(2) if not alive[k]]
    if dead and not (p.allow_one_sensor and any(alive)):
        rospy.logerr(f"[Check:{stage}] {dead} sensor is frozen/disconnected. ABORTING before placing the object.")
        rospy.logerr("    Unplug/replug that sensor (or restart the Xense service) and run again.")
        log["aborted"] = f"{dead} sensor dead at {stage}"
        return False
    if dead:
        rospy.logwarn(f"[Check:{stage}] {dead} sensor DEAD - continuing because --allow-one-sensor")
    slow = [f"{SENSOR_NAMES[k]} field" for k in range(2) if alive[k] and rates[k] < p.min_rate]
    slow += [f"{SENSOR_NAMES[k]} changing samples {crates[k]:.1f} Hz (need {MIN_CHANGE_HZ:.0f}; replug it)"
             for k in range(2) if alive[k] and crates[k] < MIN_CHANGE_HZ]
    slow += [f"{SENSOR_NAMES[k]} force" for k in range(2) if alive[k] and frates[k] < p.min_force_rate]
    if slow:
        msg = (f"{slow} too slow (field {rates} Hz, need {p.min_rate}; force {frates} Hz, "
               f"need {p.min_force_rate})")
        if not p.allow_slow:
            rospy.logerr(f"[Check:{stage}] {msg}. ABORTING - too slow to resolve incipient slip. "
                         "Try a larger --force-every, --gpu, or (dry runs only) --allow-slow.")
            log["aborted"] = f"sensor rate too low at {stage}: {msg}"
            return False
        rospy.logwarn(f"[Check:{stage}] {msg} - continuing because --allow-slow")
    return True


class ThrottledPublisher:
    """v6_16: wraps the /cmd_leap publisher. publish() can be called every loop; a message is only sent when
    the pose differs from the last one sent (at most CMD_MAX_HZ) or every CMD_KEEPALIVE_S, so the LEAP node's
    unbounded subscriber queue never builds a backlog."""

    def __init__(self, pub):
        self.pub, self.last_pose, self.t_last, self.n_sent, self.n_calls = pub, None, 0.0, 0, 0

    def publish(self, msg):
        self.n_calls += 1
        now = time.time()
        pose = tuple(round(float(v), 5) for v in msg.position)
        changed = pose != self.last_pose
        if (changed and now - self.t_last >= 1.0 / CMD_MAX_HZ) or now - self.t_last >= CMD_KEEPALIVE_S:
            self.pub.publish(msg)
            self.last_pose, self.t_last = pose, now
            self.n_sent += 1

    def get_num_connections(self):
        return self.pub.get_num_connections()


def latency_check(pub, pose, rate, log):
    """v6_16: hold `pose` until the free joints are within LATENCY_TOL of it. Returns True if reached."""
    if hand is None or not hand.ok:
        log["latency_check"] = {"result": "no joint readings"}
        return True
    t_s = time.time()
    while not rospy.is_shutdown() and time.time() - t_s < LATENCY_MAX_S:
        pub.publish(make_msg(pose)); rate.sleep()
        if hand.fresh(0.3):
            err = max(abs(float(hand.last[j]) - pose[j]) for j in LATENCY_JOINTS)
            if err < LATENCY_TOL:
                log["latency_check"] = {"result": "ok", "s": round(time.time() - t_s, 3)}
                return True
    err = None if not hand.fresh(1.0) else round(max(abs(float(hand.last[j]) - pose[j]) for j in LATENCY_JOINTS), 4)
    log["latency_check"] = {"result": "lagging", "s": LATENCY_MAX_S, "max_err_rad": err}
    return False


def hold_pose(pub, pose, seconds, rate, on_tick=None):
    """Publish a constant pose for `seconds`; on_tick() returning True stops early."""
    t_end = time.time() + seconds
    while time.time() < t_end and not rospy.is_shutdown():
        pub.publish(make_msg(pose)); rate.sleep()
        if on_tick is not None and on_tick():
            return True
    return False


def wait_enter(pub, pose, rate, timeout, on_tick=None):
    """Publish a constant pose until ENTER, timeout, or on_tick() returns True. Returns the reason."""
    done = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), done.set()), daemon=True).start()
    t_end = None if timeout is None else time.time() + timeout
    while not rospy.is_shutdown():
        pub.publish(make_msg(pose)); rate.sleep()
        if done.is_set():
            return "enter"
        if on_tick is not None and on_tick():
            return "drop"
        if t_end is not None and time.time() >= t_end:
            return "timeout"
    return "shutdown"


def settle_gate(pub, pose, rate, min_s):
    """Hold `pose` until the 2-s mean grip force of every live sensor stops drifting.
    Returns (stable, [index_mean_N, thumb_mean_N]) using the last SETTLE_WIN s."""
    hist, t_s, ok_since = [], time.time(), None
    means = list(current_fz())
    while not rospy.is_shutdown():
        pub.publish(make_msg(pose)); rate.sleep()
        now = time.time(); fi, ft = current_fz(); hist.append((now, fi, ft))
        hist = [h for h in hist if h[0] >= now - 2 * SETTLE_WIN]
        arr = np.array(hist)
        last = arr[arr[:, 0] >= now - SETTLE_WIN]
        means = [float(last[:, 1].mean()), float(last[:, 2].mean())]
        if now - t_s >= max(min_s, 2 * SETTLE_WIN) and arr[0, 0] <= now - 1.9 * SETTLE_WIN:
            prev = arr[arr[:, 0] < now - SETTLE_WIN]
            alive = sensor_alive()
            drift = [abs(means[k] - float(prev[:, 1 + k].mean())) for k in range(2) if alive[k]] or [0.0]
            if max(drift) < SETTLE_DRIFT:
                ok_since = ok_since or now
                if now - ok_since >= SETTLE_HOLD:
                    return True, means
            else:
                ok_since = None
        if now - t_s >= SETTLE_MAX:
            return False, means
    return False, means


def set_grip(pub, held, rate, target, max_force):
    """v6_15: move the squeezed DIPs (GRIP_JOINTS) together until the GRIP_WIN-s mean |Fz| of the live
    sensors stays within GRIP_TOL of `target` for GRIP_IN_BAND_S. Modifies `held` in place.
    Returns a dict for trial_info.json."""
    drop = DropDetector()
    hist, steps = [], []
    t_start = time.time()
    t_next = t_start + GRIP_WAIT_BIG
    moved, in_band_since, result, m = 0.0, None, "timeout", float("nan")
    m_last_step, dead_close = None, 0
    while not rospy.is_shutdown():
        pub.publish(make_msg(held)); rate.sleep()
        now = time.time()
        fi, ft = current_fz()
        alive = sensor_alive()
        hist.append((now, fi, ft))
        hist = [h for h in hist if h[0] >= now - GRIP_WIN]
        live = [k for k in range(2) if alive[k]] or [0, 1]
        arr = np.array(hist)
        m = float(np.mean([arr[:, 1 + k].mean() for k in live]))
        if drop.update():
            result = "dropped"; break
        if len(hist) >= 5 and max(arr[:, 1].mean(), arr[:, 2].mean()) > FZ_GLITCH_N:
            result = "glitch"; break                   # v6_25: sensor fault (or a crushing grip) -> abort
        if now - t_start < GRIP_WIN:
            continue                                   # fill the averaging window first
        err = m - target
        if abs(err) <= GRIP_TOL:
            in_band_since = in_band_since or now
            if now - in_band_since >= GRIP_IN_BAND_S:
                result = "reached"; break
            continue
        in_band_since = None
        if now - t_start >= GRIP_SET_MAX:
            result = "timeout"; break
        if now < t_next:
            continue
        flat = m_last_step is not None and abs(m - m_last_step) < GRIP_DEAD_N
        dead = err > 0 and flat
        dead_close = dead_close + 1 if (err < 0 and flat) else 0
        if dead_close >= 5:                            # closing no longer raises the force: motor at its limit
            result = "saturated"; break
        big = abs(err) > GRIP_BIG_ERR or dead
        step = GRIP_STEP_BIG if big else GRIP_STEP_SMALL
        if err > 0:                                    # too tight -> open
            if moved - step < -GRIP_MAX_OPEN:
                result = "open_limit"; break
            d = -step
        else:                                          # too loose -> close
            if max(fi, ft) >= max_force or moved + step > GRIP_MAX_CLOSE:
                result = "close_limit"; break
            d = step
        for j in GRIP_JOINTS:
            held[j] += d
        moved += d
        m_last_step = m
        steps.append([round(elapsed(), 3), round(m, 3), round(d, 4)])
        t_next = now + (GRIP_WAIT_BIG if big else GRIP_WAIT_SMALL)
    fz = current_fz()
    return {"target_N": target, "result": result, "mean_Fz_N": round(m, 4),
            "Fz_end_N": [round(v, 4) for v in fz], "moved_rad": round(moved, 4),
            "duration_s": round(time.time() - t_start, 3), "n_steps": len(steps),
            "steps_elapsed_meanFz_d": steps}


class Status:
    """1 Hz console line with forces and the pre-registered feature."""

    def __init__(self, prereg):
        self.prereg, self.next = prereg, 0.0

    def __call__(self):
        now = time.time()
        if now < self.next:
            return False
        self.next = now + 1.0
        fi, ft = current_fz()
        v = live_feat.get(self.prereg)
        if v is None and "_" in self.prereg:
            s, k = self.prereg.split("_", 1)
            v = live_feat.get(s, {}).get(k)
        vs = "n/a" if v is None or v != v else f"{v:.3f}"
        rospy.loginfo(f"    t={elapsed():6.1f}s  Fz index={fi:.2f} thumb={ft:.2f} N   {self.prereg}={vs}")
        return False


class OnlineSlipDetector:
    """v6_20: the pre-registered offline detector run live (one thread, polls the reader threads every 5 ms).
    Per finger, every NEW marker-field sample: edge lag = |mean displacement of the contact centre - mean displacement
    of the contact edge| (contact / centre / edge masks from the mean |Fz| of the first REF_WIN s after start(), the
    displacement relative to the mean marker positions of that window, smoothed over REGRASP_SMOOTH_N samples - as
    xense_features() in analyze_slip.py) and the summed |Fz| of the force field. arm(t) z-scores both signals against
    the samples from start() to t; afterwards the trigger signal (edgelag: edge lag; force: minus the summed |Fz|)
    fires when z > REGRASP_Z for REGRASP_DUR s on either finger. All samples go to regrasp_detector.csv."""

    def __init__(self, trial_dir, trigger):
        self.trigger = trigger
        self.t_start = self.t_arm = None
        self.fired = threading.Event()
        self.fire = None
        self.lock = threading.Lock()
        self.fin = [{"buf": [], "ready": False, "failed": None, "hist": deque(maxlen=REGRASP_SMOOTH_N),
                     "series": [], "mu": None, "run": None, "nchg": None} for _ in range(2)]
        self.fh = open(os.path.join(trial_dir, "regrasp_detector.csv"), "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["elapsed_s", "finger", "edge_lag_px", "fz_sum_N", "lag_z", "force_z", "armed", "fired"])
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self, t):
        self.t_start = t
        self.thread.start()

    def info(self):
        out = {}
        for k, key in enumerate(SENSOR_KEYS):
            f = self.fin[k]
            out[key] = {"ready": f["ready"], "failed": f["failed"], "n_baseline": f.get("n_base"),
                        "lag_mu": f.get("lag_mu"), "lag_sd": f.get("lag_sd"),
                        "fz_mu": f.get("fz_mu"), "fz_sd": f.get("fz_sd"), "masks": f.get("masks")}
        return out

    def _freeze(self, k):
        f = self.fin[k]
        fz_ref = np.mean([b[2] for b in f["buf"]], axis=0)
        cmask_f = (fz_ref > CONTACT_FRAC * fz_ref.max()) & (fz_ref > 1e-3)
        M = f["buf"][0][1]
        H, W = M.shape[:2]
        h, w_ = cmask_f.shape
        ri = np.clip((np.arange(H) * h / H).astype(int), 0, h - 1)
        ci = np.clip((np.arange(W) * w_ / W).astype(int), 0, w_ - 1)
        cmask = cmask_f[np.ix_(ri, ci)]
        wgt = cmask.astype(float)
        if wgt.sum() < 8:
            f["failed"] = f"contact patch only {int(wgt.sum())} markers"; return
        rr, cc = np.mgrid[0:H, 0:W]
        r0, c0 = (rr * wgt).sum() / wgt.sum(), (cc * wgt).sum() / wgt.sum()
        d = np.hypot(rr - r0, cc - c0)
        req = np.sqrt(wgt.sum() / np.pi)
        f["centre"] = cmask & (d < CENTRE_FRAC * req)
        f["edge"] = cmask & (d >= EDGE_FRAC * req)
        if f["centre"].sum() < 2 or f["edge"].sum() < 2:
            f["failed"] = f"patch too small to split (centre {int(f['centre'].sum())}, edge {int(f['edge'].sum())})"; return
        f["Mref"] = np.mean([b[1] for b in f["buf"]], axis=0)
        f["masks"] = {"contact": int(cmask.sum()), "centre": int(f["centre"].sum()), "edge": int(f["edge"].sum())}
        f["ready"] = True
        buf, f["buf"] = f["buf"], []
        for t, M_, fz_map in buf:                  # the reference window itself is part of the baseline
            self._process(k, t, M_, float(fz_map.sum()))

    def _process(self, k, t, M, fz_sum):
        f = self.fin[k]
        f["hist"].append(M - f["Mref"])
        D = np.mean(f["hist"], axis=0)
        lag = float(np.hypot(*(D[f["centre"]].mean(0) - D[f["edge"]].mean(0))))
        lz = fz_z = None
        fired_now = False
        if f["mu"] is None:
            f["series"].append((t, lag, fz_sum))
        else:
            lz = (lag - f["lag_mu"]) / f["lag_sd"]
            fz_z = (-fz_sum - f["fz_mu"]) / f["fz_sd"]
            sig = lz if self.trigger == "edgelag" else fz_z
            if sig > REGRASP_Z:
                f["run"] = t if f["run"] is None else f["run"]
                if t - f["run"] >= REGRASP_DUR and not self.fired.is_set():
                    self.fire = {"finger": SENSOR_KEYS[k], "onset_elapsed_s": round(f["run"], 4),
                                 "fired_elapsed_s": round(t, 4), "z": round(sig, 3),
                                 "edge_lag_z": round(lz, 3), "force_z": round(fz_z, 3)}
                    self.fired.set(); fired_now = True
            else:
                f["run"] = None
        self.w.writerow([round(t, 4), SENSOR_KEYS[k], round(lag, 5), round(fz_sum, 5),
                         "" if lz is None else round(lz, 3), "" if fz_z is None else round(fz_z, 3),
                         int(f["mu"] is not None), int(fired_now)])

    def arm(self, t):
        """z-score baseline = samples from start() to t (as analyze_slip.py: hold .. loosen)."""
        with self.lock:
            self.t_arm = t
            for k in range(2):
                f = self.fin[k]
                base = [x for x in f["series"] if x[0] < t]
                f["n_base"] = len(base)
                if not f["ready"] or len(base) < REGRASP_MIN_BASE:
                    rospy.logwarn(f"[regrasp] {SENSOR_NAMES[k]}: not armed ({f['failed'] or f'{len(base)} baseline samples'})")
                    continue
                lag = np.array([x[1] for x in base]); fz = -np.array([x[2] for x in base])
                f["lag_mu"], f["lag_sd"] = float(lag.mean()), max(float(lag.std()), 1e-9)
                f["fz_mu"], f["fz_sd"] = float(fz.mean()), max(float(fz.std()), 1e-9)
                f["mu"] = True
                f["series"] = []
            armed = [SENSOR_NAMES[k] for k in range(2) if self.fin[k]["mu"] is not None]
        rospy.loginfo(f"[regrasp] detector armed ({self.trigger}: z > {REGRASP_Z} for {REGRASP_DUR} s) on "
                      f"{', '.join(armed) or 'NO finger'}")
        return bool(armed)

    def _run(self):
        while not stop_flag.is_set() and not rospy.is_shutdown():
            for k in range(2):
                with latest_lock:
                    s = latest[k]
                    nchg, fl, fmap = s["nchg"], s["fl"], s["f"]
                f = self.fin[k]
                if fl is None or fmap is None or nchg == f["nchg"]:
                    continue
                f["nchg"] = nchg
                t = elapsed()
                if t is None or self.t_start is None or t < self.t_start or f["failed"]:
                    continue
                M = np.asarray(fl, dtype=np.float64)[:, :, :2]
                fz_map = np.abs(np.asarray(fmap, dtype=np.float64)[:, :, 2])
                with self.lock:
                    if not f["ready"]:
                        f["buf"].append((t, M, fz_map))
                        if t - self.t_start >= REF_WIN and len(f["buf"]) >= 3:
                            self._freeze(k)
                            if f["failed"]:
                                rospy.logwarn(f"[regrasp] {SENSOR_NAMES[k]}: {f['failed']} - this finger cannot trigger")
                    else:
                        self._process(k, t, M, float(fz_map.sum()))
            time.sleep(0.005)
        self.fh.close()


def regrasp_hold(pub, pose, rate, max_force, seconds, f0):
    """Hold the regrasp pose for `seconds` with drop detection; open REGRASP_BACKOFF rad on the loosened joints
    whenever a finger stays above max_force for 0.3 s. Returns a result dict."""
    drop = DropDetector()
    t_s = time.time()
    over_since, backoffs, t_restored, fmax = None, 0, None, [0.0, 0.0]
    while time.time() - t_s < seconds and not rospy.is_shutdown():
        pub.publish(make_msg(pose)); rate.sleep()
        fz = current_fz()
        fmax = [max(a, b) for a, b in zip(fmax, fz)]
        if t_restored is None and sum(fz) >= REGRASP_RECOVER_FRAC * f0:
            t_restored = time.time() - t_s
        if drop.update():
            return {"result": "dropped", "dropped_after_s": round(time.time() - t_s, 3), "backoffs": backoffs,
                    "grip_restored_s": None if t_restored is None else round(t_restored, 3),
                    "max_Fz_N": [round(v, 3) for v in fmax]}
        if max(fz) > max_force:
            over_since = over_since or time.time()
            if time.time() - over_since >= 0.3:
                for j, wgt in LOOSEN_WEIGHTS.items():
                    pose[j] -= wgt * REGRASP_BACKOFF
                backoffs += 1; over_since = None
                rospy.logwarn(f"[regrasp] grip above {max_force} N - opening {REGRASP_BACKOFF} rad (back-off {backoffs})")
        else:
            over_since = None
    return {"result": "caught", "backoffs": backoffs, "grip_restored_s": None if t_restored is None else round(t_restored, 3),
            "max_Fz_N": [round(v, 3) for v in fmax], "Fz_end_N": [round(v, 3) for v in current_fz()]}


def run_trial(pub, trial_dir, obj, p, log):
    global t0_global, full_save_from, full_save_pause, low_force_from
    rate = rospy.Rate(CMD_RATE)
    OPEN, PINCH = open_pose(), pinch_pose()

    rospy.loginfo("[0] Waiting 3 s for cameras and sensors ...")
    for _ in range(3 * CMD_RATE):
        if rospy.is_shutdown(): return
        rate.sleep()

    # the LEAP hand node must be listening, otherwise every command below is silently lost
    t_end = time.time() + 5.0
    while pub.get_num_connections() == 0 and time.time() < t_end and not rospy.is_shutdown():
        rate.sleep()
    log["hand_subscribers"] = pub.get_num_connections()
    if log["hand_subscribers"] == 0 and not p.no_hand_check:
        rospy.logerr("[1] Nobody is listening on /leaphand_node/cmd_leap - the LEAP hand node is NOT running. "
                     "Start ~/launch_leap.sh in its own terminal, check 'rostopic info /leaphand_node/cmd_leap', "
                     "and run again. ABORTING.")
        log["aborted"] = "LEAP hand node not running (no subscriber on /leaphand_node/cmd_leap)"
        return
    rospy.loginfo(f"[1] Hand node connected ({log['hand_subscribers']} subscriber). Opening hand ...")
    for _ in range(2 * CMD_RATE):
        pub.publish(make_msg(OPEN)); rate.sleep()

    if not p.no_reset_ref:
        reset_pause.set(); time.sleep(0.3)
        res = []
        for k, s in enumerate(SENSOR_OBJS):
            try:
                s.resetReferenceImage(); res.append("ok")
            except Exception as e:  # noqa: BLE001
                res.append(f"failed: {type(e).__name__}: {e}")
        reset_pause.clear(); time.sleep(1.5)
        log["reference_reset"] = dict(zip(SENSOR_NAMES, res))
        (rospy.loginfo if all(r == "ok" for r in res) else rospy.logwarn)(
            f"[Calib] reference images reset (hand open, nothing touching): {log['reference_reset']}")
    rospy.loginfo("[Calib] Reading unloaded sensors - do NOT touch ...")
    unloaded_fz[:] = calibrate_info()
    log["unloaded_fz_N"] = [round(v, 4) for v in unloaded_fz]
    if not preflight(log, p, "open_hand"):
        return

    rospy.loginfo(f"[2] Moving to pinch pose for {obj} ...")
    for q in interpolate(OPEN, PINCH, 80):
        if rospy.is_shutdown(): return
        pub.publish(make_msg(q)); rate.sleep()
    if not preflight(log, p, "pinch_pose"):
        return
    if hand is not None and not hand.ok:
        rospy.loginfo("[2] waiting up to 10 s for the finger positions (/leap_position) ...")
        hand.wait_ok(10.0)
    if not latency_check(pub, PINCH, rate, log):
        rospy.logerr(f"[2] The hand has not reached the pinch pose after {LATENCY_MAX_S:.0f} s - the LEAP node is lagging "
                     "behind the commands. Close ALL other scripts that command the hand, restart ~/launch_leap.sh, "
                     "and run again. ABORTING.")
        log["aborted"] = "hand lagging behind commands (latency check)"
        return
    rospy.loginfo(f"[2] Hand follows commands (pinch pose reached in {log['latency_check'].get('s', '?')} s)")
    if hand is not None and hand.ok:
        off = hand.calibrate(PINCH)
        log["hand_offset_rad"] = None if off is None else [round(float(v), 4) for v in off]
        rospy.loginfo("[2] Finger positions readable from /leap_position (logged to joint_log.csv"
                      + ("; hand control ON)" if p.hand_control else ")"))
    else:
        log["hand_offset_rad"] = None
        rospy.logwarn(f"[2] Finger positions NOT readable ({getattr(hand, 'err', 'no monitor')}) - joint_log.csv will be empty")

    t0_global = time.time()
    log["t0"] = t0_global
    rospy.loginfo(f"[t0] Recording started at {t0_global:.3f}")
    mark("placement")
    if p.mode == "supported":
        rospy.logwarn("[3] CONTROL TRIAL: the support must be under the berry so it CANNOT slip.")
    if p.mode == "pull":
        rospy.logwarn("[3] PULL TRIAL: the thread must already be tied to the berry.")
    rospy.loginfo("[3] (check: mm ruler visible in camera 2 next to the fingertips)")
    rospy.loginfo(f"[3] PLACE {obj.upper()} between the fingertips NOW and keep holding it (3 s)")
    hold_pose(pub, PINCH, 3.0, rate)
    mark("squeeze")

    # ---- squeeze until target normal force (N) ----
    held = list(PINCH); extra = 0.0
    sq_f = open(os.path.join(trial_dir, "squeeze_curve.csv"), "w", newline="")
    sq = csv.writer(sq_f); sq.writerow(["step", "extra_dip_rad", "index_Fz_N", "thumb_Fz_N", "elapsed_s"])
    for step in range(int(MAX_DIP_EXTRA / DIP_SQUEEZE_STEP)):
        extra += DIP_SQUEEZE_STEP
        held[3] = PINCH[3] + extra; held[15] = PINCH[15] + extra
        alive = sensor_alive()
        n_start = force_counts()                  # the squeeze is controlled on force: wait for a fresh force sample
        t_min, t_max = time.time() + SQUEEZE_MIN_WAIT, time.time() + 0.3
        while time.time() < t_max:
            pub.publish(make_msg(held)); rate.sleep()
            n_now = force_counts()
            if time.time() >= t_min and all(n_now[k] > n_start[k] for k in range(2) if alive[k]):
                break
        fi, ft = current_fz()
        live_f = [v for v, a in zip((fi, ft), alive) if a] or [fi, ft]
        sq.writerow([step, round(extra, 4), round(fi, 4), round(ft, 4), round(elapsed(), 4)])
        rospy.loginfo(f"    squeeze {step}: +{extra:.2f} rad  Fz index={fi:.2f} N thumb={ft:.2f} N"
                      + ("" if all(alive) else f"  (alive={alive})"))
        if max(live_f) >= p.max_force:
            rospy.logwarn("[4] Safety force limit reached"); log["squeeze_stop"] = "safety_limit"; break
        gaps = [hand.gap(held, j) for j in (3, 15)] if (hand is not None and p.hand_control) else [None, None]
        if any(g is not None and g > SQUEEZE_STALL_GAP for g in gaps):
            rospy.loginfo(f"[4] Fingers stopped moving (lag {[None if g is None else round(g, 3) for g in gaps]} rad) "
                          "- the grip is set by the object, stopping the squeeze")
            log["squeeze_stop"] = "fingers_stalled"; break
    else:
        log["squeeze_stop"] = "max_extra_reached"
    sq_f.close()
    log["squeeze_extra_rad"] = round(extra, 4)
    log["grip_Fz_after_squeeze_N"] = [round(v, 4) for v in current_fz()]
    log["held_pose"] = [round(v, 5) for v in held]

    # ---- hand-off: operator lets go ----
    mark("handoff")
    if p.mode == "pull":
        rospy.logwarn("[4] Let go of the berry. Hold the THREAD END outside the camera view, slack, and keep still.")
    if p.no_confirm:
        rospy.logwarn("[4] REMOVE YOUR HAND NOW (2 s)")
        hold_pose(pub, held, 2.0, rate)
    else:
        rospy.logwarn("[4] REMOVE YOUR HAND completely, then press ENTER")
        wait_enter(pub, held, rate, None)

    # ---- settling gate (+ trim if the settled grip is above --max-force) ----
    mark("settle")
    t_settle0 = time.time()
    unwind = {}
    if hand is not None and p.hand_control:
        for j in UNWIND_JOINTS:
            g = hand.gap(held, j)
            if g is not None and g > KEEP_GAP:
                held[j] -= g - KEEP_GAP
                unwind[j] = round(g - KEEP_GAP, 4)
    log["unwind_rad"] = unwind
    if unwind:
        rospy.loginfo(f"[5] Removed command wind-up (joint: rad) {unwind} - commands now just past the real finger positions")
    trims = []
    while True:
        stable, means = settle_gate(pub, held, rate, SETTLE_MIN if not trims else 2.0)
        alive = sensor_alive()
        over = [k for k in range(2) if alive[k] and means[k] > p.max_force]
        if not p.hand_control or not stable or not over or len(trims) >= TRIM_MAX:
            break
        for k in over:
            held[TRIM_JOINT[k]] -= TRIM_STEP
        trims.append({"fingers": [SENSOR_KEYS[k] for k in over], "mean_Fz_N": [round(m, 3) for m in means],
                      "elapsed_s": round(elapsed(), 3)})
        rospy.logwarn(f"[5] settled grip {[round(m, 2) for m in means]} N above max {p.max_force} N - opening "
                      f"{[SENSOR_NAMES[k] for k in over]} by {TRIM_STEP} rad (trim {len(trims)})")
    log["settle_s"] = round(time.time() - t_settle0, 3)
    log["settle_stable"] = stable
    log["grip_trims"] = trims
    log["held_pose_after_trim"] = [round(v, 5) for v in held]
    if not stable:
        rospy.logwarn(f"[5] grip force still drifting after {SETTLE_MAX} s - continuing (noted in trial_info)")
    elif over and p.hand_control:
        rospy.logwarn(f"[5] grip still above max force after {TRIM_MAX} trims - trial flagged")
        log["over_max_force"] = True
    elif over:
        rospy.logwarn(f"[5] settled grip is above --max-force ({p.max_force} N) - noted in trial_info")
        log["grip_above_max_force"] = True
    log["grip_Fz_at_hold_N"] = [round(m, 4) for m in means]
    rospy.loginfo(f"[5] grip settled at index {means[0]:.2f} N, thumb {means[1]:.2f} N after {log['settle_s']:.1f} s")

    # ---- v6_15: set the grip to the randomised target force ----
    if not p.no_grip_control:
        log["grip_Fz_after_settle_N"] = log["grip_Fz_at_hold_N"]
        mark("grip_set")
        rospy.loginfo(f"[5] Setting grip to {p.target_force:.2f} N (moving index/thumb DIP together) ...")
        g = set_grip(pub, held, rate, p.target_force, p.max_force)
        log["grip_set"] = g
        log["held_pose_after_grip_set"] = [round(v, 5) for v in held]
        if g["result"] == "glitch":
            rospy.logerr(f"[5] a finger read more than {FZ_GLITCH_N} N while setting the grip (Fz {g['Fz_end_N']}) - "
                         "sensor fault. ABORTING and opening the hand: remove the berry, replug that sensor, run again.")
            log["aborted"] = f"sensor glitch during grip_set (> {FZ_GLITCH_N} N)"
            for q in interpolate(held, OPEN, 40):
                pub.publish(make_msg(q)); rate.sleep()
            return
        if g["result"] == "dropped":
            rospy.logwarn("[5] Object dropped while setting the grip - trial flagged, ending the trial")
            log["dropped_in_grip_set"] = True
            mark("drop"); mark("post")
            hold_pose(pub, held, POST_RELEASE, rate)
            return
        else:
            stable, means = settle_gate(pub, held, rate, 2.0)
            log["settle_after_grip_set_stable"] = stable
            log["grip_Fz_at_hold_N"] = [round(m, 4) for m in means]
            msg = (f"[5] grip set: target {p.target_force:.2f} N, now index {means[0]:.2f} N, thumb {means[1]:.2f} N "
                   f"({g['result']}, DIPs moved {g['moved_rad']:+.3f} rad in {g['duration_s']:.1f} s)")
            (rospy.loginfo if g["result"] == "reached" else rospy.logwarn)(msg)
            if max(means) > FZ_GLITCH_N:
                bad = [SENSOR_NAMES[k] for k in range(2) if means[k] > FZ_GLITCH_N]
                rospy.logerr(f"[5] {bad} reads {max(means):.1f} N - impossible on a blueberry, the sensor is faulty. "
                             "ABORTING and opening the hand: remove the berry, unplug/replug that sensor, run again.")
                log["aborted"] = f"{bad} sensor glitch after grip_set ({max(means):.1f} N > {FZ_GLITCH_N} N)"
                for q in interpolate(held, OPEN, 40):
                    pub.publish(make_msg(q)); rate.sleep()
                return

    status = Status(p.prereg)
    drop = DropDetector()

    def tick_drop():
        status()
        return drop.update()

    if p.mode in ("pull", "nopull"):
        # ---- quiet baseline at constant grip; the cue time is fixed in advance and randomised ----
        full_save_from = elapsed() + max(0.0, p.baseline - PRE_CUE_FULL_S)
        log["cue_planned_elapsed_s"] = round(elapsed() + p.baseline, 4)
        mark("hold")
        rospy.loginfo(f"[5] Quiet baseline {p.baseline:.1f} s at constant grip - nobody touches anything")
        if hold_pose(pub, held, p.baseline, rate, tick_drop):
            rospy.logwarn("[5] Object dropped DURING the quiet baseline - trial flagged")
            log["dropped_in_baseline"] = True
            mark("drop")
        else:
            # ---- cue + load window ----
            print("\a", end="", flush=True)
            mark("load_window")
            log["cue_elapsed_s"] = round(elapsed(), 4)
            if p.mode == "pull":
                rospy.logwarn(f"[6] >>> PULL NOW ({p.load_method}): ONE slow, smooth pull. "
                              + ("Press ENTER when finished." if not p.no_confirm else f"{CONTROL_WINDOW:.0f} s window."))
                if p.no_confirm:
                    why = "drop" if hold_pose(pub, held, CONTROL_WINDOW, rate, tick_drop) else "timeout"
                else:
                    why = wait_enter(pub, held, rate, p.window or LOAD_WINDOW_MAX, tick_drop)
            else:
                rospy.logwarn(f"[6] >>> CONTROL WINDOW: do NOT touch anything ({p.window or CONTROL_WINDOW:.0f} s)")
                why = "drop" if hold_pose(pub, held, p.window or CONTROL_WINDOW, rate, tick_drop) else "timeout"
            log["load_window_end"] = why
            log["load_window_end_elapsed_s"] = round(elapsed(), 4)
            if why == "drop":
                rospy.logwarn("[6] Object dropped"); mark_event("drop_detected")
                log["drop_detected_elapsed_s"] = round(elapsed(), 4)
        mark("post")
        rospy.loginfo(f"[7] Recording {POST_PULL:.0f} s more - keep still")
        hold_pose(pub, held, POST_PULL, rate, status)
        return

    # ---- release / supported: hold then smooth controlled loosening ----
    full_save_from = elapsed()
    if p.hold > HOLD_FULL_HEAD + PRE_CUE_FULL_S:
        full_save_pause = (full_save_from + HOLD_FULL_HEAD, full_save_from + p.hold - PRE_CUE_FULL_S)
    mark("hold")
    det = None
    if p.regrasp != "none":
        det = OnlineSlipDetector(trial_dir, p.regrasp)
        det.start(elapsed())
    rospy.loginfo(f"[5] Holding for {p.hold:.2f} s (baseline)")
    gl = {"hist": [], "bad": None}

    def hold_tick():                                       # v6_24: sensor-glitch guard during the hold
        status()
        gl["hist"] = (gl["hist"] + [current_fz()])[-CMD_RATE:]
        m = [sum(f[k] for f in gl["hist"]) / len(gl["hist"]) for k in range(2)]
        if len(gl["hist"]) >= CMD_RATE // 2 and max(m) > FZ_GLITCH_N:
            gl["bad"] = ([SENSOR_NAMES[k] for k in range(2) if m[k] > FZ_GLITCH_N], max(m))
            return True
        return False

    hold_pose(pub, held, p.hold, rate, hold_tick)
    if gl["bad"]:
        rospy.logerr(f"[5] {gl['bad'][0]} reads {gl['bad'][1]:.1f} N during the hold - impossible on a blueberry, the "
                     "sensor is faulty. ABORTING and opening the hand: remove the berry, unplug/replug that sensor.")
        log["aborted"] = f"{gl['bad'][0]} sensor glitch during hold ({gl['bad'][1]:.1f} N > {FZ_GLITCH_N} N)"
        for q in interpolate(held, OPEN, 40):
            pub.publish(make_msg(q)); rate.sleep()
        return

    mark("loosen")
    if det is not None:
        log["regrasp_armed"] = det.arm(elapsed())
    rospy.loginfo(f"[6] Loosening at {p.rate:.3f} rad/s until the object drops (max {LOOSEN_MAX} rad), "
                  f"joints:weights {LOOSEN_WEIGHTS}, released stop after {RELEASED_CFG.get('s', RELEASED_LOOSEN_S):.0f} s low")
    log["released_loosen_s"] = RELEASED_CFG.get("s", RELEASED_LOOSEN_S)
    lf = open(os.path.join(trial_dir, "slip_curve.csv"), "w", newline="")
    lw = csv.writer(lf); lw.writerow(["tick", "loosen_rad", "index_Fz_N", "thumb_Fz_N", "elapsed_s"])
    log["loosen_start_elapsed_s"] = round(elapsed(), 4)
    base = list(held); amount = 0.0; tick = 0
    inc = p.rate * LOOSEN_TICK
    act0 = None if hand is None or not hand.fresh() else hand.last.copy()
    warned_follow = False
    t_loosen = elapsed(); moved_logged = False
    f0 = sum(current_fz()); low_since = None; fhist = []; released = False
    u0 = sum(unloaded_fz); f0n = f0 - u0                  # v6_22: force above the unloaded readings
    log["loosen_start_net_Fz_N"] = round(f0n, 4)
    fired = False
    drop_seen = None                                      # v6_27

    def loosen_tick():
        return drop.update() or (det is not None and det.fired.is_set())

    while amount < LOOSEN_MAX and not rospy.is_shutdown():
        amount = min(LOOSEN_MAX, amount + inc); tick += 1
        if act0 is not None and amount >= 0.1 and not warned_follow and hand.fresh():
            moved = abs(float(hand.last[3] - act0[3])) + abs(float(hand.last[15] - act0[15]))
            if moved < 0.02:
                rospy.logwarn(f"[6] fingers have not moved after {amount:.2f} rad of loosening (still pressed / stalled?)")
                warned_follow = True
        if act0 is not None and not moved_logged and hand.fresh():
            mv = sum(abs(float(hand.last[j] - act0[j])) for j in (2, 3, 14, 15))
            if mv > HAND_MOVE_RAD:
                moved_logged = True
                log["hand_move_after_loosen_s"] = round(elapsed() - t_loosen, 3)
                log["hand_move_loosen_rad"] = round(amount, 4)
                rospy.loginfo(f"[6] fingers started to move {elapsed() - t_loosen:.1f} s after loosening began "
                              f"({amount:.2f} rad commanded)")
        for j, wgt in LOOSEN_WEIGHTS.items():
            held[j] = base[j] - wgt * amount
        stop = hold_pose(pub, held, LOOSEN_TICK, rate, None if drop_seen is not None else loosen_tick)
        fired = det is not None and det.fired.is_set()
        dropped = stop and not fired
        if fired:
            break
        fi, ft = current_fz()
        fhist = (fhist + [fi + ft])[-int(round(1.0 / LOOSEN_TICK)):]
        fs = sum(fhist) / len(fhist)                  # 1-s moving average of the summed force
        if f0n > 0.2:
            if fs - u0 < LOW_FORCE_FRAC * f0n:
                low_since = elapsed() if low_since is None else low_since
                if low_force_from is None and elapsed() - low_since >= LOW_FORCE_S:
                    low_force_from = elapsed(); mark_event("low_force")
                    log["low_force_elapsed_s"] = round(low_force_from, 4)
                    rospy.loginfo(f"[6] grip force below {LOW_FORCE_FRAC:.0%} of start for {LOW_FORCE_S:.0f} s - "
                                  f"camera 2 now saves 1 frame in {LOW_FORCE_EVERY}")
                if elapsed() - low_since >= RELEASED_CFG.get("s", RELEASED_LOOSEN_S):
                    released = True
            else:
                low_since = None
        lw.writerow([tick, round(amount, 4), round(fi, 4), round(ft, 4), round(elapsed(), 4)])
        if tick % 5 == 0:
            rospy.loginfo(f"    loosen {amount:.3f} rad  Fz index={fi:.2f} thumb={ft:.2f}")
        if dropped:
            if drop_seen is None:
                drop_seen = elapsed()
                rospy.logwarn(f"[6] Object dropped at {amount:.3f} rad (force) - loosening {DROP_CONFIRM_S:.0f} s more "
                              "in case it is still resting on a finger"); mark("drop")
                log["drop_detected_elapsed_s"] = round(elapsed(), 4)
        if drop_seen is not None and elapsed() - drop_seen >= DROP_CONFIRM_S:
            rospy.loginfo(f"[6] stopping at {amount:.3f} rad, {DROP_CONFIRM_S:.0f} s after the drop detection")
            break
        if released:
            rospy.loginfo(f"[6] grip released at {amount:.3f} rad (1-s mean force < {LOW_FORCE_FRAC:.0%} of start for "
                          f"{RELEASED_CFG.get('s', RELEASED_LOOSEN_S):.0f} s) - stopping")
            mark_event("released"); log["released_during_loosen_elapsed_s"] = round(elapsed(), 4)
            break
    else:
        if not rospy.is_shutdown():
            rospy.logwarn(f"[6] reached {LOOSEN_MAX} rad without a drop - holding this pose and recording until the "
                          f"berry drops or the grip is released (max {WAIT_AFTER_LOOSEN:.0f} s)")
            state = {"low_since": None, "why": None, "hist": []}

            def after_tick():
                global low_force_from
                if det is not None and det.fired.is_set():
                    state["why"] = "fired"; return True
                if drop.update():
                    state["why"] = "drop"; return True
                state["hist"] = (state["hist"] + [sum(current_fz())])[-CMD_RATE:]
                if f0n > 0.2 and sum(state["hist"]) / len(state["hist"]) - u0 < LOW_FORCE_FRAC * f0n:
                    state["low_since"] = state["low_since"] or elapsed()
                    if low_force_from is None and elapsed() - state["low_since"] >= LOW_FORCE_S:
                        low_force_from = elapsed(); mark_event("low_force")
                        log["low_force_elapsed_s"] = round(low_force_from, 4)
                    if elapsed() - state["low_since"] >= RELEASED_S:
                        state["why"] = "released"; return True
                else:
                    state["low_since"] = None
                return False

            t_wait = elapsed()
            hold_pose(pub, held, WAIT_AFTER_LOOSEN, rate, after_tick)
            log["wait_after_loosen_s"] = round(elapsed() - t_wait, 2)
            log["wait_after_loosen_end"] = state["why"] or "timeout"
            fired = state["why"] == "fired"
            if state["why"] == "drop":
                rospy.logwarn("[6] Object dropped after loosening ended"); mark("drop")
                log["drop_detected_elapsed_s"] = round(elapsed(), 4)
            elif state["why"] == "released":
                rospy.loginfo(f"[6] grip released (force < {LOW_FORCE_FRAC:.0%} of start for {RELEASED_S:.0f} s) "
                              f"{elapsed() - t_wait:.1f} s after loosening ended")
            else:
                rospy.logwarn(f"[6] berry still held {WAIT_AFTER_LOOSEN:.0f} s after loosening ended - "
                              "this trial probably has no slip (noted in trial_info)")
    lf.close()
    log["loosen_total_rad"] = round(amount, 4)
    log["loosen_end_elapsed_s"] = round(elapsed(), 4)
    if det is not None:
        rg = {"trigger": p.regrasp, "extra_rad": p.regrasp_extra, "detector": det.info()}
        if fired:
            fz_at = current_fz()
            dropped_already = drop.update()
            mark("regrasp")
            rg.update(fire=det.fire, loosen_rad_at_fire=round(amount, 4),
                      Fz_at_fire_N=[round(v, 3) for v in fz_at], loosen_start_Fz_sum_N=round(f0, 3),
                      command_elapsed_s=round(elapsed(), 4), latency_fire_to_command_s=round(elapsed() - det.fire["fired_elapsed_s"], 3))
            for j, wgt in LOOSEN_WEIGHTS.items():
                held[j] = base[j] + wgt * p.regrasp_extra
            rospy.logwarn(f"[regrasp] SLIP DETECTED ({p.regrasp}, {det.fire['finger']}, z={det.fire['z']}) after "
                          f"{amount:.3f} rad of loosening - closing to {p.regrasp_extra:+.3f} rad past the start pose")
            res = regrasp_hold(pub, held, rate, p.max_force, REGRASP_HOLD_S, max(f0, 0.1))
            res["force_low_at_fire"] = bool(dropped_already)
            rg.update(res)
            rg["regrasp_pose"] = [round(v, 5) for v in held]
            if res["result"] == "caught":
                rospy.loginfo(f"[regrasp] CAUGHT: berry still held {REGRASP_HOLD_S:.0f} s after the regrasp "
                              f"(grip back to {REGRASP_RECOVER_FRAC:.0%} in {res['grip_restored_s']} s, Fz {res['Fz_end_N']} N)")
            else:
                rospy.logwarn(f"[regrasp] DROPPED {res['dropped_after_s']} s after the regrasp command"); mark("drop")
                log["drop_detected_elapsed_s"] = round(elapsed(), 4)
        else:
            rg["result"] = "missed" if log.get("drop_detected_elapsed_s") else "no_detection"
            rospy.logwarn(f"[regrasp] detector did not fire before the trial ended ({rg['result']})")
        log["regrasp"] = rg
    mark("post")
    hold_pose(pub, held, POST_RELEASE, rate)


# ===================== QC =====================
def quality_check(log, store, cam_meta, p):
    """Rule-based keep/reject verdict written into trial_info.json."""
    reasons = []
    if log.get("aborted"):
        reasons.append(f"aborted: {log['aborted']}")
    notes = []
    if not log.get("settle_stable", False) and not log.get("aborted"):
        notes.append("grip force was still drifting when the hold started (settle_stable=false)")
    if log.get("grip_above_max_force"):
        notes.append("settled grip above --max-force")
    g = log.get("grip_set")
    if g and g.get("result") not in ("reached", "dropped"):
        notes.append(f"grip target {g['target_N']} N not reached ({g['result']}); achieved mean {g['mean_Fz_N']} N")
    if log.get("dropped_in_grip_set"):
        reasons.append("object dropped while the grip was being set")
    if log.get("dropped_in_baseline"):
        reasons.append("object dropped during the quiet baseline")
    if log.get("over_max_force"):
        reasons.append(f"grip stayed above --max-force after {TRIM_MAX} trims")
    for k, key in enumerate(SENSOR_KEYS):
        r = log.get(f"{key}_rate_hz")
        if r is not None and r < p.min_rate:
            reasons.append(f"{key} field rate {r} Hz < {p.min_rate} Hz")
        fr = log.get(f"{key}_force_rate_hz")
        if fr is not None and fr < p.min_force_rate:
            reasons.append(f"{key} force rate {fr} Hz < {p.min_force_rate} Hz")
    # fresh fraction in the analysis window: cue-2 s .. end (pull/nopull) or loosen start .. end
    t = np.array(store["t"]) if store["t"] else np.array([])
    start = log.get("cue_elapsed_s", log.get("loosen_start_elapsed_s"))
    fresh = {}
    if t.size and start is not None:
        win = t >= start - 2.0
        if win.any():
            new = np.array(store["new"])[win]; stale = np.array(store["stale"])[win]
            for k, key in enumerate(SENSOR_KEYS):
                ok = (new[:, k] == 1) & (stale[:, k] == 0)
                times = t[win][ok]
                gaps = np.diff(times) if times.size > 1 else np.array([np.inf])
                fresh[key] = {"fresh_rows": int(ok.sum()), "window_rows": int(win.sum()),
                              "max_gap_s": round(float(gaps.max()), 3) if gaps.size else None}
                if times.size < 2 or gaps.max() > 0.2:
                    reasons.append(f"{key} sensor has a {fresh[key]['max_gap_s']} s gap without fresh data in the analysis window")
    for cam in ("cam1", "cam2"):
        m = cam_meta.get(cam, {})
        if m.get("error"):
            reasons.append(f"{cam}: {m['error']}")
        elif m.get("frames", 0) == 0:
            reasons.append(f"{cam}: no frames")
    if p.sync == "software" and not log.get("camera_frames_match", True):
        reasons.append("camera frame counts differ")
    if p.mode in ("pull", "nopull") and p.prereg.split("_", 1)[-1] in FLOW_FEATURES | {"stick_frac"}:
        if not log.get("flow_output"):
            reasons.append("pre-registered feature needs the displacement field, which was not recorded")
    refs = log.get("reference", {})
    for key in SENSOR_KEYS:
        if refs.get(key, {}).get("failed"):
            reasons.append(f"{key} reference patch failed: {refs[key]['failed']}")
    return {"pass": not reasons, "reasons": reasons, "notes": notes, "analysis_window_fresh": fresh}


def list_outputs():
    names = [n for n in dir(Sensor.OutputType) if not n.startswith("_")]
    print("Sensor.OutputType members available in this xensesdk:")
    for n in names:
        print("   ", n)


def main():
    ap = argparse.ArgumentParser(description="Synchronised tactile + camera capture for incipient-slip detection")
    ap.add_argument("object_name", nargs="?", default="blueberry")
    ap.add_argument("--mode", choices=["pull", "nopull", "release", "supported"], required=False,
                    help="pull = constant grip + operator tangential load (main protocol); nopull = its control; "
                         "release = robot loosens; supported = release control with a support")
    ap.add_argument("--load-method", choices=["thread", "weight", "tilt", "other"], default="thread",
                    help="pull mode: how the tangential load is applied (logged)")
    ap.add_argument("--baseline", type=float, default=None, help="pull/nopull: quiet baseline before the cue (s); random 15-25 if omitted")
    ap.add_argument("--window", type=float, default=None, help="pull: max s to wait for ENTER; nopull: control window (s)")
    ap.add_argument("--hold", type=float, default=None, help="release/supported: hold before loosening (s); random if omitted")
    ap.add_argument("--rate", type=float, default=None, help="release/supported: loosening speed (rad/s); random if omitted")
    ap.add_argument("--target-force", type=float, default=None,
                    help=f"grip target Fz (N), set after the hand-off; random from {FORCE_CHOICES} if omitted")
    ap.add_argument("--max-force", type=float, default=4.0, help="safety limit on either Fz (N)")
    ap.add_argument("--no-grip-control", action="store_true", help="do not set the grip force (v6_14 behaviour)")
    ap.add_argument("--regrasp", choices=["none", "edgelag", "force", "random"], default="none",
                    help="release mode: regrasp when the live detector fires (edgelag = pre-registered edge lag; "
                         "force = normal-force drop, baseline trigger; random = one of the two, chosen per trial)")
    ap.add_argument("--regrasp-extra", type=float, default=REGRASP_EXTRA,
                    help="rad closed beyond the pre-loosening pose at the regrasp")
    ap.add_argument("--released-s", type=float, default=None,
                    help="release mode: s of low grip force before loosening stops (default 15; 60 for small berries)")
    ap.add_argument("--no-reset-ref", action="store_true",
                    help="do NOT reset the Xense reference images at the start (v6_26 resets them by default)")
    ap.add_argument("--loosen-weights", default=None,
                    help="release mode: joints opened during loosening as j:w pairs, e.g. 1:0.5,2:0.5,3:1.0,14:0.5,15:1.0 "
                         f"(default {LOOSEN_WEIGHTS})")
    ap.add_argument("--berry-id", default=None, help="REQUIRED: label of this physical berry, e.g. B17a")
    ap.add_argument("--diameter-mm", type=float, default=None, help="REQUIRED: caliper diameter of this berry (mm)")
    ap.add_argument("--prereg", default=DEFAULT_PREREG,
                    help="the ONE pre-registered feature column for the lead-time claim (logged before the trial)")
    ap.add_argument("--field", choices=["marker", "mesh"], default="marker",
                    help="displacement field: marker = Marker2D (fast on CPU, default); mesh = Mesh3DFlow (needs --gpu for speed)")
    ap.add_argument("--flow-output", default=None, help="override the Xense output name used for the field")
    ap.add_argument("--force-every", type=int, default=None,
                    help=f"read Force every N sensor loops (default {FORCE_EVERY} on CPU, 1 when the GPU is active)")
    ap.add_argument("--disp-min", type=float, default=None, help="no-load displacement threshold for stick_frac (field units)")
    ap.add_argument("--gpu", action="store_true", help="run the Xense network on the GPU using ~/cuda11_libs (see header)")
    ap.add_argument("--list-outputs", action="store_true", help="print the Xense output types available and exit")
    ap.add_argument("--min-rate", type=float, default=MIN_RATE_HZ, help="minimum field rate per sensor (Hz) at pre-flight")
    ap.add_argument("--min-force-rate", type=float, default=MIN_FORCE_RATE_HZ, help="minimum force rate per sensor (Hz)")
    ap.add_argument("--allow-slow", action="store_true", help="continue even if a sensor is too slow (dry runs only)")
    ap.add_argument("--roi", type=parse_roi, default=None, help="camera crop WxH+X+Y, e.g. 640x512+320+256")
    ap.add_argument("--out-root", default=OUT_ROOT, help="folder that holds <object>/trial_<timestamp>")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--note", default="", help="free text saved in trial_info.json (e.g. berry id, diameter)")
    ap.add_argument("--allow-one-sensor", action="store_true", help="do not abort if one sensor is dead")
    ap.add_argument("--keep-raw", action="store_true", help="keep the .raw files after PNG conversion")
    ap.add_argument("--no-convert", action="store_true", help="do not convert frames to PNG after the trial")
    ap.add_argument("--sync", choices=["software", "free"], default="software", help="camera synchronisation")
    ap.add_argument("--no-confirm", action="store_true", help="no ENTER prompts (fixed 2 s hand-off, fixed load window)")
    ap.add_argument("--serial-sensors", action="store_true", help="read the two sensors one at a time (v2 behaviour)")
    ap.add_argument("--no-hand-check", action="store_true", help="do not abort when the LEAP hand node is not subscribed")
    ap.add_argument("--hand-control", action="store_true",
                    help="experimental: squeeze stall stop, wind-up removal and force trims (OFF = v2 hand behaviour)")
    p = ap.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if p.released_s:
        RELEASED_CFG["s"] = float(p.released_s)
    if p.loosen_weights:                                   # v6_21: per-trial loosening joints (logged in trial_info)
        try:
            lw_new = {int(a): float(b) for a, b in (x.split(":") for x in p.loosen_weights.split(",") if x.strip())}
        except ValueError:
            ap.error("--loosen-weights must look like 1:0.5,2:0.5,3:1.0,14:0.5,15:1.0")
        if not lw_new or any(j < 0 or j > 15 for j in lw_new):
            ap.error("--loosen-weights: joint numbers must be 0-15")
        LOOSEN_WEIGHTS.clear(); LOOSEN_WEIGHTS.update(lw_new)

    if p.list_outputs:
        list_outputs(); return
    if p.mode is None:
        sys.exit("--mode is required: pull | nopull | release | supported")
    if not p.berry_id or p.diameter_mm is None:
        sys.exit("--berry-id and --diameter-mm are required, e.g. --berry-id B17a --diameter-mm 16.9")

    global disp_min_active
    p.flow_output = p.flow_output or FIELD_OUTPUTS[p.field]
    flow_type = getattr(Sensor.OutputType, p.flow_output, None)
    if flow_type is None:
        list_outputs()
        sys.exit(f"Xense output '{p.flow_output}' not found. Pick one above with --flow-output.")
    disp_min_active = p.disp_min if p.disp_min is not None else DISP_MIN[p.field]
    if p.field == "mesh" and not p.gpu:
        rospy.logwarn("--field mesh without --gpu runs the network on the CPU (about 6-7 Hz with two sensors)")

    rng = random.Random(p.seed)
    if p.target_force is None: p.target_force = rng.choice(FORCE_CHOICES)
    if p.target_force >= p.max_force:
        sys.exit("--target-force must be below --max-force")
    p.regrasp_random = p.regrasp == "random"
    if p.regrasp_random:
        p.regrasp = rng.choice(["edgelag", "force"])
    if p.regrasp != "none" and p.mode != "release":
        sys.exit("--regrasp works with --mode release only")
    if p.hold is None: p.hold = round(rng.uniform(*HOLD_RANGE), 2)
    if p.regrasp != "none" and p.hold < REGRASP_MIN_HOLD_S:
        p.hold = REGRASP_MIN_HOLD_S
    if p.rate is None: p.rate = rng.choice(RATE_CHOICES)
    if p.baseline is None: p.baseline = round(rng.uniform(*BASELINE_RANGE), 2)

    rospy.init_node("unified_grasp_capture_v6", anonymous=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(p.out_root, p.object_name, f"trial_{run_ts}")
    os.makedirs(trial_dir, exist_ok=True)
    free_gb = shutil.disk_usage(trial_dir).free / 1e9
    if free_gb < MIN_FREE_GB:
        sys.exit(f"Only {free_gb:.1f} GB free on {p.out_root} (need {MIN_FREE_GB} GB). Free space and run again.")

    timing = (f"baseline {p.baseline} s, load method {p.load_method}" if p.mode in ("pull", "nopull")
              else f"hold {p.hold} s, loosen {p.rate} rad/s" + ("" if p.regrasp == "none" else f", REGRASP: {p.regrasp}"))
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"  {p.object_name.upper()} trial {run_ts} [{p.mode}]: berry {p.berry_id} ({p.diameter_mm} mm), "
                  f"grip target {p.target_force} N" + (" (NOT applied: --no-grip-control)" if p.no_grip_control else "")
                  + f", {timing}")
    rospy.loginfo(f"  Pre-registered feature: {p.prereg}   field: {p.flow_output}   GPU requested: {p.gpu}")
    rospy.loginfo(f"  Saving to {trial_dir}  ({free_gb:.0f} GB free)")
    rospy.loginfo("=" * 60)

    log = {"object": p.object_name, "run_ts": run_ts, "note": p.note, "script": SCRIPT_NAME, "mode": p.mode,
           "berry_id": p.berry_id, "diameter_mm": p.diameter_mm, "grip_control": not p.no_grip_control,
           "cmd_publish": {"max_hz": CMD_MAX_HZ, "keepalive_s": CMD_KEEPALIVE_S, "loop_hz": CMD_RATE},
           "force_choices_N": FORCE_CHOICES,
           "grip_rule": {"joints": list(GRIP_JOINTS), "window_s": GRIP_WIN, "tol_N": GRIP_TOL,
                         "in_band_s": GRIP_IN_BAND_S, "step_big_rad": GRIP_STEP_BIG, "step_small_rad": GRIP_STEP_SMALL,
                         "big_err_N": GRIP_BIG_ERR, "max_open_rad": GRIP_MAX_OPEN, "max_close_rad": GRIP_MAX_CLOSE,
                         "max_s": GRIP_SET_MAX},
           "preregistered_feature": p.prereg, "load_method": p.load_method if p.mode == "pull" else None,
           "baseline_s": p.baseline if p.mode in ("pull", "nopull") else None,
           "hold_s": p.hold if p.mode in ("release", "supported") else None,
           "loosen_rate_rad_s": p.rate if p.mode in ("release", "supported") else None,
           "loosen_tick_s": LOOSEN_TICK, "loosen_cap_rad": LOOSEN_CAP, "loosen_weights": LOOSEN_WEIGHTS,
           "target_force_N": p.target_force, "max_force_N": p.max_force, "seed": p.seed,
           "regrasp_mode": p.regrasp, "regrasp_randomised": p.regrasp_random, "regrasp_rule": None if p.regrasp == "none" else
               {"z": REGRASP_Z, "dur_s": REGRASP_DUR, "smooth_n": REGRASP_SMOOTH_N, "min_base": REGRASP_MIN_BASE,
                "extra_rad": p.regrasp_extra, "hold_s": REGRASP_HOLD_S, "recover_frac": REGRASP_RECOVER_FRAC,
                "backoff_rad": REGRASP_BACKOFF},
           "flow_output": p.flow_output, "field": p.field,
           "exposure_us": CAM_EXPOSURE_TIME, "squeeze_step_rad": DIP_SQUEEZE_STEP,
           "serial_sensors": p.serial_sensors, "camera_sync": p.sync, "roi": p.roi,
           "trigger_fps": TRIGGER_FPS if p.sync == "software" else None,
           "camera_save_rule": {"baseline_decimate": BASELINE_DECIMATE, "pre_cue_full_s": PRE_CUE_FULL_S,
                                 "cam1_every": CAM1_EVERY, "low_force_frac": LOW_FORCE_FRAC, "low_force_s": LOW_FORCE_S,
                                 "low_force_every": LOW_FORCE_EVERY, "hold_full_head_s": HOLD_FULL_HEAD,
                                "full_save_phases": sorted(FULL_SAVE_PHASES)},
           "feature_params": {"contact_frac": CONTACT_FRAC, "fz_min_taxel": FZ_MIN_TAXEL,
                              "min_contact_taxels": MIN_CONTACT_TAXELS, "centre_frac": CENTRE_FRAC,
                              "edge_frac": EDGE_FRAC, "stick_tol": STICK_TOL, "disp_min": disp_min_active, "noise_k": NOISE_K,
                              "ref_window_s": REF_WIN},
           "drop_rule": {"margin_N": DROP_MARGIN, "time_s": DROP_TIME},
           "settle_rule": {"min_s": SETTLE_MIN, "max_s": SETTLE_MAX, "window_s": SETTLE_WIN, "drift_N": SETTLE_DRIFT,
                           "hold_s": SETTLE_HOLD, "trim_step_rad": TRIM_STEP, "trim_max": TRIM_MAX},
           "min_rate_hz": p.min_rate, "min_force_rate_hz": p.min_force_rate,
           "hand_rule": {"squeeze_stall_gap_rad": SQUEEZE_STALL_GAP, "keep_gap_rad": KEEP_GAP,
                         "unwind_joints": list(UNWIND_JOINTS), "loosen_max_rad": LOOSEN_MAX},
           "sensors": {"index": INDEX_SENSOR_ID, "thumb": THUMB_SENSOR_ID}}
    # write the pre-registration to disk before anything is recorded
    with open(os.path.join(trial_dir, "trial_info.json"), "w") as fh:
        json.dump(log, fh, indent=2)

    s1 = Sensor.create(INDEX_SENSOR_ID); time.sleep(2.0)
    s2 = Sensor.create(THUMB_SENSOR_ID); time.sleep(2.0)
    SENSOR_OBJS[:] = [s1, s2]

    gpu_active = bool(GPU_STATUS["sessions"]) and all(s and s[0] == "CUDAExecutionProvider"
                                                      for s in GPU_STATUS["sessions"])
    if p.gpu:
        if gpu_active:
            rospy.loginfo(f"[gpu] ACTIVE: all {len(GPU_STATUS['sessions'])} Xense network sessions run on CUDA")
        else:
            rospy.logwarn(f"[gpu] requested but NOT active ({GPU_STATUS.get('error') or 'CUDA provider did not load'}) "
                          "- running on the CPU")
    if p.force_every is None:
        p.force_every = 1 if gpu_active else FORCE_EVERY
    log["gpu"] = {"requested": p.gpu, "active": gpu_active, "libs_loaded": GPU_STATUS["libs_loaded"],
                  "session_providers": GPU_STATUS["sessions"], "error": GPU_STATUS["error"]}
    log["force_every"] = p.force_every
    rospy.loginfo(f"  Reading {p.flow_output} every loop and Force every {p.force_every} loop(s)")

    # no-contact marker / field positions, once per sensor (reference for offline analysis)
    field_init = {}
    init_name = {"Marker2D": "Marker2DInit", "Mesh3DFlow": "Mesh3DInit"}.get(p.flow_output)
    if init_name and hasattr(Sensor.OutputType, init_name):
        for key, s in zip(SENSOR_KEYS, (s1, s2)):
            try:
                r = s.selectSensorInfo(getattr(Sensor.OutputType, init_name))
                field_init[f"{key}_field_init"] = np.asarray(r[0] if isinstance(r, (tuple, list)) else r)
            except Exception as e:  # noqa: BLE001
                rospy.logwarn(f"[Xense] could not read {init_name} for {key}: {e}")

    store = {"index": [], "thumb": [], "index_flow": [], "thumb_flow": [], "t": [], "seq": [],
             "new": [], "stale": [], "phase": [], "fseq": [], "force_new": []}
    refs = [FieldReference("Index"), FieldReference("Thumb")]
    readers = [threading.Thread(target=sensor_reader, args=(k, s, p.serial_sensors, flow_type, p.force_every),
                                daemon=True)
               for k, s in enumerate((s1, s2))]
    logger = threading.Thread(target=tactile_logger, args=(trial_dir, store, refs), daemon=True)
    for t in readers + [logger]: t.start()

    CONVERT_OPTS.update(sdk=not p.no_convert, keep_raw=p.keep_raw)
    cam_meta = {}
    cams = [threading.Thread(target=record_camera, args=(n, s, trial_dir, cam_meta, p.sync, p.roi))
            for n, s in CAMERA_SERIALS.items()]
    for t in cams: t.start()
    trig_meta, trig = {}, None
    if p.sync == "software":
        trig = threading.Thread(target=trigger_thread, args=(trial_dir, trig_meta), daemon=True); trig.start()

    pub = ThrottledPublisher(rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size=10))
    global hand
    hand = HandMonitor(trial_dir)
    log["hand_monitor"] = "ok" if hand.ok else hand.err
    threading.Thread(target=hand.run, daemon=True).start()
    try:
        run_trial(pub, trial_dir, p.object_name, p, log)
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("Interrupted"); log["interrupted"] = True
    finally:
        stop_flag.set()
        for t in readers + [logger]: t.join(timeout=5)
        for s in (s1, s2):                # Xense sensors are not needed any more: release them before anything heavy
            try: s.release()
            except Exception: pass  # noqa: BLE001
        log["events"] = [{"event": e, "wall_time": w_, "elapsed_s": el} for e, w_, el in events]
        log["cmd_messages"] = {"sent": pub.n_sent, "publish_calls": pub.n_calls}
        with open(os.path.join(trial_dir, "events.csv"), "w", newline="") as fh:
            ew = csv.writer(fh); ew.writerow(["event", "wall_time", "elapsed_s"]); ew.writerows(events)
        log["reference"] = {key: ({"failed": r.failed} if r.failed else r.info if r.ready else {"failed": "never set"})
                            for key, r in zip(SENSOR_KEYS, refs)}
        if store["t"]:
            seq = np.array(store["seq"])
            arrays = dict(index=np.stack(store["index"]), thumb=np.stack(store["thumb"]), t=np.array(store["t"]),
                          seq=seq, new=np.array(store["new"], dtype=np.uint8),
                          stale=np.array(store["stale"], dtype=np.int32), phase=np.array(store["phase"]))
            fseq = np.array(store["fseq"])
            arrays.update(fseq=fseq, force_new=np.array(store["force_new"], dtype=np.uint8), **field_init)
            if all(x is not None for x in store["index_flow"] + store["thumb_flow"]):
                arrays["index_flow"] = np.stack(store["index_flow"])
                arrays["thumb_flow"] = np.stack(store["thumb_flow"])
            np.savez_compressed(os.path.join(trial_dir, "taxels.npz"), **arrays)
            dur = store["t"][-1] - store["t"][0]
            log["tactile_rows"] = len(store["t"])
            log["tactile_rate_hz"] = round((len(store["t"]) - 1) / dur, 2) if dur > 0 else None
            if dur > 0:   # true per-sensor rates (new samples per second): field stream and force stream
                log["index_rate_hz"] = round(float(seq[-1, 0] - seq[0, 0]) / dur, 2)
                log["thumb_rate_hz"] = round(float(seq[-1, 1] - seq[0, 1]) / dur, 2)
                log["index_force_rate_hz"] = round(float(fseq[-1, 0] - fseq[0, 0]) / dur, 2)
                log["thumb_force_rate_hz"] = round(float(fseq[-1, 1] - fseq[0, 1]) / dur, 2)
        log["status"] = "tactile data saved; camera frames were still being written/converted when this was written"
        with open(os.path.join(trial_dir, "trial_info.json"), "w") as fh:
            json.dump(log, fh, indent=2)
        if store["t"]:
            rospy.loginfo("Tactile data saved. Finishing camera frames / PNGs - do NOT close this terminal ...")
        CONVERT_GO.set()
        for t in cams: t.join()
        if trig is not None: trig.join(timeout=3)
        if p.sync == "software":
            log["trigger"] = trig_meta
            fr = [cam_meta.get(c, {}).get("frames", 0) for c in ("cam1", "cam2")]
            log["camera_frames_match"] = (fr[0] == fr[1])
            if fr[0] != fr[1]:
                rospy.logwarn(f"[sync] frame counts differ: cam1 {fr[0]}, cam2 {fr[1]} (triggers {trig_meta.get('triggers')})")
        log["status"] = "complete"
        if not p.no_convert and not log.get("aborted"):
            rospy.loginfo("Converting camera frames to PNG (all CPU cores) ...")
            try:
                convert_frames(trial_dir, cam_meta, p.keep_raw)
            except Exception as e:  # noqa: BLE001
                rospy.logwarn(f"PNG conversion failed ({e}); raw frames kept")
        log["cameras"] = cam_meta
        log["qc"] = quality_check(log, store, cam_meta, p)
        with open(os.path.join(trial_dir, "trial_info.json"), "w") as fh:
            json.dump(log, fh, indent=2)
        rospy.loginfo("=" * 60)
        if log.get("aborted"):
            rospy.logerr(f"  TRIAL ABORTED: {log['aborted']} (folder kept with trial_info.json only)")
        rospy.loginfo(f"  DONE [{p.mode}]. Field rate index {log.get('index_rate_hz')} Hz, thumb {log.get('thumb_rate_hz')} Hz; "
                      f"force rate index {log.get('index_force_rate_hz')} Hz, thumb {log.get('thumb_force_rate_hz')} Hz; "
                      + ", ".join(f"{k}: {v.get('frames', 0)} frames @ {v.get('fps', '?')} fps, {v.get('saved', 0)} saved"
                                  for k, v in cam_meta.items()))
        if p.sync == "software" and trig_meta:
            rospy.loginfo(f"  Software trigger: {trig_meta.get('trigger_rate_hz')} Hz, cam1->cam2 spread mean "
                          f"{trig_meta.get('cam_spread_mean_ms')} ms (max {trig_meta.get('cam_spread_max_ms')} ms), "
                          f"frames match: {log.get('camera_frames_match')}")
        rospy.loginfo(f"  Settled: {log.get('settle_stable')} after {log.get('settle_s')} s")
        qc = log["qc"]
        if qc["pass"]:
            rospy.loginfo("  QC: PASS - keep this trial")
        else:
            rospy.logwarn("  QC: FAIL - " + "; ".join(qc["reasons"]))
        rospy.loginfo(f"  Data: {trial_dir}")
        rospy.loginfo("=" * 60)


if __name__ == "__main__":
    main()
