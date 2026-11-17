#!/usr/bin/env python3
import rospy
import threading
import numpy as np
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from xensesdk import Sensor

INDEX_SENSOR_ID = "OG000961"
THUMB_SENSOR_ID = "OG000324"

CONTACT_DELTA_THRESHOLD = 1.0
TARGET_GRIP_DELTA       = 4.0
MAX_GRIP_DELTA          = 3.5
DIP_SQUEEZE_STEP        = 0.05
MAX_DIP_EXTRA           = 0.5

JOINT_NAMES = [
    "joint_0",  "joint_1",  "joint_2",  "joint_3",
    "joint_4",  "joint_5",  "joint_6",  "joint_7",
    "joint_8",  "joint_9",  "joint_10", "joint_11",
    "joint_12", "joint_13", "joint_14", "joint_15",
]

def open_pose():
    return [0.0] * 16


def grape_pinch_pose():
    return [
         0.057,  1.800,  0.059, -0.268,   # Index
        -0.028, -0.005, -0.004, -0.000,   # Middle
        -0.028,  0.047, -0.086,  0.041,   # Ring
         1.533,  0.041,  0.434, -0.426,   # Thumb
    ]


def interpolate(a, b, steps=80):
    return [[x + (i/steps)*(y-x) for x,y in zip(a,b)] for i in range(steps+1)]

def make_msg(pos):
    msg = JointState()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.position = pos
    msg.velocity = []
    msg.effort = []
    return msg

index_delta    = 0.0
thumb_delta    = 0.0
index_baseline = 0.0
thumb_baseline = 0.0

def calibrate(s1, s2, samples=30):
    global index_baseline, thumb_baseline
    rospy.loginfo("[Calib] Calibrating — do NOT touch sensors...")
    idx_vals, thm_vals = [], []
    for _ in range(samples):
        f1 = s1.selectSensorInfo(Sensor.OutputType.Force)
        f2 = s2.selectSensorInfo(Sensor.OutputType.Force)
        idx_vals.append(float(np.sum(np.abs(f1))))
        thm_vals.append(float(np.sum(np.abs(f2))))
        time.sleep(0.05)
    index_baseline = float(np.mean(idx_vals))
    thumb_baseline  = float(np.mean(thm_vals))
    rospy.loginfo(f"[Calib] Index={index_baseline:.3f}  Thumb={thumb_baseline:.3f}")

def monitor_forces(s1, s2):
    global index_delta, thumb_delta
    while not rospy.is_shutdown():
        try:
            f1 = s1.selectSensorInfo(Sensor.OutputType.Force)
            f2 = s2.selectSensorInfo(Sensor.OutputType.Force)
            index_delta = max(0.0, float(np.sum(np.abs(f1))) - index_baseline)
            thumb_delta  = max(0.0, float(np.sum(np.abs(f2))) - thumb_baseline)
        except Exception as e:
            rospy.logwarn(f"[Xsense] error: {e}")
        time.sleep(0.02)

def max_delta():
    return max(index_delta, thumb_delta)

def run():
    rospy.init_node("leap_xsense_grasp", anonymous=True)
    pub  = rospy.Publisher("/leaphand_node/cmd_allegro", JointState, queue_size=10)
    rate = rospy.Rate(30)

    rospy.loginfo("[Xsense] Initializing sensors...")
    s1 = Sensor.create(INDEX_SENSOR_ID)
    s2 = Sensor.create(THUMB_SENSOR_ID)
    calibrate(s1, s2)

    t = threading.Thread(target=monitor_forces, args=(s1, s2), daemon=True)
    t.start()

    OPEN  = open_pose()
    PINCH = grape_pinch_pose()

    # Step 1 — Open hand
    rospy.loginfo("[1] Opening hand...")
    for _ in range(60):
        if rospy.is_shutdown(): return
        pub.publish(make_msg(OPEN))
        rate.sleep()

    # Step 2 — Move to grape pinch pose
    rospy.loginfo("[2] Moving to grape pinch pose...")
    for frame in interpolate(OPEN, PINCH, steps=80):
        if rospy.is_shutdown(): return
        pub.publish(make_msg(frame))
        rate.sleep()

    # Step 3 — Hold at pinch pose, place grape
    rospy.loginfo("[3] Holding pinch pose — PLACE GRAPE NOW (3 sec)")
    # Print matrix shape info once
    _f1_test = s1.selectSensorInfo(Sensor.OutputType.Force)
    rospy.loginfo("Sensor matrix shape: %s (rows x cols x force_components)" % str(_f1_test.shape))
    for _ in range(90):
        if rospy.is_shutdown(): return
        pub.publish(make_msg(PINCH))
        rate.sleep()

    # Step 4 — Squeeze joint_3 + joint_15 for grip force
    rospy.loginfo("[4] Squeezing joint_3 (index DIP) + joint_15 (thumb DIP)...")
    held_pose  = list(PINCH)
    extra_dip  = 0.0
    squeeze_steps = int(MAX_DIP_EXTRA / DIP_SQUEEZE_STEP)

    for step in range(squeeze_steps):
        if rospy.is_shutdown(): return
        extra_dip    += DIP_SQUEEZE_STEP
        held_pose[3]  = PINCH[3]  + extra_dip
        held_pose[15] = PINCH[15] + extra_dip
        pub.publish(make_msg(held_pose))
        time.sleep(0.15)
        rospy.loginfo(f"    Squeeze {step} — extra={extra_dip:.2f} — Index_delta={index_delta:.2f}  Thumb_delta={thumb_delta:.2f}")
        if max_delta() >= TARGET_GRIP_DELTA:
            break
        if max_delta() >= MAX_GRIP_DELTA:
            rospy.loginfo(f"[4] MAX safety force reached! Stopping.")
            break

    # Start heatmap saving thread
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os as _os
    _os.makedirs(os.path.expanduser("~/force_heatmaps"), exist_ok=True)
    _hm_frame = [0]

    # CSV logging
    import csv as _csv
    _csv_file = open(os.path.expanduser("~/grasp_force_log.csv"), 'w', newline='')
    _csv_writer = _csv.writer(_csv_file)
    _csv_writer.writerow(['frame', 'time_s', 'index_force_N', 'index_contact_x', 'index_contact_y', 'thumb_force_N', 'thumb_contact_x', 'thumb_contact_y'])
    _t0 = time.time()

    def save_heatmaps():
        while not rospy.is_shutdown():
            try:
                f1m, rf1m = s1.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)
                f2m, rf2m = s2.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)
                # Write CSV row
                _csv_writer.writerow([
                    _hm_frame[0],
                    round(time.time() - _t0, 2),
                    round(abs(rf1m[2]), 4),
                    round(rf1m[3], 3),
                    round(rf1m[4], 3),
                    round(abs(rf2m[2]), 4),
                    round(rf2m[3], 3),
                    round(rf2m[4], 3)
                ])
                _csv_file.flush()
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle("Grape Grasp Force — Frame %d" % _hm_frame[0], fontsize=13)
                axes[0].imshow(np.linalg.norm(f1m, axis=2), cmap="jet", vmin=0, interpolation="nearest")
                axes[0].set_title("Index Tip\n%.3fN" % float(np.sum(np.abs(f1m))))
                axes[1].imshow(np.linalg.norm(f2m, axis=2), cmap="jet", vmin=0, interpolation="nearest")
                axes[1].set_title("Thumb Tip\n%.3fN" % float(np.sum(np.abs(f2m))))
                plt.tight_layout()
                plt.savefig(os.path.expanduser("~/force_heatmaps/grasp_%04d.png") % _hm_frame[0])
                plt.close()
                _hm_frame[0] += 1
                time.sleep(0.5)
            except:
                time.sleep(0.5)

    _hm_thread = threading.Thread(target=save_heatmaps, daemon=True)
    _hm_thread.start()
    rospy.loginfo("Heatmaps saving to ~/force_heatmaps/")

    # Step 5 — Hold forever
    rospy.loginfo(f"[5] Holding grape! joint_3={held_pose[3]:.3f}  joint_15={held_pose[15]:.3f}")
    while not rospy.is_shutdown():
        pub.publish(make_msg(held_pose))
        try:
            f1r, rf1 = s1.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)
            f2r, rf2 = s2.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)
            rospy.loginfo_throttle(2.0, "    Holding - Index=(%.2f,%.2f) %.3fN | Thumb=(%.2f,%.2f) %.3fN" % (rf1[3],rf1[4],abs(rf1[2]),rf2[3],rf2[4],abs(rf2[2])))
        except:
            rospy.loginfo_throttle(2.0, "    Holding - Index=%.2f  Thumb=%.2f" % (index_delta, thumb_delta))
        # Print raw force matrix every 5 seconds
        try:
            import numpy as _np
            _f1m = s1.selectSensorInfo(Sensor.OutputType.Force)
            _f2m = s2.selectSensorInfo(Sensor.OutputType.Force)
            rospy.loginfo_throttle(5.0, "\n--- INDEX SENSOR MATRIX shape=%s Fz channel (8x8 sample) ---\n%s" % (str(_f1m.shape), str(_np.round(_f1m[:8,:8,2], 4))))
            rospy.loginfo_throttle(5.0, "\n--- THUMB SENSOR MATRIX shape=%s Fz channel (8x8 sample) ---\n%s" % (str(_f2m.shape), str(_np.round(_f2m[:8,:8,2], 4))))
        except Exception as _me:
            pass
        rate.sleep()

    s1.release()
    s2.release()

if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        pass
