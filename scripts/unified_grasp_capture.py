#!/usr/bin/env python3
"""
unified_grasp_capture.py
Unified script that synchronizes:
  - Dual high-speed Hikrobot cameras (~156 fps)
  - Dual Xense tactile sensors (~50 Hz)
  - LEAP Hand robot control (30 Hz)
All with a shared t0 timestamp for post-experiment alignment.

Usage:
  /usr/share/bin/xensegui/bin/python3 unified_grasp_capture.py [object_name]
  Examples: grape (default), pen, earphone, cherry, marble
"""

import rospy
import argparse
import threading
import numpy as np
import time
import csv
import os
import sys
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from xensesdk import Sensor
from ctypes import memmove, byref, c_ubyte
import cv2
from queue import Queue
from collections import deque

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *

# ===== Xense Sensor IDs =====
INDEX_SENSOR_ID = "OG000961"
THUMB_SENSOR_ID = "OG000324"

# ===== Camera Serials =====
CAMERA_SERIALS = {
    "cam1": "DB1216927",
    "cam2": "DB1216964",
}

# ===== Camera Settings =====
CAM_WIDTH = 1280
CAM_HEIGHT = 1024
CAM_EXPOSURE_TIME = 1000.0  # microseconds
MAX_CAMERA_FRAMES = 2000  # ~15 seconds @ 133 fps

# ===== Grasp Parameters =====
CONTACT_DELTA_THRESHOLD = 1.0
TARGET_GRIP_DELTA = 4.0
MAX_GRIP_DELTA = 3.5
DIP_SQUEEZE_STEP = 0.05
MAX_DIP_EXTRA = 0.5

JOINT_NAMES = [f"joint_{i}" for i in range(16)]

# ===== Global shared timestamp =====
t0_global = None
stop_flag = threading.Event()

# ===== Tactile globals =====
index_delta = 0.0
thumb_delta = 0.0
index_baseline = 0.0
thumb_baseline = 0.0


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
    return [[x + (i/steps)*(y-x) for x, y in zip(a, b)] for i in range(steps+1)]


def make_msg(pos):
    msg = JointState()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.position = pos
    msg.velocity = []
    msg.effort = []
    return msg


def calibrate(s1, s2, samples=30):
    global index_baseline, thumb_baseline
    idx_vals, thm_vals = [], []
    for _ in range(samples):
        f1 = s1.selectSensorInfo(Sensor.OutputType.Force)
        f2 = s2.selectSensorInfo(Sensor.OutputType.Force)
        idx_vals.append(float(np.sum(np.abs(f1))))
        thm_vals.append(float(np.sum(np.abs(f2))))
        time.sleep(0.05)
    index_baseline = float(np.mean(idx_vals))
    thumb_baseline = float(np.mean(thm_vals))
    rospy.loginfo(f"[Calib] Index baseline={index_baseline:.3f}  Thumb baseline={thumb_baseline:.3f}")


def monitor_forces(s1, s2):
    global index_delta, thumb_delta
    while not rospy.is_shutdown() and not stop_flag.is_set():
        try:
            f1 = s1.selectSensorInfo(Sensor.OutputType.Force)
            f2 = s2.selectSensorInfo(Sensor.OutputType.Force)
            index_delta = max(0.0, float(np.sum(np.abs(f1))) - index_baseline)
            thumb_delta = max(0.0, float(np.sum(np.abs(f2))) - thumb_baseline)
        except Exception as e:
            rospy.logwarn(f"[Xsense] monitor_forces error: {e}")
        time.sleep(0.02)


def max_delta():
    return max(index_delta, thumb_delta)


# ===== Camera helper functions =====
def find_camera_by_serial(serial):
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        raise RuntimeError(f"EnumDevices failed, ret={ret}")

    for i in range(deviceList.nDeviceNum):
        dev_info = deviceList.pDeviceInfo[i].contents
        if dev_info.nTLayerType == MV_USB_DEVICE:
            sn = bytes(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode().rstrip('\x00')
            if sn == serial:
                return dev_info
    raise RuntimeError(f"Camera with serial {serial} not found")


def background_writer(write_queue, output_dir, cam_name):
    """Background thread that writes frames to disk"""
    frame_count = 0
    while True:
        item = write_queue.get()
        if item is None:  # Poison pill to stop
            break

        img, frame_num = item
        fname = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
        cv2.imwrite(fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        frame_count += 1
        write_queue.task_done()

    print(f"[{cam_name}] Background writer finished: {frame_count} frames written")


def record_camera(cam_name, serial, run_ts, object_name):
    global t0_global
    output_dir = os.path.expanduser(f"~/{object_name}_unified_{run_ts}/camera_{cam_name}")
    csv_path = os.path.expanduser(f"~/{object_name}_unified_{run_ts}/camera_{cam_name}_timestamps.csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{cam_name}] Looking for camera {serial}...")
    dev_info = find_camera_by_serial(serial)

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(dev_info)
    if ret != 0:
        print(f"[{cam_name}] CreateHandle failed, ret={ret}")
        return

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"[{cam_name}] OpenDevice failed, ret={ret}")
        return

    # Configure camera for maximum FPS
    cam.MV_CC_SetIntValue("Width", CAM_WIDTH)
    cam.MV_CC_SetIntValue("Height", CAM_HEIGHT)
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    cam.MV_CC_SetFloatValue("ExposureTime", CAM_EXPOSURE_TIME)

    # Disable frame rate limiter
    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)

    # Optimize transfer settings
    cam.MV_CC_SetEnumValue("TriggerMode", 0)  # Free-running
    cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)  # Packet size for GigE

    print(f"[{cam_name}] Camera configured for max FPS")

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"[{cam_name}] StartGrabbing failed, ret={ret}")
        return

    # Create background writer thread with queue
    write_queue = Queue(maxsize=2000)  # Buffer up to 2000 frames (~2.6GB RAM per camera)
    writer_thread = threading.Thread(target=background_writer, args=(write_queue, output_dir, cam_name))
    writer_thread.start()

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "wall_time", "elapsed_s"])

    stFrame = MV_FRAME_OUT()
    frame_count = 0
    fps_start = time.time()

    print(f"[{cam_name}] Camera ready. Waiting for t0_global...")
    # Wait for grasp sequence to set t0_global
    while t0_global is None and not stop_flag.is_set():
        time.sleep(0.01)

    print(f"[{cam_name}] Recording started at t0={t0_global}")
    fps_start = time.time()  # Reset after waiting

    try:
        while not stop_flag.is_set() and (frame_count < MAX_CAMERA_FRAMES):
            ret = cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == 0:
                now = time.time()

                # Print FPS every 100 frames
                if frame_count > 0 and frame_count % 100 == 0:
                    elapsed = now - fps_start
                    actual_fps = frame_count / elapsed
                    print(f"[{cam_name}] {frame_count} frames @ {actual_fps:.1f} fps")

                # Copy frame data to numpy array
                buf = (c_ubyte * stFrame.stFrameInfo.nFrameLen)()
                memmove(byref(buf), stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                arr = np.frombuffer(buf, dtype=np.uint8)
                img = arr.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth)).copy()

                # Queue for background writing (non-blocking)
                write_queue.put((img, frame_count))

                csv_writer.writerow([frame_count, now, round(now - t0_global, 6)])
                csv_file.flush()

                frame_count += 1
                cam.MV_CC_FreeImageBuffer(stFrame)
    except Exception as e:
        print(f"[{cam_name}] Error: {e}")
    finally:
        # Stop background writer
        write_queue.put(None)  # Poison pill
        writer_thread.join()  # Wait for all writes to complete

        csv_file.close()
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        print(f"[{cam_name}] Done. {frame_count} frames saved to {output_dir}")


def run_grasp_and_tactile(s1, s2, run_ts, object_name):
    global t0_global

    pub = rospy.Publisher("/leaphand_node/cmd_allegro", JointState, queue_size=10)
    rate = rospy.Rate(30)

    OPEN = open_pose()
    PINCH = grape_pinch_pose()

    # Give cameras time to initialize (but don't start recording yet)
    rospy.loginfo("[Unified] Waiting 3 seconds for cameras to initialize...")
    for _ in range(90):
        if rospy.is_shutdown(): return
        rate.sleep()

    # Step 1 — Open hand
    rospy.loginfo("[1] Opening hand...")
    for _ in range(60):
        if rospy.is_shutdown(): return
        pub.publish(make_msg(OPEN))
        rate.sleep()

    # Calibrate tactile sensors AFTER opening hand (user's hand should be clear now)
    rospy.loginfo("[Calib] Calibrating tactile sensors NOW — do NOT touch...")
    calibrate(s1, s2)

    # Step 2 — Move to pinch pose
    rospy.loginfo(f"[2] Moving to {object_name} pinch pose...")
    for frame in interpolate(OPEN, PINCH, steps=80):
        if rospy.is_shutdown(): return
        pub.publish(make_msg(frame))
        rate.sleep()

    # Step 3 — Hold at pinch pose, place object
    rospy.loginfo(f"[3] Holding pinch pose — PLACE {object_name.upper()} NOW (3 sec)")
    for _ in range(90):
        if rospy.is_shutdown(): return
        pub.publish(make_msg(PINCH))
        rate.sleep()

    # SET GLOBAL T0 - START RECORDING NOW (right before squeeze)
    t0_global = time.time()
    rospy.loginfo(f"[Unified] t0_global set: {t0_global}")
    rospy.loginfo("[Unified] Camera recording started!")

    # Step 4 — Squeeze
    rospy.loginfo("[4] Squeezing DIP joints (index + thumb)...")
    held_pose = list(PINCH)
    extra_dip = 0.0
    squeeze_steps = int(MAX_DIP_EXTRA / DIP_SQUEEZE_STEP)

    squeeze_csv_path = os.path.expanduser(f"~/{object_name}_unified_{run_ts}/squeeze_curve.csv")
    squeeze_csv_file = open(squeeze_csv_path, 'w', newline='')
    squeeze_csv_writer = csv.writer(squeeze_csv_file)
    squeeze_csv_writer.writerow(['step', 'extra_dip_rad', 'index_force_N', 'thumb_force_N'])

    for step in range(squeeze_steps):
        if rospy.is_shutdown(): return
        extra_dip += DIP_SQUEEZE_STEP
        held_pose[3] = PINCH[3] + extra_dip
        held_pose[15] = PINCH[15] + extra_dip
        pub.publish(make_msg(held_pose))
        time.sleep(0.15)
        rospy.loginfo(f"    Squeeze {step} — extra={extra_dip:.2f} — Index={index_delta:.2f} Thumb={thumb_delta:.2f}")
        squeeze_csv_writer.writerow([step, round(extra_dip, 4), round(index_delta, 4), round(thumb_delta, 4)])
        squeeze_csv_file.flush()
        if max_delta() >= TARGET_GRIP_DELTA:
            break
        if max_delta() >= MAX_GRIP_DELTA:
            rospy.loginfo(f"[4] MAX safety force reached!")
            break

    squeeze_csv_file.close()

    # Step 5 — Start detailed tactile logging + heatmaps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap_dir = os.path.expanduser(f"~/{object_name}_unified_{run_ts}/heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    tactile_csv_path = os.path.expanduser(f"~/{object_name}_unified_{run_ts}/tactile_data.csv")
    tactile_csv_file = open(tactile_csv_path, 'w', newline='')
    tactile_csv_writer = csv.writer(tactile_csv_file)
    tactile_csv_writer.writerow(['frame', 'wall_time', 'elapsed_s',
                                   'index_force_N', 'index_contact_x', 'index_contact_y', 'index_contact_area', 'index_shear_N', 'index_pressure',
                                   'thumb_force_N', 'thumb_contact_x', 'thumb_contact_y', 'thumb_contact_area', 'thumb_shear_N', 'thumb_pressure'])

    data_frame = [0]
    hm_frame = [0]
    last_thumb = [None]
    thumb_stale_count = [0]

    def fast_tactile_loop():
        while not rospy.is_shutdown() and not stop_flag.is_set():
            loop_start = time.time()
            try:
                f1m, rf1m = s1.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)
                f2m, rf2m = s2.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)

                idx_area = int((abs(f1m[:, :, 2]) > 1e-4).sum())
                thm_area = int((abs(f2m[:, :, 2]) > 1e-4).sum())
                idx_shear = float(np.sum(np.sqrt(f1m[:, :, 0]**2 + f1m[:, :, 1]**2)))
                thm_shear = float(np.sum(np.sqrt(f2m[:, :, 0]**2 + f2m[:, :, 1]**2)))
                idx_pressure = abs(rf1m[2]) / idx_area if idx_area > 0 else 0.0
                thm_pressure = abs(rf2m[2]) / thm_area if thm_area > 0 else 0.0

                now = time.time()
                thumb_val = round(abs(rf2m[2]), 6)

                # Stale detection
                if last_thumb[0] is not None and thumb_val == last_thumb[0]:
                    thumb_stale_count[0] += 1
                    if thumb_stale_count[0] == 25:
                        rospy.logwarn("[Xsense] Thumb sensor reading unchanged for 25 samples — may be stalled.")
                else:
                    thumb_stale_count[0] = 0
                last_thumb[0] = thumb_val

                tactile_csv_writer.writerow([
                    data_frame[0], round(now, 6), round(now - t0_global, 6),
                    round(abs(rf1m[2]), 4), round(rf1m[3], 3), round(rf1m[4], 3), idx_area, round(idx_shear, 4), round(idx_pressure, 5),
                    round(abs(rf2m[2]), 4), round(rf2m[3], 3), round(rf2m[4], 3), thm_area, round(thm_shear, 4), round(thm_pressure, 5)
                ])
                tactile_csv_file.flush()
                data_frame[0] += 1

                fast_tactile_loop.latest = (f1m, f2m, rf1m, rf2m)
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[Xsense] fast_tactile_loop error: {e}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, 0.02 - elapsed))  # ~50 Hz

    def slow_heatmap_loop():
        while not rospy.is_shutdown() and not stop_flag.is_set():
            try:
                if hasattr(fast_tactile_loop, "latest"):
                    f1m, f2m, rf1m, rf2m = fast_tactile_loop.latest
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    fig.suptitle("Unified Grape Grasp — Frame %d" % hm_frame[0], fontsize=13)
                    axes[0].imshow(np.linalg.norm(f1m, axis=2), cmap="jet", vmin=0, interpolation="nearest")
                    axes[0].set_title("Index Tip\n%.3fN" % float(np.sum(np.abs(f1m))))
                    axes[1].imshow(np.linalg.norm(f2m, axis=2), cmap="jet", vmin=0, interpolation="nearest")
                    axes[1].set_title("Thumb Tip\n%.3fN" % float(np.sum(np.abs(f2m))))
                    plt.tight_layout()
                    plt.savefig(os.path.join(heatmap_dir, f"heatmap_{hm_frame[0]:06d}.png"))
                    plt.close()
                    hm_frame[0] += 1
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[Xsense] slow_heatmap_loop error: {e}")
            time.sleep(0.5)

    threading.Thread(target=fast_tactile_loop, daemon=True).start()
    threading.Thread(target=slow_heatmap_loop, daemon=True).start()

    rospy.loginfo(f"[5] Holding {object_name}! Tactile + heatmap logging started.")
    rospy.loginfo(f"    Data dir: ~/{object_name}_unified_{run_ts}/")

    # Hold until camera capture completes (2000 frames @ ~135 fps = ~15 seconds)
    hold_duration = (MAX_CAMERA_FRAMES / 135.0) + 1.0  # ~15.8 seconds
    hold_start = time.time()

    while not rospy.is_shutdown() and (time.time() - hold_start < hold_duration):
        pub.publish(make_msg(held_pose))
        try:
            f1r, rf1 = s1.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)
            f2r, rf2 = s2.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.ForceResultant)
            rospy.loginfo_throttle(2.0, "    Holding - Index=(%.2f,%.2f) %.3fN | Thumb=(%.2f,%.2f) %.3fN" %
                                   (rf1[3], rf1[4], abs(rf1[2]), rf2[3], rf2[4], abs(rf2[2])))
        except:
            rospy.loginfo_throttle(2.0, "    Holding - Index=%.2f  Thumb=%.2f" % (index_delta, thumb_delta))
        rate.sleep()

    rospy.loginfo("[Unified] Hold complete. Stopping all recording...")
    stop_flag.set()
    tactile_csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="Unified grasp capture with tactile + camera sync")
    parser.add_argument('object_name', nargs='?', default='grape',
                        help='Object being grasped (grape, pen, earphone, cherry, marble, etc.)')
    args = parser.parse_args()
    object_name = args.object_name

    rospy.init_node("unified_grasp_capture", anonymous=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    rospy.loginfo("="*60)
    rospy.loginfo(f"  UNIFIED {object_name.upper()} GRASP CAPTURE")
    rospy.loginfo("  Cameras: 6cm from object, side/oblique angles")
    rospy.loginfo("  Run timestamp: %s" % run_ts)
    rospy.loginfo("="*60)

    # Initialize sensors
    rospy.loginfo("[Xsense] Initializing tactile sensors...")
    s1 = Sensor.create(INDEX_SENSOR_ID)
    s2 = Sensor.create(THUMB_SENSOR_ID)
    # Calibration moved to after hand opens in run_grasp_and_tactile()

    # Start force monitoring thread
    t_monitor = threading.Thread(target=monitor_forces, args=(s1, s2), daemon=True)
    t_monitor.start()

    # Start camera threads
    cam_threads = []
    for cam_name, serial in CAMERA_SERIALS.items():
        t = threading.Thread(target=record_camera, args=(cam_name, serial, run_ts, object_name))
        t.start()
        cam_threads.append(t)

    # Run grasp sequence
    try:
        run_grasp_and_tactile(s1, s2, run_ts, object_name)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("[Unified] Interrupted by user.")
        stop_flag.set()

    # Wait for all threads
    for t in cam_threads:
        t.join()

    s1.release()
    s2.release()

    rospy.loginfo("="*60)
    rospy.loginfo("  CAPTURE COMPLETE")
    rospy.loginfo("  All data saved to: ~/grape_unified_%s/" % run_ts)
    rospy.loginfo("="*60)


if __name__ == "__main__":
    main()
